import copy
import io
import json
import uuid
import wave
from contextlib import contextmanager

import pytest

from proto_gen import stream_pb2 as pb
from drsynth_common.final_tracks import (
    ContractError, TranscriptResult, canonical_json, identities, plan_headers,
    plan_proto, read_plan, validate_recording,
)
from finalizer_worker.track_processing import process_recording


@pytest.fixture
def recording():
    tenant, session, plan_id, source = (str(uuid.uuid4()) for _ in range(4))
    plan = dict(schema_version=1, tenant_id=tenant, session_id=session, plan_id=plan_id,
                final_tracks=[dict(track_id="whisperx", profile_id="whisperx-medium-final-r1", primary=True),
                              dict(track_id="test-secondary", profile_id="test-final-r1", primary=False)])
    event = pb.RecordingFinished(
        tenant_id=tenant, session_id=session, duration_s=1, sample_rate=16000,
        recording_url=f"s3://recordings/recordings/{tenant}/{session}/{source}.wav",
        lang="en", created_at_ns=1,
        source=pb.AudioArtifact(artifact_id=source, sha256="a" * 64, size_bytes=32044,
                                sample_rate=16000, sample_count=16000, media_type="audio/wav"),
        final_plan=plan_proto(plan))
    event.source.storage_uri = event.recording_url
    return event, plan, plan_headers(plan) + [("x-outputs", b"final"), ("x-store-recording", b"true")]


def test_plan_roundtrip_and_replay_identity(recording):
    event, plan, headers = recording
    assert read_plan(headers, event.tenant_id, event.session_id) == plan
    assert validate_recording(event, headers) == plan
    first = identities(plan, event.source, plan["final_tracks"][0])
    assert first == identities(copy.deepcopy(plan), event.source, plan["final_tracks"][0])
    assert first != identities(plan, event.source, plan["final_tracks"][1])
    rerun = dict(plan, plan_id=str(uuid.uuid4()))
    assert first != identities(rerun, event.source, plan["final_tracks"][0])


@pytest.mark.parametrize("mutation", ["owner", "duplicate", "primary", "unknown", "version", "header", "typed"])
def test_reject_ambiguous_or_tampered_plan(recording, mutation):
    event, plan, headers = recording
    if mutation == "owner":
        event.tenant_id = str(uuid.uuid4())
    elif mutation == "duplicate":
        headers.append(headers[0])
    elif mutation == "header":
        headers[2] = ("x-final-track-ids", b"test-secondary")
    elif mutation == "typed":
        event.final_plan.plan_id = str(uuid.uuid4())
    else:
        if mutation == "primary":
            plan["final_tracks"][1]["primary"] = True
        elif mutation == "version":
            plan["schema_version"] = 2
        else:
            plan["endpoint"] = "http://untrusted.invalid"
        headers[0] = ("x-asr-plan", canonical_json(plan))
    with pytest.raises(ContractError):
        validate_recording(event, headers)


class MemoryStore:
    def __init__(self, path):
        self.path = path
        self.downloads = 0

    def validate_source(self, *args):
        pass

    @contextmanager
    def audio_file(self, *args):
        self.downloads += 1
        yield self.path

class FakeTrack:
    def __init__(self, failures=0, profile="whisperx-medium-final-r1"):
        self.calls, self.failures, self.profile = 0, failures, profile

    def describe(self):
        return {"profile_id": self.profile}

    def process(self, audio, context):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("sensitive provider exception must not reach event")
        return TranscriptResult("fixture", [{"text": "fixture", "start_s": 0.0, "end_s": 0.5}],
                                "en", {"segment_timestamps": True}, {"fake": True})


def test_retry_replay_has_stable_identity_and_inline_content(recording, tmp_path):
    event, _, headers = recording
    store, track = MemoryStore(tmp_path / "fixture.wav"), FakeTrack(failures=1)
    kwargs = dict(track_id="whisperx", profile_id=track.profile, provider=track, store=store)
    first = process_recording(event, headers, **kwargs)
    assert first.schema_version == 2 and first.status == "succeeded"
    assert first.full_text == "fixture" and first.segments[0].text == "fixture"
    assert track.calls == 2
    replay = process_recording(event, headers, **kwargs)
    assert replay.result_id == first.result_id and replay.full_text == first.full_text
    assert track.calls == 3 and store.downloads == 2
    failed = FakeTrack(failures=10, profile="test-final-r1")
    second = process_recording(event, headers, track_id="test-secondary", profile_id=failed.profile,
                               provider=failed, store=store)
    assert second.status == "failed" and not second.full_text and not second.segments
    assert second.error_code == "inference_failed" and failed.calls == 3
    assert second.result_id != first.result_id
    assert b"sensitive" not in second.SerializeToString()


def test_unselected_track_does_not_download_or_infer(recording, tmp_path):
    event, _, headers = recording
    store, track = MemoryStore(tmp_path), FakeTrack()
    assert process_recording(event, headers, track_id="unselected", profile_id=track.profile,
                             provider=track, store=store) is None
    assert track.calls == store.downloads == 0


def test_retention_rejected_before_io(recording, tmp_path):
    event, _, headers = recording
    headers[-1] = ("x-store-recording", b"false")
    store, track = MemoryStore(tmp_path), FakeTrack()
    with pytest.raises(ContractError, match="unsupported_retention"):
        process_recording(event, headers, track_id="whisperx", profile_id=track.profile,
                          provider=track, store=store)
    assert store.downloads == 0


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, **kwargs):
        from botocore.exceptions import ClientError
        if (Bucket, Key) in self.objects and IfNoneMatch == "*":
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Bucket, Key] = Body.read() if hasattr(Body, "read") else Body
        return {"VersionId": "fixture-version"}

    def get_object(self, *, Bucket, Key, **kwargs):
        from botocore.exceptions import ClientError
        from botocore.response import StreamingBody
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data = self.objects[Bucket, Key]
        return {"Body": StreamingBody(io.BytesIO(data), len(data)), "ContentLength": len(data)}


def test_only_source_audio_uses_s3_and_temporary_files_are_removed(recording, tmp_path):
    from drsynth_common.final_track_artifacts import S3Artifacts
    event, plan, headers = recording
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(32000))
    client = FakeS3()
    store = S3Artifacts(client, "recordings")
    source = store.put_audio(path, event.tenant_id, event.session_id, event.source.artifact_id)
    event.source.CopyFrom(source)
    with store.audio_file(source, event.tenant_id, event.session_id) as temporary:
        assert temporary.exists()
    assert not temporary.exists()
    track = FakeTrack()
    kwargs = dict(track_id="whisperx", profile_id=track.profile, provider=track, store=store)
    before = set(client.objects)
    accepted = process_recording(event, headers, **kwargs)
    assert accepted.status == "succeeded" and accepted.full_text == "fixture"
    assert process_recording(event, headers, **kwargs).result_id == accepted.result_id
    assert track.calls == 2 and set(client.objects) == before
    key = store.validate_source(source, event.tenant_id, event.session_id)
    client.objects["recordings", key] = bytes(32044)
    with pytest.raises(ContractError, match="audio_digest_mismatch"):
        with store.audio_file(source, event.tenant_id, event.session_id):
            pytest.fail("corrupt source was accepted")


def test_kafka_publication_failure_repeats_inference_with_same_result_id(recording, tmp_path, monkeypatch):
    from finalizer_worker import track_worker
    event, _, headers = recording
    trace = ("traceparent", b"00-00000000000000000000000000000001-0000000000000002-01")
    headers.append(trace)
    class Message:
        def value(self): return event.SerializeToString()
        def headers(self): return headers
        def key(self): return event.session_id.encode()
    store, track, published = MemoryStore(tmp_path), FakeTrack(), []
    def publish(_producer, **kwargs):
        assert trace in kwargs["headers"]
        published.append((kwargs["topic"], kwargs["value"]))
        if len(published) == 1:
            raise RuntimeError("uncertain_publication")
    monkeypatch.setattr(track_worker, "produce_acked", publish)
    kwargs = dict(producer=None, track_id="whisperx", profile_id=track.profile, provider=track, store=store)
    with pytest.raises(RuntimeError, match="uncertain_publication"):
        track_worker.handle_message(Message(), **kwargs)
    track_worker.handle_message(Message(), **kwargs)
    assert track.calls == 2 and len(published) == 2
    assert {topic for topic, _ in published} == {"transcripts.final-tracks"}
    assert len({pb.FinalTrackResult.FromString(value).result_id for _, value in published}) == 1


def test_storage_outage_leaves_no_terminal_outcome(recording, tmp_path, monkeypatch):
    event, _, headers = recording
    store, track = MemoryStore(tmp_path), FakeTrack()
    def unavailable(*args):
        raise ConnectionError("fixture outage")
    monkeypatch.setattr(store, "audio_file", unavailable)
    with pytest.raises(ConnectionError):
        process_recording(event, headers, track_id="whisperx", profile_id=track.profile,
                          provider=track, store=store)
    assert track.calls == 0


def test_missing_timing_and_speaker_enrichment_remains_explicit():
    from finalizer_worker.track_processing import normalize_transcript
    result = TranscriptResult("fixture", [{"text": "fixture"}], "en",
                              {"word_timestamps": False, "speaker_labels": False}, {},
                              ["alignment_unavailable", "diarization_unavailable"])
    normalized = normalize_transcript(result, 1)
    assert normalized["segments"] == [{"text": "fixture"}]
    assert normalized["degradations"] == result.degradations


def test_recorder_propagates_plan_and_typed_source(recording, monkeypatch):
    from recorder_worker import recorder_worker as recorder
    event, plan, _ = recording
    class Writer:
        source = event.source
        def close_and_get_url(self): return event.recording_url
    rec = recorder.SessionRecording(event.session_id, event.tenant_id, 16000, "en", Writer(),
                                     final_plan=plan, total_samples=16000)
    sent = []
    monkeypatch.setattr(recorder, "produce_acked", lambda _producer, **kw: sent.append(kw))
    recorder.finalize_session(event.session_id, rec, None)
    propagated = pb.RecordingFinished.FromString(sent[0]["value"])
    assert validate_recording(propagated, sent[0]["headers"]) == plan
    assert propagated.source == event.source

@pytest.mark.parametrize("kind", ["silence", "text-only", "oversized"])
def test_inline_output_bounds_and_availability(recording, tmp_path, kind):
    event, _, headers = recording
    class Provider(FakeTrack):
        def process(self, audio, context):
            if kind == "silence":
                return TranscriptResult("", [], "en", {}, {})
            if kind == "text-only":
                return TranscriptResult("fixture", [{"text": "fixture"}], "en", {}, {},
                                        ["alignment_unavailable"])
            return TranscriptResult("x" * 900000, [], "en", {}, {})
    track = Provider()
    result = process_recording(event, headers, track_id="whisperx", profile_id=track.profile,
                               provider=track, store=MemoryStore(tmp_path))
    assert not result.segment_timestamps and not result.word_timestamps and not result.speaker_labels
    assert result.ByteSize() <= 900000
    if kind == "oversized":
        assert result.status == "failed" and result.error_code == "result_too_large"
        assert not result.full_text and not result.segments
    else:
        assert result.status == "succeeded"
        if kind == "silence":
            assert not result.full_text and not result.segments
        else:
            assert result.full_text == result.segments[0].text == "fixture"


def test_inline_proto_reserves_retired_fields():
    from google.protobuf.descriptor_pb2 import DescriptorProto
    descriptor = DescriptorProto()
    pb.FinalTrackResult.DESCRIPTOR.CopyToProto(descriptor)
    assert len(descriptor.field) == 18
    assert {f.number for f in descriptor.field}.isdisjoint({4, 7, 8, 10, 11, 14, 15, 16, 22, 26})
    assert {f.name: f.number for f in descriptor.field}["full_text"] == 27
    assert {f.name: f.number for f in descriptor.field}["segments"] == 28
    assert set(descriptor.reserved_name) == {"stage", "run_id", "attempt_id", "unit_id", "revision",
                                             "result_uri", "result_sha256", "primary", "provenance_json"}
