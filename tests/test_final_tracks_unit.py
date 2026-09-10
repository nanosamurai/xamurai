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
        self.path, self.outcomes, self.transcripts = path, {}, {}
        self.downloads = 0

    def validate_source(self, *args):
        pass

    def read_outcome(self, event):
        return self.outcomes.get(event.result_id)

    @contextmanager
    def audio_file(self, *args):
        self.downloads += 1
        yield self.path

    def put_transcript(self, event, transcript):
        self.transcripts[event.result_id] = transcript
        return "s3://recordings/result.json", "b" * 64

    def publish_once(self, event, legacy):
        value = (pb.FinalTrackResult.FromString(event.SerializeToString()),
                 event.SerializeToString(deterministic=True), legacy)
        return self.outcomes.setdefault(event.result_id, value)


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


def test_retry_replay_and_primary_are_independent(recording, tmp_path):
    event, plan, headers = recording
    store, track = MemoryStore(tmp_path / "fixture.wav"), FakeTrack(failures=1)
    kwargs = dict(track_id="whisperx", profile_id=track.profile, provider=track, store=store)
    accepted = process_recording(event, headers, **kwargs)
    assert track.calls == 2
    assert accepted[0].status == "succeeded"
    assert pb.SessionTranscript.FromString(accepted[2]).full_text == "fixture"
    assert process_recording(event, headers, **kwargs) == accepted
    assert track.calls == 2 and store.downloads == 1
    failed = FakeTrack(failures=10, profile="test-final-r1")
    second = process_recording(event, headers, track_id="test-secondary", profile_id=failed.profile,
                               provider=failed, store=store)
    assert second[0].status == "failed" and second[2] is None and failed.calls == 3
    assert second[0].result_id != accepted[0].result_id
    assert "sensitive" not in second[0].provenance_json
    assert len(store.outcomes) == 2


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


def test_s3_manifest_winner_digest_and_temporary_cleanup(recording, tmp_path):
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
    accepted = process_recording(event, headers, **kwargs)
    assert accepted[0].status == "succeeded"
    losing = pb.FinalTrackResult.FromString(accepted[1])
    losing.attempt_id = str(uuid.uuid4())
    losing.status = "failed"
    assert store.publish_once(losing, None) == accepted
    assert process_recording(event, headers, **kwargs) == accepted and track.calls == 1
    key = store.validate_source(source, event.tenant_id, event.session_id)
    client.objects["recordings", key] = bytes(32044)
    with pytest.raises(ContractError, match="audio_digest_mismatch"):
        with store.audio_file(source, event.tenant_id, event.session_id):
            pytest.fail("corrupt source was accepted")


@pytest.mark.parametrize("fail_on", [1, 2])
def test_kafka_publication_failure_replays_accepted_bytes(recording, tmp_path, monkeypatch, fail_on):
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
        if len(published) == fail_on:
            raise RuntimeError("uncertain_publication")
    monkeypatch.setattr(track_worker, "produce_acked", publish)
    kwargs = dict(producer=None, track_id="whisperx", profile_id=track.profile, provider=track, store=store)
    with pytest.raises(RuntimeError, match="uncertain_publication"):
        track_worker.handle_message(Message(), **kwargs)
    track_worker.handle_message(Message(), **kwargs)
    assert track.calls == 1 and published[:fail_on] == published[fail_on:fail_on * 2]


def test_storage_outage_leaves_no_terminal_outcome(recording, tmp_path, monkeypatch):
    event, _, headers = recording
    store, track = MemoryStore(tmp_path), FakeTrack()
    def unavailable(*args):
        raise ConnectionError("fixture outage")
    monkeypatch.setattr(store, "read_outcome", unavailable)
    with pytest.raises(ConnectionError):
        process_recording(event, headers, track_id="whisperx", profile_id=track.profile,
                          provider=track, store=store)
    assert track.calls == 0 and not store.outcomes


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
