import hashlib
import uuid

import pytest
from proto_gen import stream_pb2 as pb
from drsynth_common.final_tracks import ContractError, plan_headers, plan_dict
from drsynth_common.final_track_artifacts import S3Artifacts
from drsynth_common.refinement_tracks import validate_window, selected_audio
from recorder_worker.refinement_windows import WindowProducer
from finalizer_worker.track_processing import process_recording
from tests.test_final_tracks_unit import FakeS3, FakeTrack


class ReplayS3(FakeS3):
    def head_object(self, *, Bucket, Key):
        data = self.objects[Bucket, Key]
        return {"ContentLength": len(data), "Metadata": {"sha256": hashlib.sha256(data).hexdigest()},
                "VersionId": "fixture-version"}


@pytest.fixture
def rig():
    tenant, session, generation = (str(uuid.uuid4()) for _ in range(3))
    plan = dict(schema_version=1, tenant_id=tenant, session_id=session, plan_id=generation,
                final_tracks=[], refinement_window_samples=160000,
                refinement_tracks=[dict(track_id="whisperx", profile_id="whisperx-medium-refined-r1", primary=True),
                                   dict(track_id="shadow", profile_id="test-refined-r1", primary=False)])
    client, sent = ReplayS3(), []
    store = S3Artifacts(client, "recordings", recording_prefix="refinement-windows", result_prefix="refined-tracks")
    producer = WindowProducer(store, lambda *args: sent.append(args))
    return plan, store, sent, producer


class Message:
    def __init__(self, plan, seq, samples=0, *, eof=False, offset=None, partition=0):
        self.plan, self.part = plan, partition
        self.off = seq - 1 if offset is None else offset
        self.chunk = pb.AudioChunk(tenant_id=plan["tenant_id"], session_id=plan["session_id"],
                                   seq=seq, sample_rate=16000, pcm16_le=bytes(samples * 2), lang="cs")
        self.hdrs = plan_headers(plan) + [("x-outputs", b"refined"), ("x-store-recording", b"true")]
        if eof:
            self.hdrs.append(("x-audio-eof", b"true"))

    def value(self): return self.chunk.SerializeToString()
    def key(self): return self.chunk.session_id.encode()
    def headers(self): return self.hdrs
    def offset(self): return self.off
    def partition(self): return self.part


def test_live_windows_eof_replay_and_absolute_primary(rig):
    plan, store, sent, producer = rig
    messages = [Message(plan, 1, 160000), Message(plan, 2, 80000), Message(plan, 3, eof=True)]
    producer.accept(messages[0])
    assert not sent and producer.frontier(0, 1) == 0
    producer.accept(messages[1])
    assert len(sent) == 1 and producer.frontier(0, 2) == 0
    producer.accept(messages[2])
    assert len(sent) == 2 and producer.frontier(0, 3) == 3
    original = list(sent)
    replay = WindowProducer(store, lambda *args: sent.append(args))
    for message in messages:
        replay.accept(message)
    assert sent[2:] == original
    windows = [pb.RefinementWindow.FromString(value) for _, value, _ in original]
    assert [(w.start_sample, w.end_sample, w.flush_reason) for w in windows] == [
        (0, 160000, "slice"), (160000, 240000, "eof")]
    run_ids = []
    for window, (_, _, headers) in zip(windows, original):
        assert validate_window(window, headers) == plan_dict(window.recording.final_plan)
        primary = FakeTrack(profile="whisperx-medium-refined-r1")
        args = dict(track_id="whisperx", profile_id=primary.profile, provider=primary, store=store, window=window)
        outcome, canonical, legacy = process_recording(window.recording, headers, **args)
        assert outcome.stage == "refined" and outcome.status == "succeeded"
        run_ids.append(outcome.run_id)
        projected = pb.RefinedEvent.FromString(legacy)
        assert projected.start_s == window.start_sample / 16000
        assert projected.segments[0].start_s >= projected.start_s
        assert process_recording(window.recording, headers, **args)[1:] == (canonical, legacy)
        assert primary.calls == 1
        failed = FakeTrack(failures=9, profile="test-refined-r1")
        result = process_recording(window.recording, headers, window=window, store=store,
                                    track_id="shadow", profile_id=failed.profile, provider=failed)
        assert result[0].status == "failed" and result[2] is None
        assert result[0].source == outcome.source and result[0].run_id != outcome.run_id
    assert run_ids[0] == run_ids[1]


def test_restart_before_tail_reconstructs_without_checkpoint(rig):
    plan, store, sent, producer = rig
    first, second = Message(plan, 1, 200000), Message(plan, 2, eof=True)
    producer.accept(first)
    original = sent[0]
    replacement = WindowProducer(store, lambda *args: sent.append(args))
    replacement.accept(first)
    replacement.accept(second)
    assert sent[1] == original and pb.RefinementWindow.FromString(sent[-1][1]).end_sample == 200000


def test_idle_boundary_and_exact_window_are_replay_stable(rig):
    plan, store, sent, producer = rig
    message = Message(plan, 1, 160000)
    producer.accept(message)
    producer.idle(0)
    assert pb.RefinementWindow.FromString(sent[0][1]).flush_reason == "idle"
    WindowProducer(store, lambda *args: sent.append(args)).accept(message)
    assert sent[0] == sent[1]
    with pytest.raises(ContractError):
        producer.accept(Message(plan, 2, 16000))


def test_partition_frontier_never_skips_another_open_session(rig):
    plan, store, sent, producer = rig
    second = dict(plan, session_id=str(uuid.uuid4()), plan_id=str(uuid.uuid4()))
    producer.accept(Message(plan, 1, 16000, offset=3))
    producer.accept(Message(second, 1, 16000, offset=4))
    producer.accept(Message(second, 2, eof=True, offset=5))
    assert producer.frontier(0, 6) == 3
    producer.accept(Message(plan, 2, eof=True, offset=6))
    assert producer.frontier(0, 7) == 7


@pytest.mark.parametrize("bad", ["sequence", "tenant", "plan", "retention", "odd", "key", "policy"])
def test_tampering_rejected_without_publishing(rig, bad):
    plan, store, sent, producer = rig
    producer.accept(Message(plan, 1, 16000))
    message = Message(plan, 2, 16000)
    if bad == "sequence": message.chunk.seq = 3
    if bad == "tenant": message.chunk.tenant_id = str(uuid.uuid4())
    if bad == "plan":
        changed = dict(plan, plan_id=str(uuid.uuid4()))
        message.hdrs = plan_headers(changed)
    if bad == "retention": message.hdrs[-1] = ("x-store-recording", b"false")
    if bad == "odd": message.chunk.pcm16_le = b"x"
    if bad == "key": message.key = lambda: b"other"
    if bad == "policy":
        changed = dict(plan, refinement_window_samples=320000)
        message.hdrs = plan_headers(changed)
    with pytest.raises(ContractError): producer.accept(message)
    assert not sent and producer.frontier(0, 2) == 0


def test_legacy_skip_and_failure_after_durable_window(rig):
    plan, store, sent, producer = rig
    message = Message(plan, 1, 200000)
    assert selected_audio(message.value(), message.headers())
    def fail(*args): raise ConnectionError("publication failed")
    producer.publish = fail
    with pytest.raises(ConnectionError): producer.accept(message)
    assert producer.frontier(0, 1) == 0
    replay = WindowProducer(store, lambda *args: sent.append(args))
    replay.accept(message)
    assert len(sent) == 1
