import queue
import time

import grpc
import pytest

from nemotron_rtservice.native import MODEL_DIGEST, MODEL_REVISION, TranscriptUpdate
from nemotron_rtservice.server import PROFILE_ID, create_server
from proto_gen import stream_pb2, stream_pb2_grpc


class _FakeStream:
    def __init__(self):
        self.chunks = []
        self.closed = 0

    def push(self, pcm16_le, sample_rate):
        self.chunks.append((pcm16_le, sample_rate))
        return (
            TranscriptUpdate(
                text=f"partial-{len(self.chunks)}",
                final=False,
                audio_processed_s=sum(len(data) for data, _ in self.chunks) / 2 / 16_000,
                language="cs-CZ",
            ),
        )

    def finish(self):
        return (
            TranscriptUpdate(
                text="native final",
                final=True,
                audio_processed_s=sum(len(data) for data, _ in self.chunks) / 2 / 16_000,
                language="cs-CZ",
            ),
        )

    def close(self):
        self.closed += 1


class _FakeBackend:
    runtime = "nemo-speech-cpp==test"

    def __init__(self):
        self.opens = []

    def open(self, session_id, language):
        stream = _FakeStream()
        self.opens.append((session_id, language, stream))
        return stream


def _chunk(session_id="nemotron-session", sequence=1, payload=b"\x01\x00" * 1600,
           lang="cs"):
    return stream_pb2.AudioChunk(
        session_id=session_id,
        seq=sequence,
        sample_rate=16_000,
        pcm16_le=payload,
        lang=lang,
    )


def _call(stub, requests, session_id="nemotron-session", timeout=3):
    return stub.Stream(
        requests,
        timeout=timeout,
        metadata=(("x-session-id", session_id),),
    )


@pytest.fixture
def running_server(monkeypatch):
    monkeypatch.setenv("NEMOTRON_RTSERVICE_BIND_ADDR", "127.0.0.1")
    backend = _FakeBackend()
    server = create_server(backend, port=0, maximum_sessions=2)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    try:
        yield backend, stream_pb2_grpc.RealtimeASRStub(channel)
    finally:
        channel.close()
        server.stop(grace=None).wait()


def test_nemotron_stream_pushes_only_new_chunks_and_flushes_final(running_server):
    backend, stub = running_server
    first = b"\x01\x00" * 800
    second = b"\x02\x00" * 1200

    capabilities = stub.GetCapabilities(stream_pb2.RealtimeCapabilitiesRequest(), timeout=2)
    events = list(
        _call(
            stub,
            iter(
                [
                    _chunk(sequence=7, payload=first),
                    _chunk(sequence=8, payload=second),
                ]
            ),
        )
    )

    assert capabilities.provider_profile_id == PROFILE_ID
    assert capabilities.native_streaming is True
    assert capabilities.windowed_realtime is False
    assert capabilities.batch is True
    assert capabilities.maximum_concurrent_sessions == 2
    assert capabilities.model_revision == MODEL_REVISION
    assert capabilities.model_digest == MODEL_DIGEST
    assert "cs" in capabilities.supported_languages
    assert events[0].type == stream_pb2.SESSION_ACCEPTED
    assert events[0].serving_instance_id
    assert [(event.text, event.type, event.lang) for event in events[1:]] == [
        ("partial-1", stream_pb2.PARTIAL, "cs"),
        ("partial-2", stream_pb2.PARTIAL, "cs"),
        ("native final", stream_pb2.FINAL, "cs"),
    ]
    assert backend.opens[0][:2] == ("nemotron-session", "cs-CZ")
    native_stream = backend.opens[0][2]
    assert native_stream.chunks == [(first, 16_000), (second, 16_000)]
    assert native_stream.closed == 1


class _HeldRequests:
    _END = object()

    def __init__(self):
        self._items = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self._items.get(timeout=3)
        if item is self._END:
            raise StopIteration
        return item

    def finish(self):
        self._items.put(self._END)


def test_nemotron_admits_two_sessions_and_recovers_after_cancel(running_server):
    _, stub = running_server
    held_one = _HeldRequests()
    held_two = _HeldRequests()
    first = _call(stub, held_one, "held-one", timeout=5)
    second = _call(stub, held_two, "held-two", timeout=5)
    try:
        assert next(first).type == stream_pb2.SESSION_ACCEPTED
        assert next(second).type == stream_pb2.SESSION_ACCEPTED

        with pytest.raises(grpc.RpcError) as rejected:
            list(_call(stub, iter(()), "rejected", timeout=2))
        assert rejected.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert rejected.value.details() == "REPLICA_FULL"

        first.cancel()
        for _ in range(20):
            try:
                recovered = _call(stub, iter(()), "recovered", timeout=1)
                assert next(recovered).type == stream_pb2.SESSION_ACCEPTED
                recovered.cancel()
                break
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.RESOURCE_EXHAUSTED:
                    raise
                time.sleep(0.05)
        else:
            pytest.fail("cancelled session slot was not released")
    finally:
        first.cancel()
        second.cancel()
        held_one.finish()
        held_two.finish()


@pytest.mark.parametrize(
    ("chunks", "detail"),
    [
        ([_chunk(sequence=1), _chunk(sequence=3)], "audio chunk sequence is not contiguous"),
        ([_chunk(payload=b"\x00")], "PCM16 payload length must be even"),
        ([_chunk(lang="zz")], "Nemotron realtime service does not support"),
    ],
)
def test_nemotron_rejects_invalid_audio(running_server, chunks, detail):
    _, stub = running_server
    with pytest.raises(grpc.RpcError) as failure:
        list(_call(stub, iter(chunks)))
    assert failure.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert detail in failure.value.details()
