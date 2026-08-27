import logging
import threading
import time
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

from proto_gen import stream_pb2, stream_pb2_grpc
from qwen_rtservice.server import (
    MODEL_DIGEST,
    MODEL_REVISION,
    PROFILE_ID,
    _load_runtime_config,
    create_server,
)


class _FakeQwenBackend:
    runtime = "qwen-asr==test;vllm==test"
    supported_languages = ("English", "Czech")

    def __init__(self):
        self.opened_languages = []
        self.opened_contexts = []
        self.maximum_state_samples = 0

    def open(self, language, context):
        self.opened_languages.append(language)
        self.opened_contexts.append(context)
        return SimpleNamespace(text="", language=language or "", samples=0)

    def push(self, pcm16, state):
        state.samples += pcm16.size
        self.maximum_state_samples = max(self.maximum_state_samples, state.samples)
        state.text = "native partial"
        return state

    def finish(self, state):
        state.text = "native final"
        return state


def _chunk(*, session_id="qwen-session", sequence=1):
    return stream_pb2.AudioChunk(
        session_id=session_id,
        seq=sequence,
        sample_rate=16000,
        pcm16_le=np.ones(1600, dtype=np.int16).tobytes(),
        lang="cs",
    )


def test_qwen_rtservice_exposes_capabilities_and_native_stream(monkeypatch):
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    backend = _FakeQwenBackend()
    server = create_server(backend, port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    try:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        capabilities = stub.GetCapabilities(stream_pb2.RealtimeCapabilitiesRequest(), timeout=2)
        events = list(stub.Stream(iter([_chunk()]), timeout=2))

        assert capabilities.provider_profile_id == PROFILE_ID
        assert capabilities.native_streaming is True
        assert capabilities.windowed_realtime is False
        assert capabilities.word_timestamps is False
        assert capabilities.segment_timestamps is False
        assert capabilities.maximum_audio_seconds == 0
        assert capabilities.maximum_concurrent_sessions == 1
        assert capabilities.model_revision == MODEL_REVISION
        assert capabilities.model_digest == MODEL_DIGEST
        assert list(capabilities.supported_languages) == ["en", "cs"]
        assert backend.opened_languages == ["Czech"]
        assert backend.opened_contexts == [""]
        assert [(event.text, event.type) for event in events] == [
            ("native partial", stream_pb2.PARTIAL),
            ("native final", stream_pb2.FINAL),
        ]
        assert all(event.provider_profile_id == PROFILE_ID for event in events)
        assert all(event.lang == "cs" for event in events)
        assert [(event.start_s, event.end_s) for event in events] == [(0.0, 0.1), (0.0, 0.1)]
    finally:
        channel.close()
        server.stop(grace=None).wait()


def test_qwen_client_cancellation_releases_session_without_error(monkeypatch, caplog):
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    server = create_server(_FakeQwenBackend(), port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    release_request = threading.Event()

    def held_request():
        yield _chunk(session_id="cancelled")
        release_request.wait(2)

    caplog.set_level(logging.ERROR)
    try:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        cancelled = stub.Stream(held_request(), timeout=5)
        assert next(cancelled).text == "native partial"
        cancelled.cancel()
        release_request.set()
        time.sleep(0.1)

        recovered = list(stub.Stream(iter([_chunk(session_id="recovered")]), timeout=2))
        assert [event.text for event in recovered] == ["native partial", "native final"]
        assert "Qwen realtime stream failed" not in caplog.text
    finally:
        release_request.set()
        channel.close()
        server.stop(grace=None).wait()


def test_qwen_inference_failure_does_not_kill_rtservice(monkeypatch):
    backend = _FakeQwenBackend()
    original_push = backend.push
    failures_remaining = 1

    def fail_once(pcm16, state):
        nonlocal failures_remaining
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("sensitive backend detail")
        return original_push(pcm16, state)

    backend.push = fail_once
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    server = create_server(backend, port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    try:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        with pytest.raises(grpc.RpcError) as failure:
            list(stub.Stream(iter([_chunk(session_id="failure")]), timeout=2))
        assert failure.value.code() == grpc.StatusCode.INTERNAL
        assert failure.value.details() == "Qwen realtime inference failed"

        recovered = list(stub.Stream(iter([_chunk(session_id="recovered")]), timeout=2))
        assert [event.text for event in recovered] == ["native partial", "native final"]
    finally:
        channel.close()
        server.stop(grace=None).wait()


def test_qwen_rtservice_rejects_noncontiguous_audio(monkeypatch):
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    server = create_server(_FakeQwenBackend(), port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    try:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        with pytest.raises(grpc.RpcError) as failure:
            list(stub.Stream(iter([_chunk(sequence=1), _chunk(sequence=3)]), timeout=2))
        assert failure.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert failure.value.details() == "audio chunk sequence is not contiguous"
    finally:
        channel.close()
        server.stop(grace=None).wait()


def test_qwen_rtservice_rejects_unsupported_language_code(monkeypatch):
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    server = create_server(_FakeQwenBackend(), port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    try:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        chunk = _chunk()
        chunk.lang = "zz"
        with pytest.raises(grpc.RpcError) as failure:
            list(stub.Stream(iter([chunk]), timeout=2))
        assert failure.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert failure.value.details() == "Qwen realtime service does not support the requested language code"
    finally:
        channel.close()
        server.stop(grace=None).wait()


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {"QWEN_STREAM_EPOCH_SECONDS": "600", "QWEN_MAX_MODEL_LEN": "2048"},
            "QWEN_MAX_MODEL_LEN is too small",
        ),
        (
            {
                "QWEN_STREAM_EPOCH_SECONDS": "10",
                "QWEN_MAX_MODEL_LEN": "8192",
                "QWEN_KV_CACHE_MIB": "512",
            },
            "QWEN_KV_CACHE_MIB is too small",
        ),
    ],
)
def test_qwen_runtime_rejects_incompatible_context_and_cache_limits(monkeypatch, environment, message):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        _load_runtime_config()


class _EpochQwenBackend(_FakeQwenBackend):
    def open(self, language, context):
        state = super().open(language, context)
        state.epoch = len(self.opened_contexts)
        state.context = context
        return state

    def push(self, pcm16, state):
        state.samples += pcm16.size
        self.maximum_state_samples = max(self.maximum_state_samples, state.samples)
        state.text = f"{state.context} word-{state.epoch}".strip()
        return state

    def finish(self, state):
        return state


def test_qwen_rtservice_rolls_epochs_without_public_cutoff_or_duplicate_text(monkeypatch):
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    monkeypatch.setenv("QWEN_STREAM_EPOCH_SECONDS", "10")
    backend = _EpochQwenBackend()
    server = create_server(backend, port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")

    def chunks():
        samples = np.ones(801_600, dtype=np.int16).tobytes()
        for sequence in range(1, 8):
            yield stream_pb2.AudioChunk(
                session_id="long-qwen-session",
                seq=sequence,
                sample_rate=16000,
                pcm16_le=samples,
                lang="en",
            )

    try:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        events = list(stub.Stream(chunks(), timeout=10))

        final_events = [event for event in events if event.type == stream_pb2.FINAL]
        assert events[-1].type == stream_pb2.FINAL
        assert events[-1].end_s > 300
        assert " ".join(event.text for event in final_events) == " ".join(
            f"word-{epoch}" for epoch in range(1, len(final_events) + 1)
        )
        assert all(event.start_s < event.end_s for event in final_events)
        assert all(
            left.end_s == right.start_s
            for left, right in zip(final_events, final_events[1:])
        )
        assert backend.maximum_state_samples <= 10 * 16000
        assert backend.opened_contexts[:3] == ["", "word-1", "word-1 word-2"]
    finally:
        channel.close()
        server.stop(grace=None).wait()
