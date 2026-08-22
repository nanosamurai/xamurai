import logging
import time
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_provider.server import MODEL_DIGEST, MODEL_REVISION, create_server
from rtservice.providers import GrpcSpeechProvider, QWEN3_ASR_06B_VLLM_PROFILE, SpeechProviderError


class _FakeQwenBackend:
    runtime = "qwen-asr==test;vllm==test"
    supported_languages = ("English", "Czech")

    def open(self, language):
        return SimpleNamespace(text="", language=language or "", samples=0)

    def push(self, pcm16, state):
        state.samples += pcm16.size
        state.text = "native partial"
        return state

    def finish(self, state):
        state.text = "native final"
        return state


def test_qwen_servicer_exposes_native_vllm_profile(monkeypatch):
    monkeypatch.setenv("QWEN_PROVIDER_BIND_ADDR", "127.0.0.1")
    server = create_server(_FakeQwenBackend(), port=0)
    server.start()
    try:
        provider = GrpcSpeechProvider(
            endpoint=f"127.0.0.1:{server.bound_port}",
            profile_id=QWEN3_ASR_06B_VLLM_PROFILE,
            request_timeout_seconds=2,
        )
        session = provider.open_stream()
        pcm = np.ones(1600, dtype=np.int16).tobytes()
        partial = session.push(pcm, sample_rate=16000, language="English")
        final = session.push(b"", sample_rate=16000, language="English", end_of_stream=True)

        assert provider.capabilities.native_streaming is True
        assert provider.capabilities.word_timestamps is False
        assert provider.capabilities.segment_timestamps is False
        assert provider.provenance.model_revision == MODEL_REVISION
        assert provider.provenance.model_digest == MODEL_DIGEST
        assert (partial.text, partial.start_sample, partial.end_sample) == ("native partial", 0, 1600)
        assert (final.text, final.terminal, final.end_sample) == ("native final", True, 1600)
        provider.close()
    finally:
        server.stop(grace=None).wait()


def test_qwen_client_cancellation_releases_session_without_error(monkeypatch, caplog):
    monkeypatch.setenv("QWEN_PROVIDER_BIND_ADDR", "127.0.0.1")
    server = create_server(_FakeQwenBackend(), port=0)
    server.start()
    provider = GrpcSpeechProvider(
        endpoint=f"127.0.0.1:{server.bound_port}",
        profile_id=QWEN3_ASR_06B_VLLM_PROFILE,
        request_timeout_seconds=2,
    )
    pcm = np.ones(1600, dtype=np.int16).tobytes()
    caplog.set_level(logging.ERROR)
    try:
        cancelled = provider.open_stream()
        assert cancelled.push(pcm, sample_rate=16000, language="English").text == "native partial"
        cancelled.cancel()
        time.sleep(0.1)

        recovered = provider.open_stream()
        assert recovered.push(pcm, sample_rate=16000, language="English").text == "native partial"
        assert recovered.push(b"", sample_rate=16000, language="English", end_of_stream=True).terminal
        assert "Qwen provider stream failed" not in caplog.text
    finally:
        provider.close()
        server.stop(grace=None).wait()


def test_qwen_inference_failure_does_not_kill_the_provider(monkeypatch):
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
    monkeypatch.setenv("QWEN_PROVIDER_BIND_ADDR", "127.0.0.1")
    server = create_server(backend, port=0)
    server.start()
    provider = GrpcSpeechProvider(
        endpoint=f"127.0.0.1:{server.bound_port}",
        profile_id=QWEN3_ASR_06B_VLLM_PROFILE,
        request_timeout_seconds=2,
    )
    pcm = np.ones(1600, dtype=np.int16).tobytes()
    try:
        with pytest.raises(SpeechProviderError, match="Qwen provider inference failed"):
            provider.open_stream().push(pcm, sample_rate=16000, language="English")

        recovered = provider.open_stream()
        assert recovered.push(pcm, sample_rate=16000, language="English").text == "native partial"
        assert recovered.push(b"", sample_rate=16000, language="English", end_of_stream=True).terminal
    finally:
        provider.close()
        server.stop(grace=None).wait()
