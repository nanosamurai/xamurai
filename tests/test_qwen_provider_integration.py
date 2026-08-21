from types import SimpleNamespace

import numpy as np

from qwen_provider.server import MODEL_DIGEST, MODEL_REVISION, create_server
from rtservice.providers import GrpcSpeechProvider, QWEN3_ASR_06B_VLLM_PROFILE


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
