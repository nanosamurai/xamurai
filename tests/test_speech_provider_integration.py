from concurrent import futures
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

from proto_gen import stream_pb2, stream_pb2_grpc
from rtservice.providers import (
    FASTER_WHISPER_MEDIUM_PROFILE,
    LocalFasterWhisperProvider,
    WindowRequest,
)
from rtservice.server import RealtimeASRServicer
from rtservice.server import create_realtime_asr_server


class _FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, _wave, **kwargs):
        self.calls.append(kwargs)
        if kwargs["word_timestamps"]:
            segment = SimpleNamespace(
                words=[
                    SimpleNamespace(word="hello", start=0.1, end=0.4),
                    SimpleNamespace(word=" world", start=0.4, end=0.8),
                ]
            )
        else:
            segment = SimpleNamespace(words=None, text="partial text")
        return iter([segment]), SimpleNamespace()


def test_local_profile_preserves_final_and_partial_decode_settings(monkeypatch):
    monkeypatch.setenv("RT_ASR_TEMPERATURES", "0,0.3,0.6")
    monkeypatch.setenv("RT_ASR_COMPRESSION_RATIO_THRESHOLD", "2.7")
    model = _FakeWhisperModel()
    provider = LocalFasterWhisperProvider(model=model)
    pcm = (np.ones(16000, dtype=np.int16)).tobytes()

    final = provider.transcribe_window(WindowRequest(pcm, 16000, "en", 32000, 48000, False))
    partial = provider.transcribe_window(WindowRequest(pcm, 16000, "en", 0, 16000, True))

    assert provider.profile_id == FASTER_WHISPER_MEDIUM_PROFILE
    assert final.text == "hello  world"
    assert [(word.text, word.start_sample, word.end_sample) for word in final.words] == [
        ("hello", 33600, 38400),
        (" world", 38400, 44800),
    ]
    assert final.terminal is True
    assert partial.text == "partial text"
    assert partial.terminal is False
    assert model.calls[0]["beam_size"] == 5
    assert model.calls[0]["word_timestamps"] is True
    assert model.calls[1]["beam_size"] == 1
    assert model.calls[1]["word_timestamps"] is False
    assert model.calls[0]["without_timestamps"] is False
    assert model.calls[0]["temperature"] == (0.0, 0.3, 0.6)
    assert model.calls[1]["without_timestamps"] is True
    assert model.calls[1]["temperature"] == 0.0
    for call in model.calls:
        assert call["compression_ratio_threshold"] == 2.7
        assert call["condition_on_previous_text"] is False
        assert call["vad_filter"] is False


def test_faster_rtservice_broadcasts_its_fixed_capabilities():
    provider = LocalFasterWhisperProvider(model=_FakeWhisperModel())
    engine = SimpleNamespace(default_provider=lambda: provider)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    stream_pb2_grpc.add_RealtimeASRServicer_to_server(RealtimeASRServicer(engine), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        capabilities = stream_pb2_grpc.RealtimeASRStub(channel).GetCapabilities(
            stream_pb2.RealtimeCapabilitiesRequest(), timeout=2
        )

        assert capabilities.provider_profile_id == FASTER_WHISPER_MEDIUM_PROFILE
        assert capabilities.windowed_realtime is True
        assert capabilities.native_streaming is False
        assert capabilities.word_timestamps is True
        assert capabilities.speaker_labels is True
        assert list(capabilities.aligned_diarized_languages) == []
        assert capabilities.model_revision == provider.model_revision
        assert capabilities.model_digest == provider.model_digest
    finally:
        channel.close()
        server.stop(grace=None).wait()


@pytest.mark.parametrize("temperatures", ["", "0,,0.2", "nan", "-0.1", "1.1", "0.5,0.2"])
def test_local_profile_rejects_invalid_temperature_schedules(monkeypatch, temperatures):
    monkeypatch.setenv("RT_ASR_TEMPERATURES", temperatures)
    with pytest.raises(ValueError, match="RT_ASR_TEMPERATURES"):
        LocalFasterWhisperProvider(model=_FakeWhisperModel())
