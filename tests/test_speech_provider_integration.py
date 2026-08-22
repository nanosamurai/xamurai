from concurrent import futures
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

from proto_gen import speech_provider_pb2, speech_provider_pb2_grpc
from proto_gen import stream_pb2, stream_pb2_grpc
from rtservice.engine import RealtimeEngine
from rtservice.providers import (
    FASTER_WHISPER_MEDIUM_PROFILE,
    QWEN3_ASR_06B_VLLM_PROFILE,
    GrpcSpeechProvider,
    LocalFasterWhisperProvider,
    ProviderRegistry,
    WindowRequest,
)
from rtservice.server import create_realtime_asr_server


class _FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, _wave, **kwargs):
        self.calls.append(kwargs)
        if kwargs["word_timestamps"]:
            segment = SimpleNamespace(words=[SimpleNamespace(word="hello"), SimpleNamespace(word=" world")])
        else:
            segment = SimpleNamespace(words=None, text="partial text")
        return iter([segment]), SimpleNamespace()


def test_local_profile_preserves_final_and_partial_decode_settings(monkeypatch):
    monkeypatch.setenv("RT_ASR_TEMPERATURES", "0,0.3,0.6")
    monkeypatch.setenv("RT_ASR_COMPRESSION_RATIO_THRESHOLD", "2.7")
    model = _FakeWhisperModel()
    provider = LocalFasterWhisperProvider(model=model)
    pcm = (np.ones(16000, dtype=np.int16)).tobytes()

    final = provider.transcribe_window(WindowRequest(pcm, 16000, "en", 0, 16000, False))
    partial = provider.transcribe_window(WindowRequest(pcm, 16000, "en", 0, 16000, True))

    assert provider.profile_id == FASTER_WHISPER_MEDIUM_PROFILE
    assert final.text == "hello  world"
    assert final.terminal is True
    assert partial.text == "partial text"
    assert partial.terminal is False
    assert model.calls[0]["beam_size"] == 5
    assert model.calls[0]["word_timestamps"] is True
    assert model.calls[1]["beam_size"] == 1
    assert model.calls[1]["word_timestamps"] is False
    for call in model.calls:
        assert call["temperature"] == (0.0, 0.3, 0.6)
        assert call["compression_ratio_threshold"] == 2.7
        assert call["condition_on_previous_text"] is False
        assert call["vad_filter"] is False


@pytest.mark.parametrize("temperatures", ["", "0,,0.2", "nan", "-0.1", "1.1", "0.5,0.2"])
def test_local_profile_rejects_invalid_temperature_schedules(monkeypatch, temperatures):
    monkeypatch.setenv("RT_ASR_TEMPERATURES", temperatures)
    with pytest.raises(ValueError, match="RT_ASR_TEMPERATURES"):
        LocalFasterWhisperProvider(model=_FakeWhisperModel())


class _FakeNativeProvider(speech_provider_pb2_grpc.SpeechProviderServicer):
    def GetCapabilities(self, request, context):
        assert request.profile_id == QWEN3_ASR_06B_VLLM_PROFILE
        return speech_provider_pb2.CapabilitiesResponse(
            capabilities=speech_provider_pb2.ProviderCapabilities(
                native_streaming=True,
                language_detection=True,
                stateful=True,
                preferred_sample_rate=16000,
                maximum_audio_seconds=30,
                maximum_concurrent_sessions=1,
            ),
            provenance=speech_provider_pb2.ProviderProvenance(
                profile_id=QWEN3_ASR_06B_VLLM_PROFILE,
                runtime="fake-vllm",
                model_revision="fixed-revision",
                model_digest="sha256:fake",
                implementation_revision="test",
            ),
        )

    def TranscribeWindow(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "native profile")

    def StreamTranscribe(self, request_iterator, context):
        total_samples = 0
        for frame in request_iterator:
            assert frame.profile_id == QWEN3_ASR_06B_VLLM_PROFILE
            total_samples += len(frame.pcm16_le) // 2
            yield speech_provider_pb2.ProviderTranscriptEvent(
                request_id=frame.request_id,
                provider_session_id=frame.provider_session_id,
                provider_sequence=frame.provider_sequence,
                text="validated" if frame.end_of_stream else "valid",
                language=frame.language,
                start_sample=0,
                end_sample=total_samples,
                terminal=frame.end_of_stream,
            )


def test_remote_native_profile_handshake_and_stream_round_trip():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    speech_provider_pb2_grpc.add_SpeechProviderServicer_to_server(_FakeNativeProvider(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        provider = GrpcSpeechProvider(
            endpoint=f"127.0.0.1:{port}",
            profile_id=QWEN3_ASR_06B_VLLM_PROFILE,
            request_timeout_seconds=2,
        )
        stream = provider.open_stream()
        pcm = np.ones(1600, dtype=np.int16).tobytes()

        partial = stream.push(pcm, sample_rate=16000, language="en")
        final = stream.push(b"", sample_rate=16000, language="en", end_of_stream=True)

        assert provider.capabilities.native_streaming is True
        assert provider.provenance.model_revision == "fixed-revision"
        assert partial.text == "valid"
        assert (partial.start_sample, partial.end_sample, partial.terminal) == (0, 1600, False)
        assert final.text == "validated"
        assert (final.start_sample, final.end_sample, final.terminal) == (0, 1600, True)
        provider.close()
    finally:
        server.stop(grace=None).wait()


def test_public_realtime_stream_flushes_native_provider_on_request_eof(monkeypatch):
    provider_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    speech_provider_pb2_grpc.add_SpeechProviderServicer_to_server(_FakeNativeProvider(), provider_server)
    provider_port = provider_server.add_insecure_port("127.0.0.1:0")
    provider_server.start()
    provider = GrpcSpeechProvider(
        endpoint=f"127.0.0.1:{provider_port}",
        profile_id=QWEN3_ASR_06B_VLLM_PROFILE,
        request_timeout_seconds=2,
    )
    engine = RealtimeEngine(
        provider_registry=ProviderRegistry((provider,), default_profile_id=provider.profile_id)
    )
    monkeypatch.setenv("RT_GRPC_BIND_ADDR", "127.0.0.1")
    rt_server = create_realtime_asr_server(port=0, engine=engine)
    rt_server.start()
    try:
        channel = grpc.insecure_channel(f"127.0.0.1:{rt_server.bound_port}")
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        pcm = np.ones(1600, dtype=np.int16).tobytes()
        events = list(
            stub.Stream(
                iter(
                    [
                        stream_pb2.AudioChunk(
                            session_id="native-session",
                            seq=1,
                            sample_rate=16000,
                            pcm16_le=pcm,
                            lang="en",
                            tenant_id="tenant-a",
                        )
                    ]
                ),
                timeout=5,
            )
        )

        assert [(event.text, event.type) for event in events] == [
            ("valid", stream_pb2.PARTIAL),
            ("validated", stream_pb2.FINAL),
        ]
        assert [(event.start_s, event.end_s) for event in events] == [(0.0, 0.1), (0.0, 0.1)]
        channel.close()
    finally:
        rt_server.stop(grace=None).wait()
        engine.close()
        provider_server.stop(grace=None).wait()
