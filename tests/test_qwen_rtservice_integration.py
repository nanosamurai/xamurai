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
from qwen_rtservice.enrichment import EnrichedSegment, QwenEpochEnricher
from drsynth_common.diarization_assign import DiarizationSegment


class _FakeQwenBackend:
    runtime = "qwen-asr==test;vllm==test"
    supported_languages = ("English", "Czech")
    alignment_languages = ("English",)

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


def _chunk(*, session_id="qwen-session", sequence=1, lang="cs", samples=1600):
    return stream_pb2.AudioChunk(
        session_id=session_id,
        seq=sequence,
        sample_rate=16000,
        pcm16_le=np.ones(samples, dtype=np.int16).tobytes(),
        lang=lang,
    )


def _stream(stub, requests, *, session_id="qwen-session", timeout=2):
    return stub.Stream(
        requests,
        timeout=timeout,
        metadata=(("x-session-id", session_id),),
    )


def _transcript_events(stub, requests, *, session_id="qwen-session", timeout=2):
    events = list(_stream(stub, requests, session_id=session_id, timeout=timeout))
    assert events[0].type == stream_pb2.SESSION_ACCEPTED
    assert events[0].session_id == session_id
    assert events[0].serving_instance_id
    return events[1:]


def test_qwen_rtservice_exposes_capabilities_and_native_stream(monkeypatch):
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    backend = _FakeQwenBackend()
    server = create_server(backend, port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    try:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        capabilities = stub.GetCapabilities(stream_pb2.RealtimeCapabilitiesRequest(), timeout=2)
        events = _transcript_events(stub, iter([_chunk()]))

        assert capabilities.provider_profile_id == PROFILE_ID
        assert capabilities.native_streaming is True
        assert capabilities.windowed_realtime is False
        assert capabilities.word_timestamps is False
        assert capabilities.segment_timestamps is False
        assert capabilities.speaker_labels is False
        assert list(capabilities.aligned_diarized_languages) == []
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
        cancelled = _stream(stub, held_request(), session_id="cancelled", timeout=5)
        assert next(cancelled).type == stream_pb2.SESSION_ACCEPTED
        assert next(cancelled).text == "native partial"

        with pytest.raises(grpc.RpcError) as full:
            list(_stream(stub, iter(()), session_id="rejected", timeout=2))
        assert full.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert full.value.details() == "REPLICA_FULL"

        cancelled.cancel()
        release_request.set()
        time.sleep(0.1)

        recovered = _transcript_events(
            stub,
            iter([_chunk(session_id="recovered")]),
            session_id="recovered",
        )
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
            list(
                _stream(
                    stub,
                    iter([_chunk(session_id="failure")]),
                    session_id="failure",
                )
            )
        assert failure.value.code() == grpc.StatusCode.INTERNAL
        assert failure.value.details() == "Qwen realtime inference failed"

        recovered = _transcript_events(
            stub,
            iter([_chunk(session_id="recovered")]),
            session_id="recovered",
        )
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
            list(_stream(stub, iter([_chunk(sequence=1), _chunk(sequence=3)])))
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
            list(_stream(stub, iter([chunk])))
        assert failure.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert failure.value.details() == (
            "Qwen realtime service does not support the requested language code"
        )
    finally:
        channel.close()
        server.stop(grace=None).wait()


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {"QWEN_STREAM_EPOCH_SECONDS": "600", "QWEN_MAX_MODEL_LEN": "2048"},
            "QWEN_STREAM_EPOCH_SECONDS must be between 10.0 and 300.0",
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
def test_qwen_runtime_rejects_incompatible_context_and_cache_limits(
    monkeypatch, environment, message
):
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


class _FakeEnricher:
    supported_languages = ("English",)

    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def enrich(self, pcm16, text, language, *, epoch_number):
        self.calls.append((pcm16.copy(), text, language, epoch_number))
        if self.fail:
            raise RuntimeError("sensitive enrichment detail")
        duration_s = pcm16.size / 16000
        return (
            EnrichedSegment(
                0.01,
                duration_s / 2,
                "native ",
                f"EPOCH_{epoch_number:04d}/SPEAKER_00",
            ),
            EnrichedSegment(
                duration_s / 2,
                duration_s,
                "final",
                f"EPOCH_{epoch_number:04d}/SPEAKER_01",
            ),
        )


class _EpochEnricher:
    supported_languages = ("English",)

    def __init__(self):
        self.calls = []

    def enrich(self, pcm16, text, language, *, epoch_number):
        self.calls.append((pcm16.size, text, language, epoch_number))
        return (
            EnrichedSegment(
                0.0,
                pcm16.size / 16000,
                text,
                f"EPOCH_{epoch_number:04d}/SPEAKER_00",
            ),
        )


def test_qwen_supported_language_final_is_aligned_and_diarized(monkeypatch):
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    enricher = _FakeEnricher()
    server = create_server(_FakeQwenBackend(), enricher=enricher, port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    try:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        capabilities = stub.GetCapabilities(stream_pb2.RealtimeCapabilitiesRequest(), timeout=2)
        events = _transcript_events(stub, iter([_chunk(lang="en")]))
        finals = [event for event in events if event.type == stream_pb2.FINAL]

        assert capabilities.segment_timestamps is True
        assert capabilities.word_timestamps is False
        assert capabilities.speaker_labels is True
        assert list(capabilities.aligned_diarized_languages) == ["en"]
        assert [(event.text, event.speaker) for event in finals] == [
            ("native", "EPOCH_0001/SPEAKER_00"),
            ("final", "EPOCH_0001/SPEAKER_01"),
        ]
        assert [(event.start_s, event.end_s) for event in finals] == [(0.01, 0.05), (0.05, 0.1)]
        assert len(enricher.calls) == 1
        assert enricher.calls[0][0].size == 1600
        assert enricher.calls[0][1:] == ("native final", "English", 1)
    finally:
        channel.close()
        server.stop(grace=None).wait()


@pytest.mark.parametrize("lang", ["cs", "en"])
def test_qwen_enrichment_limit_or_failure_preserves_speakerless_final(monkeypatch, lang):
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    enricher = _FakeEnricher(fail=lang == "en")
    server = create_server(_FakeQwenBackend(), enricher=enricher, port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{server.bound_port}")
    try:
        events = _transcript_events(
            stream_pb2_grpc.RealtimeASRStub(channel),
            iter([_chunk(lang=lang)]),
        )
        final = events[-1]
        assert final.type == stream_pb2.FINAL
        assert final.text == "native final"
        assert final.speaker == ""
        assert len(enricher.calls) == (1 if lang == "en" else 0)
    finally:
        channel.close()
        server.stop(grace=None).wait()


def test_qwen_epoch_enricher_restores_text_and_assigns_overlap_speakers():
    class Aligner:
        alignment_languages = ("English",)

        def align(self, pcm16, text, language):
            assert text == "Hello, real-time world!"
            assert language == "English"
            return (
                SimpleNamespace(text="Hello", start_time=0.0, end_time=0.6),
                SimpleNamespace(text="realtime", start_time=0.6, end_time=1.2),
                SimpleNamespace(text="world", start_time=1.2, end_time=1.6),
            )

    class Diarizer:
        runtime = "pyannote-audio==test"

        def diarize(self, pcm16):
            return (
                DiarizationSegment(0.0, 1.1, "SPEAKER_00"),
                DiarizationSegment(1.1, 2.0, "SPEAKER_01"),
            )

    segments = QwenEpochEnricher(Aligner(), Diarizer()).enrich(
        np.ones(32000, dtype=np.int16),
        "Hello, real-time world!",
        "English",
        epoch_number=2,
    )

    assert segments == (
        EnrichedSegment(0.0, 1.2, "Hello, real-time", "EPOCH_0002/SPEAKER_00"),
        EnrichedSegment(1.2, 1.6, "world!", "EPOCH_0002/SPEAKER_01"),
    )


def test_qwen_epoch_enricher_assigns_nearest_speaker_across_silence():
    class Aligner:
        alignment_languages = ("English",)

        def align(self, pcm16, text, language):
            return (
                SimpleNamespace(text="Hello", start_time=0.0, end_time=0.4),
                SimpleNamespace(text="world", start_time=0.4, end_time=0.8),
            )

    class Diarizer:
        runtime = "pyannote-audio==test"

        def diarize(self, pcm16):
            return (DiarizationSegment(0.0, 0.3, "SPEAKER_00"),)

    segments = QwenEpochEnricher(Aligner(), Diarizer()).enrich(
        np.ones(16000, dtype=np.int16),
        "Hello world",
        "English",
        epoch_number=1,
    )

    assert segments == (
        EnrichedSegment(0.0, 0.8, "Hello world", "EPOCH_0001/SPEAKER_00"),
    )


def test_qwen_epoch_enricher_ignores_zero_duration_alignment_units():
    class Aligner:
        alignment_languages = ("English",)

        def align(self, pcm16, text, language):
            return (
                SimpleNamespace(text="Hello", start_time=0.0, end_time=0.4),
                SimpleNamespace(text="stalled", start_time=0.4, end_time=0.4),
                SimpleNamespace(text="world", start_time=0.4, end_time=0.8),
            )

    class Diarizer:
        runtime = "pyannote-audio==test"

        def diarize(self, pcm16):
            return (DiarizationSegment(0.0, 1.0, "SPEAKER_00"),)

    segments = QwenEpochEnricher(Aligner(), Diarizer()).enrich(
        np.ones(16000, dtype=np.int16),
        "Hello stalled world",
        "English",
        epoch_number=1,
    )

    assert segments == (
        EnrichedSegment(0.0, 0.8, "Hello stalled world", "EPOCH_0001/SPEAKER_00"),
    )


def test_qwen_rtservice_rolls_epochs_without_public_cutoff_or_duplicate_text(monkeypatch):
    monkeypatch.setenv("QWEN_RTSERVICE_BIND_ADDR", "127.0.0.1")
    monkeypatch.setenv("QWEN_STREAM_EPOCH_SECONDS", "10")
    backend = _EpochQwenBackend()
    enricher = _EpochEnricher()
    server = create_server(backend, enricher=enricher, port=0)
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
        events = _transcript_events(
            stub,
            chunks(),
            session_id="long-qwen-session",
            timeout=10,
        )

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
        assert all(call[0] <= 10 * 16000 for call in enricher.calls)
        assert [call[1] for call in enricher.calls] == [
            f"word-{epoch}" for epoch in range(1, len(final_events) + 1)
        ]
        assert all(event.speaker.startswith("EPOCH_") for event in final_events)
    finally:
        channel.close()
        server.stop(grace=None).wait()
