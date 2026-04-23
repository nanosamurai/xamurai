import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Unit tests can run without whisperx installed.
# We will stub the lazy import mechanism.

from whisperx_worker import whisperx_worker


class _DummyProducer:
    def __init__(self):
        self.produced = []

    def produce(self, *, topic, key, value, headers=None):
        self.produced.append({"topic": topic, "key": key, "value": value, "headers": headers})

    def poll(self, _timeout):
        return None


class DummyWhisperXModule:
    def __init__(self):
        self._load_model_calls = []

    def load_model(self, name, device, compute_type, vad_method=None):
        self._load_model_calls.append((name, device, compute_type, vad_method))
        return DummyWhisperXModel([])

    def load_audio(self, path):
        return np.zeros(16000, dtype=np.float32)

    def load_align_model(self, language_code, device):
        return object(), {"language": language_code}

    def align(self, *args, **kwargs):
        return {"segments": []}


class DummyWhisperXModel:
    def __init__(self, segments: List[Dict[str, Any]], language: str = "en"):
        self._segments = segments
        self._language = language

    def transcribe(self, audio, batch_size=16, language=None):
        # We ignore audio; return a pre-canned result
        return {
            "segments": self._segments,
            "language": self._language,
        }


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch):
    """Make sure each test starts from a clean-ish worker state."""

    # stub whisperx module and bypass importlib
    dummy_whisperx = DummyWhisperXModule()
    monkeypatch.setattr(whisperx_worker, "whisperx", dummy_whisperx, raising=False)
    monkeypatch.setattr(whisperx_worker, "_ensure_whisperx_imported", lambda: None, raising=False)

    monkeypatch.setattr(whisperx_worker, "_WHISPERX_MODEL", None, raising=False)
    monkeypatch.setattr(whisperx_worker, "_ALIGN_MODEL", None, raising=False)
    monkeypatch.setattr(whisperx_worker, "_ALIGN_METADATA", None, raising=False)
    yield
    whisperx_worker._WHISPERX_MODEL = None
    whisperx_worker._ALIGN_MODEL = None
    whisperx_worker._ALIGN_METADATA = None


def test_merge_diarization_turns_merges_same_speaker_and_drops_tiny() -> None:
    diar = [
        whisperx_worker.DiarizationSegment(start_s=0.0, end_s=0.3, speaker="A"),  # tiny -> drop
        whisperx_worker.DiarizationSegment(start_s=0.3, end_s=1.0, speaker="A"),
        whisperx_worker.DiarizationSegment(start_s=1.05, end_s=2.0, speaker="A"),  # merge (gap 0.05)
        whisperx_worker.DiarizationSegment(start_s=2.2, end_s=3.0, speaker="B"),
        whisperx_worker.DiarizationSegment(start_s=3.05, end_s=3.6, speaker="B"),  # merge (gap 0.05)
    ]

    out = whisperx_worker._merge_diarization_turns(
        diar,
        min_turn_sec=0.5,
        merge_gap_sec=0.15,
        max_turns=100,
    )

    assert [(round(s.start_s, 2), round(s.end_s, 2), s.speaker) for s in out] == [
        (0.3, 2.0, "A"),
        (2.2, 3.6, "B"),
    ]


def test_run_whisperx_diarized_split_mode_uses_diar_turns(monkeypatch, tmp_path):
    """When split-mode is enabled and diarization yields multiple speakers,
    expect multiple output segments with speaker labels even if baseline ASR is coarse.
    """

    # Baseline coarse ASR: a single long segment.
    dummy_model = DummyWhisperXModel([
        {"start": 0.0, "end": 10.0, "text": "Hello world"},
    ])

    # avoid real model init
    monkeypatch.setattr(whisperx_worker, "_init_whisperx", lambda lang_hint=None: None)
    whisperx_worker._WHISPERX_MODEL = dummy_model

    # Ensure split mode enabled.
    monkeypatch.setattr(whisperx_worker, "_DIAR_SPLIT_MODE", True, raising=False)
    monkeypatch.setattr(whisperx_worker, "_DIAR_SPLIT_MAX_AUDIO_SEC", 90.0, raising=False)
    monkeypatch.setattr(whisperx_worker, "_DIAR_SPLIT_MIN_TURN_SEC", 0.5, raising=False)
    monkeypatch.setattr(whisperx_worker, "_DIAR_SPLIT_MERGE_GAP_SEC", 0.15, raising=False)
    monkeypatch.setattr(whisperx_worker, "_DIAR_SPLIT_MAX_TURNS", 40, raising=False)

    # Fake diarization output: A then B.
    monkeypatch.setattr(
        whisperx_worker,
        "_diarize_audio",
        lambda _audio: [
            whisperx_worker.DiarizationSegment(start_s=0.0, end_s=2.0, speaker="A"),
            whisperx_worker.DiarizationSegment(start_s=2.0, end_s=4.0, speaker="B"),
        ],
        raising=False,
    )

    # Pretend diar pipeline exists so run_whisperx_diarized treats diarization as enabled.
    monkeypatch.setattr(whisperx_worker, "_ENABLE_DIARIZATION", True, raising=False)
    monkeypatch.setattr(whisperx_worker, "_DIAR_PIPE", object(), raising=False)

    # Split path requires torch available (otherwise worker falls back).
    monkeypatch.setattr(whisperx_worker, "torch", object(), raising=False)

    # avoid real audio load
    monkeypatch.setattr(
        whisperx_worker.whisperx,
        "load_audio",
        lambda _p: np.zeros(10 * 16000, dtype=np.float32),
    )

    # Also patch sf.read fallback just in case
    monkeypatch.setattr(
        whisperx_worker.sf,
        "read",
        lambda _p, dtype=None: (np.zeros(10 * 16000, dtype=np.float32), 16000),
    )

    # Make per-turn transcribe return different text segments so we can see they were invoked.
    calls = {"n": 0}

    def _fake_transcribe_arr(_arr, *, lang):
        calls["n"] += 1
        if calls["n"] == 1:
            return "aaa", [(0.0, 1.0, "aaa")]
        return "bbb", [(0.0, 1.0, "bbb")]

    monkeypatch.setattr(whisperx_worker, "_transcribe_audio_array", _fake_transcribe_arr, raising=False)

    wav_path = str(tmp_path / "dummy.wav")
    Path(wav_path).write_bytes(b"")

    full_text, out_segments = whisperx_worker.run_whisperx_diarized(
        wav_path,
        tenant="t",
        lang="en",
        use_alignment=False,
    )

    assert calls["n"] == 2, "Expected per-turn transcription to run twice"
    assert full_text == "aaa bbb"
    # Expect 2 segments with speakers A and B.
    assert [(round(s0, 2), round(s1, 2), txt, spk) for (s0, s1, txt, spk) in out_segments] == [
        (0.0, 1.0, "aaa", "A"),
        (2.0, 3.0, "bbb", "B"),
    ]


def test_init_whisperx_respects_env_compute_type(monkeypatch, tmp_path):
    """
    Unit-level check: _init_whisperx picks compute_type from WHISPERX_COMPUTE_TYPE,
    and calls whisperx.load_model with that value.
    """
    calls = {}

    def fake_load_model(name, device, compute_type, vad_method=None):
        calls["name"] = name
        calls["device"] = device
        calls["compute_type"] = compute_type
        calls["vad_method"] = vad_method
        return DummyWhisperXModel([])

    # Provide a minimal torch stub so _init_whisperx can decide on device.
    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

    monkeypatch.setattr(whisperx_worker, "torch", _Torch(), raising=False)
    monkeypatch.setattr(whisperx_worker.whisperx, "load_model", fake_load_model)

    os.environ["WHISPERX_COMPUTE_TYPE"] = "int8_float32"

    whisperx_worker._init_whisperx()

    assert calls["name"] == "medium"
    assert calls["device"] == "cpu"
    assert calls["compute_type"] == "int8_float32"
    assert calls["vad_method"] is None


def test_run_whisperx_no_alignment(monkeypatch, tmp_path):
    """
    Unit-level check: when use_alignment=False, run_whisperx returns segments
    directly from the ASR result and concatenates text.
    """
    # fake ASR segments
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.0, "text": "world"},
        {"start": 2.0, "end": 3.0, "text": ""},
    ]

    dummy_model = DummyWhisperXModel(segments, language="en")

    # avoid real model init
    monkeypatch.setattr(whisperx_worker, "_init_whisperx", lambda lang_hint=None: None)

    # patch model + load_audio
    whisperx_worker._WHISPERX_MODEL = dummy_model
    monkeypatch.setattr(
        whisperx_worker.whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32)
    )

    # create temp wav path (we won't actually read from it in fake load_audio)
    wav_path = str(tmp_path / "dummy.wav")
    Path(wav_path).write_bytes(b"")

    full_text, out_segments = whisperx_worker.run_whisperx(
        wav_path, lang="en", use_alignment=False
    )

    assert full_text == "Hello world"
    assert len(out_segments) == 2
    assert out_segments[0][0] == pytest.approx(0.0)
    assert out_segments[0][1] == pytest.approx(1.0)
    assert out_segments[0][2] == "Hello"
    assert out_segments[1][2] == "world"


def test_run_whisperx_diarized_smoke_no_diarization(monkeypatch, tmp_path):
    """Unit-level check: run_whisperx_diarized falls back to run_whisperx when
    diarization is disabled/unavailable.

    This keeps unit tests light (no pyannote required).
    """

    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
    ]

    dummy_model = DummyWhisperXModel(segments, language="en")

    # avoid real model init
    monkeypatch.setattr(whisperx_worker, "_init_whisperx", lambda lang_hint=None: None)

    # patch model + load_audio
    whisperx_worker._WHISPERX_MODEL = dummy_model
    monkeypatch.setattr(
        whisperx_worker.whisperx, "load_audio", lambda path: np.zeros(16000, dtype=np.float32)
    )

    # force diarization off
    monkeypatch.setattr(whisperx_worker, "_ENABLE_DIARIZATION", False, raising=False)

    wav_path = str(tmp_path / "dummy.wav")
    Path(wav_path).write_bytes(b"")

    full_text, out_segments = whisperx_worker.run_whisperx_diarized(
        wav_path,
        tenant="t",
        lang="en",
        use_alignment=False,
    )

    assert full_text == "Hello"
    assert out_segments == [(0.0, 1.0, "Hello", "")]


def test_run_inference_and_publish_emits_single_refined_window_event(monkeypatch, tmp_path):
    """Regression: whisperx_worker should publish ONE RefinedEvent per slice,
    with segments populated (window transcript semantics).
    """

    # Make inference return 2 segments.
    monkeypatch.setattr(
        whisperx_worker,
        "run_whisperx_diarized",
        lambda *_a, **_kw: ("hello world", [(0.0, 1.0, "hello", "A"), (1.0, 2.0, "world", "B")]),
        raising=False,
    )

    # Avoid creating/cleaning a real tmp wav.
    monkeypatch.setattr(whisperx_worker.sf, "write", lambda *_a, **_kw: None, raising=False)
    monkeypatch.setattr(whisperx_worker.os, "unlink", lambda *_a, **_kw: None, raising=False)

    # Avoid trace context complications.
    monkeypatch.setattr(whisperx_worker, "extracted_context_from_headers", lambda _h: __import__("contextlib").nullcontext())
    monkeypatch.setattr(whisperx_worker, "with_current_trace_context", lambda: [])

    producer = _DummyProducer()

    job = {
        "session_id": "s1",
        "pcm16": np.zeros(16000, dtype="<i2"),
        "base_start_s": 10.0,
        "window_sec": 20.0,
        "lang": "en",
        "tenant_id": "t1",
        "bff_origin_uri": "http://bff",
        "trace_headers": [],
        "flush_reason": "slice",
        "slice_index": 3,
    }

    whisperx_worker._run_inference_and_publish(job=job, producer=producer)

    assert len(producer.produced) == 1
    msg = producer.produced[0]
    ev = whisperx_worker.stream_pb2.RefinedEvent()
    ev.ParseFromString(msg["value"])

    assert ev.session_id == "s1"
    assert ev.start_s == pytest.approx(10.0)
    assert ev.window_sec == pytest.approx(20.0)
    assert ev.slice_index == 3
    assert ev.flush_reason == "slice"
    assert len(ev.segments) == 2
    assert [s.speaker for s in ev.segments] == ["A", "B"]
    assert ev.text.strip() == "hello world"


def test_should_evict_idle_session_true_when_idle_and_polling_recently():
    now = 1000.0
    last_activity = {"s": now - 31.0}
    last_poll_s = now - 0.5
    assert whisperx_worker._should_evict_idle_session(
        "s",
        now_s=now,
        last_activity_s=last_activity,
        last_poll_s=last_poll_s,
        idle_sec=30.0,
    )


def test_should_not_evict_when_not_idle():
    now = 1000.0
    last_activity = {"s": now - 10.0}
    last_poll_s = now - 0.5
    assert not whisperx_worker._should_evict_idle_session(
        "s",
        now_s=now,
        last_activity_s=last_activity,
        last_poll_s=last_poll_s,
        idle_sec=30.0,
    )


def test_should_not_evict_when_consumer_poll_was_blocked_by_inference():
    """Regression test for refined-timing reset bug.

    If inference blocks the main loop for > idle_sec, wall-clock-based eviction is
    unsafe because we may be behind on consuming audio.
    """

    now = 1000.0
    last_activity = {"s": now - 31.0}
    # last poll was also a long time ago -> indicates main loop was blocked
    last_poll_s = now - 120.0
    assert not whisperx_worker._should_evict_idle_session(
        "s",
        now_s=now,
        last_activity_s=last_activity,
        last_poll_s=last_poll_s,
        idle_sec=30.0,
    )


def test_decoupled_runtime_job_collection_scheduler_only(monkeypatch):
    """Unit-level sanity check for Phase 2 decoupled runtime.

    This test intentionally avoids Kafka/WhisperX dependencies.

    We verify that `_queue_get_many` collects multiple jobs when configured,
    which is the foundation for future true multi-audio batching.
    """

    from whisperx_worker.decoupled_runtime import _queue_get_many
    from queue import Queue

    q: "Queue[dict]" = Queue()
    q.put({"session_id": "a"})
    q.put({"session_id": "b"})

    batch = _queue_get_many(q, max_items=2, max_wait_ms=1)
    assert [j["session_id"] for j in batch] == ["a", "b"]
