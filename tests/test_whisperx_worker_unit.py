import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Unit tests can run without whisperx installed.
# We will stub the lazy import mechanism.

from whisperx_worker import whisperx_worker


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

    # Pretend we have no GPU so device="cpu"
    monkeypatch.setattr(whisperx_worker.torch.cuda, "is_available", lambda: False)
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
