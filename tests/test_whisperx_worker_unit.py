import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Unit tests can run without whisperx installed.
import importlib.util

if importlib.util.find_spec("whisperx") is None:
    pytest.skip(
        "whisperx not installed in this env; skipping whisperx worker unit tests",
        allow_module_level=True,
    )

from whisperx_worker import whisperx_worker


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
    """
    Make sure each test starts from a clean-ish worker state.
    """
    monkeypatch.setattr(whisperx_worker, "_WHISPERX_MODEL", None, raising=False)
    monkeypatch.setattr(whisperx_worker, "_ALIGN_MODEL", None, raising=False)
    monkeypatch.setattr(whisperx_worker, "_ALIGN_METADATA", None, raising=False)
    yield
    # best-effort reset after
    whisperx_worker._WHISPERX_MODEL = None
    whisperx_worker._ALIGN_MODEL = None
    whisperx_worker._ALIGN_METADATA = None


def test_init_whisperx_respects_env_compute_type(monkeypatch, tmp_path):
    """
    Unit-level check: _init_whisperx picks compute_type from WHISPERX_COMPUTE_TYPE,
    and calls whisperx.load_model with that value.
    """
    calls = {}

    def fake_load_model(name, device, compute_type):
        calls["name"] = name
        calls["device"] = device
        calls["compute_type"] = compute_type
        return DummyWhisperXModel([])

    # Pretend we have no GPU so device="cpu"
    monkeypatch.setattr(whisperx_worker.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(whisperx_worker.whisperx, "load_model", fake_load_model)

    os.environ["WHISPERX_COMPUTE_TYPE"] = "int8_float32"

    whisperx_worker._init_whisperx()

    assert calls["name"] == "medium"
    assert calls["device"] == "cpu"
    assert calls["compute_type"] == "int8_float32"


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
