from types import SimpleNamespace

import numpy as np
import pytest

from rtservice.engine import (
    DEFAULT_FINAL_DIAR_MERGE_GAP_SEC,
    DEFAULT_FINAL_DIAR_MIN_TRANSCRIBE_SEC,
    _nonnegative_float_env,
)
from rtservice.providers import LocalFasterWhisperProvider, WindowRequest


def test_float_config_uses_defaults_and_accepts_custom_values(monkeypatch):
    """Validated float settings support their defaults and documented custom values."""

    settings = [
        (
            "RT_FINAL_DIAR_MERGE_GAP_SEC",
            DEFAULT_FINAL_DIAR_MERGE_GAP_SEC,
            True,
            "0.5",
            0.5,
        ),
        (
            "RT_FINAL_DIAR_MIN_TRANSCRIBE_SEC",
            DEFAULT_FINAL_DIAR_MIN_TRANSCRIBE_SEC,
            True,
            "0",
            0.0,
        ),
    ]
    for name, default, allow_zero, custom, expected in settings:
        monkeypatch.delenv(name, raising=False)
        assert _nonnegative_float_env(name, default, allow_zero=allow_zero) == default
        monkeypatch.setenv(name, custom)
        assert _nonnegative_float_env(name, default, allow_zero=allow_zero) == expected


@pytest.mark.parametrize("raw", ["", "not-a-number", "nan", "inf", "-1"])
def test_float_config_rejects_invalid_values(monkeypatch, raw):
    """Malformed validated float settings fail clearly during startup."""

    name = "RT_TEST_NONNEGATIVE_FLOAT"
    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError, match=name):
        _nonnegative_float_env(name, 1.0)


class _FakeWhisperModel:
    """Record faster-whisper calls while returning lightweight fake segments."""

    def __init__(self):
        self.calls = []

    def transcribe(self, _wave, **kwargs):
        self.calls.append(kwargs)
        if kwargs["word_timestamps"]:
            segment = SimpleNamespace(
                words=[SimpleNamespace(word="hello"), SimpleNamespace(word=" world")],
                text="",
            )
        else:
            segment = SimpleNamespace(words=None, text="partial text")
        return iter([segment]), SimpleNamespace()


def test_final_and_partial_decode_use_native_repetition_fallback(monkeypatch):
    """FINAL and PARTIAL decoding pass the native fallback settings unchanged."""

    temperatures = (0.0, 0.3, 0.6)
    compression_threshold = 2.7
    monkeypatch.setenv("RT_ASR_TEMPERATURES", ",".join(str(value) for value in temperatures))
    monkeypatch.setenv("RT_ASR_COMPRESSION_RATIO_THRESHOLD", str(compression_threshold))
    monkeypatch.setenv("RT_ASR_SERIALIZE", "false")

    fake_model = _FakeWhisperModel()
    provider = LocalFasterWhisperProvider(model=fake_model)
    pcm = np.ones(16000, dtype=np.int16).tobytes()
    final = provider.transcribe_window(WindowRequest(pcm, 16000, "en", 0, 16000, False))
    partial = provider.transcribe_window(WindowRequest(pcm, 16000, "en", 0, 16000, True))
    assert final.text
    assert partial.text == "partial text"

    final_call, partial_call = fake_model.calls
    assert final_call["beam_size"] == 5
    assert final_call["word_timestamps"] is True
    assert partial_call["beam_size"] == 1
    assert partial_call["word_timestamps"] is False

    for call in fake_model.calls:
        assert call["temperature"] == temperatures
        assert call["compression_ratio_threshold"] == compression_threshold
        assert call["condition_on_previous_text"] is False
        assert call["vad_filter"] is False
