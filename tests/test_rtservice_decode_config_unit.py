from types import SimpleNamespace

import numpy as np
import pytest

import rtservice.engine as engine_module
from rtservice.engine import (
    DEFAULT_ASR_COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_ASR_TEMPERATURES,
    RealtimeConfig,
    RealtimeModelBundle,
    _parse_asr_compression_ratio_threshold,
    _parse_asr_temperatures,
)


def test_asr_decode_config_defaults():
    """Missing ASR decode settings use the documented production defaults."""

    assert _parse_asr_temperatures(None) == DEFAULT_ASR_TEMPERATURES
    assert _parse_asr_compression_ratio_threshold(None) == DEFAULT_ASR_COMPRESSION_RATIO_THRESHOLD


def test_asr_decode_config_accepts_custom_values():
    """Operators can provide a valid fallback schedule and threshold."""

    assert _parse_asr_temperatures("0, 0.35, 0.7") == (0.0, 0.35, 0.7)
    assert _parse_asr_compression_ratio_threshold(" 3.1 ") == 3.1


@pytest.mark.parametrize(
    "raw",
    ["", "0.0,,0.2", "not-a-number", "nan", "inf", "-0.1", "1.1", "0.5,0.2"],
)
def test_asr_temperature_config_rejects_invalid_values(raw):
    """Invalid temperature schedules fail instead of silently disabling fallback."""

    with pytest.raises(ValueError, match="RT_ASR_TEMPERATURES"):
        _parse_asr_temperatures(raw)


@pytest.mark.parametrize("raw", ["", "not-a-number", "nan", "inf", "0", "-1"])
def test_asr_compression_threshold_rejects_invalid_values(raw):
    """Invalid compression thresholds fail instead of changing decode behavior."""

    with pytest.raises(ValueError, match="RT_ASR_COMPRESSION_RATIO_THRESHOLD"):
        _parse_asr_compression_ratio_threshold(raw)


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
    monkeypatch.setattr(engine_module, "RT_ASR_TEMPERATURES", temperatures)
    monkeypatch.setattr(engine_module, "RT_ASR_COMPRESSION_RATIO_THRESHOLD", compression_threshold)
    monkeypatch.setattr(engine_module, "RT_ASR_SERIALIZE", False)

    fake_model = _FakeWhisperModel()
    bundle = object.__new__(RealtimeModelBundle)
    bundle._cfg = RealtimeConfig(sr=16000, finalize_min_dur_sec=0.1)
    bundle._asr = fake_model

    wave = np.ones(16000, dtype=np.float32)
    assert bundle.asr_text(wave, "en")
    assert bundle.asr_text_partial(wave, "en") == "partial text"

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
