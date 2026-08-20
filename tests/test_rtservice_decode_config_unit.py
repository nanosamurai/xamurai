from types import SimpleNamespace

import numpy as np
import pytest

import rtservice.engine as engine_module
from rtservice.engine import (
    DEFAULT_ASR_COMPRESSION_RATIO_THRESHOLD,
    DEFAULT_ASR_TEMPERATURES,
    DEFAULT_FINAL_DIAR_MERGE_GAP_SEC,
    DEFAULT_FINAL_DIAR_MIN_TRANSCRIBE_SEC,
    RealtimeConfig,
    RealtimeModelBundle,
    _nonnegative_float_env,
    _parse_asr_temperatures,
)


def test_float_config_uses_defaults_and_accepts_custom_values(monkeypatch):
    """Validated float settings support their defaults and documented custom values."""

    settings = [
        (
            "RT_ASR_COMPRESSION_RATIO_THRESHOLD",
            DEFAULT_ASR_COMPRESSION_RATIO_THRESHOLD,
            False,
            "3.1",
            3.1,
        ),
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


def test_asr_temperature_config_uses_default_and_accepts_custom_values():
    """The temperature schedule keeps its production default and custom parsing."""

    assert _parse_asr_temperatures(None) == DEFAULT_ASR_TEMPERATURES
    assert _parse_asr_temperatures("0, 0.35, 0.7") == (0.0, 0.35, 0.7)


def test_realtime_partial_cadence_defaults_to_one_point_five_seconds():
    """The compute-safe realtime PARTIAL cadence remains the application default."""

    assert engine_module.EMIT_EVERY_SEC == pytest.approx(1.5)
    assert RealtimeConfig().emit_every_sec == pytest.approx(1.5)


@pytest.mark.parametrize(
    "raw",
    ["", "0.0,,0.2", "not-a-number", "nan", "inf", "-0.1", "1.1", "0.5,0.2"],
)
def test_asr_temperature_config_rejects_invalid_values(raw):
    """Invalid temperature schedules fail instead of silently disabling fallback."""

    with pytest.raises(ValueError, match="RT_ASR_TEMPERATURES"):
        _parse_asr_temperatures(raw)


@pytest.mark.parametrize("raw", ["", "not-a-number", "nan", "inf", "-1"])
def test_float_config_rejects_invalid_values(monkeypatch, raw):
    """Malformed validated float settings fail clearly during startup."""

    name = "RT_TEST_NONNEGATIVE_FLOAT"
    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError, match=name):
        _nonnegative_float_env(name, 1.0)


def test_compression_threshold_rejects_zero(monkeypatch):
    """The faster-whisper compression threshold remains strictly positive."""

    monkeypatch.setenv("RT_ASR_COMPRESSION_RATIO_THRESHOLD", "0")
    with pytest.raises(ValueError, match="RT_ASR_COMPRESSION_RATIO_THRESHOLD"):
        _nonnegative_float_env("RT_ASR_COMPRESSION_RATIO_THRESHOLD", 2.4, allow_zero=False)


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
