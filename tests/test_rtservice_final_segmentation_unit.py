import math

import numpy as np
import pytest

from rtservice.engine import (
    RealtimeConfig,
    RealtimeEngine,
    RealtimeSessionProcessor,
    _normalize_and_merge_diarization_turns,
)


def test_normalize_and_merge_keeps_speaker_boundaries_and_large_gaps():
    """Only consecutive same-speaker turns within the configured gap are merged."""

    turns = _normalize_and_merge_diarization_turns(
        [
            {"start": 1.7, "end": 2.5, "speaker": "A"},
            {"start": 2.6, "end": 3.4, "speaker": "B"},
            {"start": 0.0, "end": 0.8, "speaker": "A"},
            {"start": math.nan, "end": 1.0, "speaker": "A"},
            {"start": -1.0, "end": -0.5, "speaker": "A"},
            {"start": 3.5, "end": 4.5, "speaker": ""},
        ],
        window_duration_sec=4.0,
        merge_gap_sec=0.75,
    )

    assert turns == [
        {"start": 0.0, "end": 0.8, "speaker": "A"},
        {"start": 1.7, "end": 2.5, "speaker": "A"},
        {"start": 2.6, "end": 3.4, "speaker": "B"},
    ]


def _make_processor(*, diarization, asr_calls, speaker_calls):
    """Create a lightweight FINAL processor that records ASR and speaker-map waveforms."""

    cfg = RealtimeConfig(
        sr=1000,
        window_sec=4.0,
        overlap_sec=0.0,
        partial_enable=False,
        finalize_min_dur_sec=0.25,
        final_diar_merge_gap_sec=0.75,
        final_diar_min_transcribe_sec=0.7,
        key_resolution_sec=0.001,
    )

    def gate_fn(_wave):
        return True

    def diarize_fn(_wave):
        return list(diarization)

    def asr_fn(wave, _lang):
        asr_calls.append(wave.copy())
        return f"decoded-{wave.size}"

    def map_speaker_fn(_tenant_id, diar_label, wave):
        speaker_calls.append((diar_label, wave.copy()))
        return f"mapped-{diar_label}"

    processor = RealtimeSessionProcessor(
        cfg=cfg,
        partial_gate_fn=gate_fn,
        window_gate_fn=gate_fn,
        diarize_fn=diarize_fn,
        asr_fn=asr_fn,
        map_speaker_fn=map_speaker_fn,
    )
    return cfg, RealtimeEngine(cfg=cfg, processor=processor)


def _full_window_pcm(cfg):
    """Return one non-silent PCM16 window for the lightweight processor."""

    return np.arange(1, cfg.window_samples + 1, dtype=np.int16).tobytes()


def test_tiny_same_speaker_turns_merge_before_final_asr_and_speaker_mapping():
    """The observed sub-second pattern becomes one contextual FINAL decode."""

    asr_calls = []
    speaker_calls = []
    diarization = [
        {"start": 0.0, "end": 0.253, "speaker": "SPEAKER_00"},
        {"start": 0.633, "end": 1.527, "speaker": "SPEAKER_00"},
        {"start": 2.147, "end": 2.485, "speaker": "SPEAKER_00"},
        {"start": 2.725, "end": 3.7, "speaker": "SPEAKER_00"},
    ]
    cfg, engine = _make_processor(
        diarization=diarization,
        asr_calls=asr_calls,
        speaker_calls=speaker_calls,
    )

    results = engine.feed("session", _full_window_pcm(cfg), tenant_id="tenant", lang="en")

    assert len(results) == 1
    assert (results[0].start_s, results[0].end_s) == (0.0, 3.7)
    assert results[0].speaker == "mapped-SPEAKER_00"
    assert results[0].text == "decoded-3700"
    assert [wave.size for wave in asr_calls] == [3700]
    assert [(speaker, wave.size) for speaker, wave in speaker_calls] == [("SPEAKER_00", 3700)]
    assert engine._sessions[("tenant", "session")].emitted_keys == {(0.0, 3.7, "SPEAKER_00")}


def test_residual_short_turn_never_reaches_asr_or_speaker_mapping():
    """An isolated unsafe turn is suppressed while a valid turn still emits a FINAL."""

    asr_calls = []
    speaker_calls = []
    cfg, engine = _make_processor(
        diarization=[
            {"start": 0.0, "end": 0.3, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "speaker": "B"},
        ],
        asr_calls=asr_calls,
        speaker_calls=speaker_calls,
    )

    results = engine.feed("session", _full_window_pcm(cfg), tenant_id="tenant", lang="en")

    assert len(results) == 1
    assert (results[0].start_s, results[0].end_s, results[0].speaker) == (
        1.0,
        2.0,
        "mapped-B",
    )
    assert [wave.size for wave in asr_calls] == [1000]
    assert [(speaker, wave.size) for speaker, wave in speaker_calls] == [("B", 1000)]


def test_all_suppressed_turns_use_one_full_window_final_fallback():
    """A window containing only unsafe diarization turns retains the existing fallback."""

    asr_calls = []
    speaker_calls = []
    cfg, engine = _make_processor(
        diarization=[
            {"start": 0.0, "end": 0.3, "speaker": "A"},
            {"start": 1.0, "end": 1.4, "speaker": "B"},
        ],
        asr_calls=asr_calls,
        speaker_calls=speaker_calls,
    )

    results = engine.feed("session", _full_window_pcm(cfg), tenant_id="tenant", lang="en")

    assert len(results) == 1
    assert (results[0].start_s, results[0].end_s, results[0].speaker) == (0.0, 4.0, None)
    assert results[0].text == "decoded-4000"
    assert [wave.size for wave in asr_calls] == [4000]
    assert speaker_calls == []
