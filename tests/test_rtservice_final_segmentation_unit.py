import math

import numpy as np
import pytest

from rtservice.engine import (
    RealtimeConfig,
    RealtimeEngine,
    RealtimeSessionProcessor,
    _normalize_and_merge_diarization_turns,
)
from rtservice.providers import ProviderCandidate, ProviderWord


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


def _make_processor(
    *,
    diarization,
    words,
    asr_calls,
    speaker_calls,
    window_sec=4.0,
    overlap_sec=0.0,
):
    """Create a FINAL processor whose fake provider repeats words in overlapping context."""

    cfg = RealtimeConfig(
        sr=1000,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
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

    def asr_fn(wave, lang, start_sample, partial):
        asr_calls.append((start_sample, wave.copy(), partial))
        end_sample = start_sample + wave.size
        visible = [
            ProviderWord(text, int(start_s * cfg.sr), int(end_s * cfg.sr))
            for text, start_s, end_s in words
            if int(end_s * cfg.sr) > start_sample and int(start_s * cfg.sr) < end_sample
        ]
        return ProviderCandidate(
            text="".join(word.text for word in visible).strip(),
            language=lang,
            start_sample=start_sample,
            end_sample=end_sample,
            terminal=not partial,
            provider_sequence=0,
            words=tuple(visible) if not partial else (),
        )

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
    """Merged diarization still maps a contiguous set of provider words once."""

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
        words=[
            ("one", 0.1, 0.2),
            (" two", 0.8, 0.9),
            (" three", 2.2, 2.3),
            (" four", 3.0, 3.1),
        ],
        asr_calls=asr_calls,
        speaker_calls=speaker_calls,
    )

    results = engine.feed("session", _full_window_pcm(cfg), tenant_id="tenant", lang="en")

    assert len(results) == 1
    assert (results[0].start_s, results[0].end_s) == (0.1, 3.1)
    assert results[0].speaker == "mapped-SPEAKER_00"
    assert results[0].text == "one two three four"
    assert [(start, wave.size, partial) for start, wave, partial in asr_calls] == [(0, 4000, False)]
    assert [(speaker, wave.size) for speaker, wave in speaker_calls] == [("SPEAKER_00", 3700)]
    state = engine._sessions[("tenant", "session")]
    assert state.next_commit_sample == 4000
    assert state.pcm16_buffer == bytearray()


def test_short_turn_uses_raw_label_while_long_turn_keeps_enrollment_mapping():
    """A short turn no longer loses its word but avoids an unsafe embedding call."""

    asr_calls = []
    speaker_calls = []
    cfg, engine = _make_processor(
        diarization=[
            {"start": 0.0, "end": 0.3, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "speaker": "B"},
        ],
        words=[("short", 0.1, 0.2), (" long", 1.1, 1.2)],
        asr_calls=asr_calls,
        speaker_calls=speaker_calls,
    )

    results = engine.feed("session", _full_window_pcm(cfg), tenant_id="tenant", lang="en")

    assert [(result.text, result.speaker) for result in results] == [
        ("short", "A"),
        ("long", "mapped-B"),
    ]
    assert [(start, wave.size) for start, wave, _ in asr_calls] == [(0, 4000)]
    assert [(speaker, wave.size) for speaker, wave in speaker_calls] == [("B", 1000)]


def test_missing_diarization_emits_owned_words_without_a_speaker():
    """ASR finals remain available when pyannote yields no turns."""

    asr_calls = []
    speaker_calls = []
    cfg, engine = _make_processor(
        diarization=[],
        words=[("decoded", 1.0, 2.0)],
        asr_calls=asr_calls,
        speaker_calls=speaker_calls,
    )

    results = engine.feed("session", _full_window_pcm(cfg), tenant_id="tenant", lang="en")

    assert len(results) == 1
    assert (results[0].start_s, results[0].end_s, results[0].speaker) == (1.0, 2.0, None)
    assert results[0].text == "decoded"
    assert [(start, wave.size) for start, wave, _ in asr_calls] == [(0, 4000)]
    assert speaker_calls == []


def test_contextual_word_midpoints_own_seam_words_once_and_bound_memory():
    """Words repeated by adjacent contextual decodes have one deterministic owner."""

    asr_calls = []
    speaker_calls = []
    cfg, engine = _make_processor(
        diarization=[{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}],
        words=[
            ("alpha", 9.4, 9.8),
            (" boundary", 9.8, 10.2),
            (" omega", 10.2, 10.6),
            (" later", 15.0, 15.4),
        ],
        asr_calls=asr_calls,
        speaker_calls=speaker_calls,
        window_sec=10.0,
        overlap_sec=1.0,
    )

    pcm = np.ones(21 * cfg.sr, dtype=np.int16).tobytes()
    results = engine.feed("session", pcm, tenant_id="tenant", lang="en")

    assert [result.text for result in results] == ["alpha", "boundary omega later"]
    assert sum(result.text.count("boundary") for result in results) == 1
    assert [(start, wave.size) for start, wave, _ in asr_calls] == [
        (0, 11000),
        (9000, 12000),
    ]
    state = engine._sessions[("tenant", "session")]
    assert state.next_commit_sample == 20000
    assert state.buffer_start_sample == 19000
    assert len(state.pcm16_buffer) == 2 * 2000


def test_eof_commits_pending_context_and_short_tail_once():
    """EOF replaces missing future context and does not drop a seam word or tail."""

    asr_calls = []
    speaker_calls = []
    cfg, engine = _make_processor(
        diarization=[],
        words=[
            ("alpha", 9.4, 9.8),
            (" boundary", 9.8, 10.2),
            (" end", 10.2, 10.4),
        ],
        asr_calls=asr_calls,
        speaker_calls=speaker_calls,
        window_sec=10.0,
        overlap_sec=1.0,
    )

    pcm = np.ones(int(10.5 * cfg.sr), dtype=np.int16).tobytes()
    assert engine.feed("session", pcm, tenant_id="tenant", lang="en") == []
    results = engine.feed("session", b"", tenant_id="tenant", lang="en")

    assert [result.text for result in results] == ["alpha", "boundary end"]
    assert sum(result.text.count("boundary") for result in results) == 1
    assert [(start, wave.size) for start, wave, _ in asr_calls] == [
        (0, 10500),
        (9000, 1500),
    ]
    state = engine._sessions[("tenant", "session")]
    assert state.pcm16_buffer == bytearray()
    assert state.next_commit_sample == 10500


def test_unusable_provider_word_timing_falls_back_to_exact_commit_decode():
    """A malformed timing cannot leak the contextual transcript into both commits."""

    asr_calls = []
    speaker_calls = []
    cfg, engine = _make_processor(
        diarization=[],
        words=[("decoded", 1.0, 1.0)],
        asr_calls=asr_calls,
        speaker_calls=speaker_calls,
        window_sec=4.0,
        overlap_sec=1.0,
    )

    pcm = np.ones(5 * cfg.sr, dtype=np.int16).tobytes()
    results = engine.feed("session", pcm, tenant_id="tenant", lang="en")

    assert [(result.start_s, result.end_s, result.text, result.speaker) for result in results] == [
        (0.0, 4.0, "decoded", None)
    ]
    assert [(start, wave.size) for start, wave, _ in asr_calls] == [
        (0, 5000),
        (0, 4000),
    ]
