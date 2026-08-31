import numpy as np
from types import SimpleNamespace

from rtservice import engine as engine_module
from rtservice.engine import (
    RealtimeConfig,
    RealtimeEngine,
    RealtimeSessionProcessor,
)
from rtservice.providers import (
    LocalFasterWhisperProvider,
    ProviderCandidate,
    ProviderRegistry,
    ProviderWord,
)


def _candidate(text, wave, lang, start_sample, partial):
    words = ()
    if text and not partial and wave.size:
        words = (ProviderWord(text, start_sample, start_sample + wave.size),)
    return ProviderCandidate(
        text=text,
        language=lang,
        start_sample=start_sample,
        end_sample=start_sample + wave.size,
        terminal=not partial,
        provider_sequence=0,
        words=words,
    )


def _lag_guard_engine(asr_fn):
    cfg = RealtimeConfig(
        sr=1000,
        window_sec=1.0,
        overlap_sec=0.0,
        partial_enable=True,
        emit_every_sec=0.1,
        partial_stability_repeats=1,
        partial_min_buffer_sec=0.0,
        partial_min_transcribe_sec=0.0,
        partial_max_behind_sec=2.0,
        partial_idle_reset_sec=3.0,
        finalize_min_dur_sec=0.1,
        key_resolution_sec=0.1,
    )
    processor = RealtimeSessionProcessor(
        cfg=cfg,
        partial_gate_fn=lambda _wave: True,
        window_gate_fn=lambda _wave: True,
        diarize_fn=lambda _wave: [],
        asr_fn=asr_fn,
        map_speaker_fn=lambda _tenant_id, diar_label, _wave: diar_label,
    )
    return cfg, RealtimeEngine(cfg=cfg, processor=processor)


def test_rtservice_engine_isolates_sessions_without_audio_mix():
    """Unit test that proves per-session state isolation.

    We inject a lightweight processor (no torch/pyannote).

    Scenario:
    - session A feeds enough audio to produce one window result
    - session B feeds just a bit of audio; it MUST NOT produce results just
      because session A has a filled buffer

    This specifically guards against the previous bug where one global rolling
    buffer mixed all sessions.
    """

    cfg = RealtimeConfig(sr=16000, window_sec=1.0, overlap_sec=0.0, finalize_min_dur_sec=0.1, key_resolution_sec=0.1)

    def gate_fn(_w):
        return True

    def diarize_fn(_w):
        return [{"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]

    def asr_fn(chunk, lang, start_sample, partial):
        return _candidate("hello", chunk, lang, start_sample, partial)

    def map_speaker_fn(_tenant_id, diar_label, _chunk):
        return diar_label

    processor = RealtimeSessionProcessor(
        cfg=cfg,
        partial_gate_fn=gate_fn,
        window_gate_fn=gate_fn,
        diarize_fn=diarize_fn,
        asr_fn=asr_fn,
        map_speaker_fn=map_speaker_fn,
    )

    engine = RealtimeEngine(cfg=cfg, processor=processor)

    # Build 1s of PCM16 audio (any values)
    pcm16 = (np.ones(int(cfg.sr * cfg.window_sec), dtype=np.int16)).tobytes()

    out_a = engine.feed("sA", pcm16, tenant_id="t1", lang="en")
    assert len(out_a) == 1

    # session B provides only half window; should not produce output
    pcm16_half = (np.ones(int(cfg.sr * cfg.window_sec / 2), dtype=np.int16)).tobytes()
    out_b = engine.feed("sB", pcm16_half, tenant_id="t1", lang="en")
    assert out_b == []


def test_rtservice_engine_per_session_overrides_affect_partial_emission():
    """Unit test: per-session overrides should influence PARTIAL cadence.

    We use a lightweight processor and ASR function that always returns text.

    Scenario:
    - session A uses emit_every_sec=0.2 (more frequent partials)
    - session B uses emit_every_sec=0.6 (less frequent)
    - after feeding the same amount of audio (<window), A should have emitted a
      PARTIAL while B should not.
    """

    cfg = RealtimeConfig(
        sr=16000,
        window_sec=1.0,
        overlap_sec=0.0,
        emit_every_sec=0.5,
        partial_enable=True,
        partial_stability_repeats=1,
        partial_min_buffer_sec=0.0,
        partial_min_transcribe_sec=0.0,
        finalize_min_dur_sec=0.1,
        key_resolution_sec=0.1,
    )

    def gate_fn(_w):
        return True

    def diarize_fn(_w):
        # No diarization segments: forces fallback final path only when window completes.
        return []

    def map_speaker_fn(_tenant_id, diar_label, _chunk):
        return diar_label

    class _FakeBundle:
        def gate_window(self, w):
            return gate_fn(w)

        def gate_speech(self, w):
            return gate_fn(w)

        def diarize_window(self, w):
            return diarize_fn(w)

        def map_speaker_label(self, tenant_id, diar_label, chunk):
            return map_speaker_fn(tenant_id, diar_label, chunk)

    class _FakeWhisper:
        def transcribe(self, _wave, **_kwargs):
            return iter([SimpleNamespace(words=None, text="hello")]), SimpleNamespace()

    # IMPORTANT: we do NOT inject a fixed processor here, because per-session
    # overrides are implemented by selecting a processor per effective cfg.
    provider = LocalFasterWhisperProvider(model=_FakeWhisper())
    engine = RealtimeEngine(
        cfg=cfg,
        model_bundle=_FakeBundle(),
        provider_registry=ProviderRegistry((provider,), default_profile_id=provider.profile_id),
    )

    # Feed 0.3s worth of audio; should trigger partial for emit_every_sec=0.2.
    pcm16_03 = (np.ones(int(cfg.sr * 0.3), dtype=np.int16)).tobytes()

    out_a = engine.feed("sA", pcm16_03, tenant_id="t1", lang="en", rt_emit_every_sec=0.2)
    out_b = engine.feed("sB", pcm16_03, tenant_id="t1", lang="en", rt_emit_every_sec=0.6)

    assert any((not r.is_final) for r in out_a), "Expected PARTIAL for session A"
    assert not any((not r.is_final) for r in out_b), "Did not expect PARTIAL for session B"


def test_rtservice_partial_min_transcribe_sec_suppresses_too_short_audio():
    """Regression test: avoid early hallucinated PARTIALs.

    Faster-Whisper often hallucinates on very short audio. We should not even
    run ASR until at least `partial_min_transcribe_sec` of audio is available
    for the lookback chunk.

    Scenario:
    - min_buffer allows early emission
    - emit_every allows emission
    - but min_transcribe requires >= 1.0s
    - feeding only 0.8s must yield no PARTIAL
    """

    cfg = RealtimeConfig(
        sr=16000,
        window_sec=5.0,
        overlap_sec=0.5,
        partial_enable=True,
        emit_every_sec=0.2,
        partial_stability_repeats=1,
        partial_min_buffer_sec=0.0,
        partial_lookback_sec=2.0,
        partial_min_transcribe_sec=1.0,
        finalize_min_dur_sec=0.1,
        key_resolution_sec=0.1,
    )

    def gate_fn(_w):
        return True

    def diarize_fn(_w):
        return []

    def asr_fn(chunk, lang, start_sample, partial):
        return _candidate("hello", chunk, lang, start_sample, partial)

    def map_speaker_fn(_tenant_id, diar_label, _chunk):
        return diar_label

    processor = RealtimeSessionProcessor(
        cfg=cfg,
        partial_gate_fn=gate_fn,
        window_gate_fn=gate_fn,
        diarize_fn=diarize_fn,
        asr_fn=asr_fn,
        map_speaker_fn=map_speaker_fn,
    )

    engine = RealtimeEngine(cfg=cfg, processor=processor)

    pcm16_08 = (np.ones(int(cfg.sr * 0.8), dtype=np.int16)).tobytes()
    out = engine.feed("s", pcm16_08, tenant_id="t1", lang="en")
    assert not any((not r.is_final) for r in out)


def test_rtservice_partials_are_cumulative_within_window_by_default():
    """PARTIALs should be cumulative within the current window.

    I.e. for repeated emits (<window), start_s should stay constant and end_s
    should grow monotonically.
    """

    cfg = RealtimeConfig(
        sr=16000,
        window_sec=5.0,
        overlap_sec=0.5,
        partial_enable=True,
        partial_mode="cumulative",
        emit_every_sec=0.1,
        partial_stability_repeats=1,
        partial_min_buffer_sec=0.0,
        partial_min_transcribe_sec=0.0,
        finalize_min_dur_sec=0.1,
        key_resolution_sec=0.1,
    )

    def gate_fn(_w):
        return True

    def diarize_fn(_w):
        return []

    def asr_fn(chunk, lang, start_sample, partial):
        # Encode length into text so we can see changes.
        return _candidate(f"len={chunk.size}", chunk, lang, start_sample, partial)

    def map_speaker_fn(_tenant_id, diar_label, _chunk):
        return diar_label

    processor = RealtimeSessionProcessor(
        cfg=cfg,
        partial_gate_fn=gate_fn,
        window_gate_fn=gate_fn,
        diarize_fn=diarize_fn,
        asr_fn=asr_fn,
        map_speaker_fn=map_speaker_fn,
    )
    engine = RealtimeEngine(cfg=cfg, processor=processor)

    pcm_a = (np.ones(int(cfg.sr * 0.25), dtype=np.int16)).tobytes()
    pcm_b = (np.ones(int(cfg.sr * 0.25), dtype=np.int16)).tobytes()

    out1 = engine.feed("s", pcm_a, tenant_id="t1", lang="en")
    out2 = engine.feed("s", pcm_b, tenant_id="t1", lang="en")

    p1 = [r for r in out1 if not r.is_final]
    p2 = [r for r in out2 if not r.is_final]
    assert p1 and p2

    assert p2[-1].start_s == p1[-1].start_s
    assert p2[-1].end_s > p1[-1].end_s


def test_rtservice_skips_full_interval_partial_while_waiting_for_right_context():
    """A full cumulative PARTIAL must not block the contextual FINAL pass."""

    cfg = RealtimeConfig(
        sr=1000,
        window_sec=10.0,
        overlap_sec=1.0,
        partial_enable=True,
        partial_mode="cumulative",
        emit_every_sec=2.0,
        partial_stability_repeats=1,
        partial_min_buffer_sec=0.0,
        partial_min_transcribe_sec=1.5,
        finalize_min_dur_sec=0.1,
        key_resolution_sec=0.1,
    )
    asr_calls = []

    def gate_fn(_wave):
        return True

    def asr_fn(wave, lang, start_sample, partial):
        asr_calls.append((wave.size, partial))
        return _candidate("partial" if partial else "final", wave, lang, start_sample, partial)

    processor = RealtimeSessionProcessor(
        cfg=cfg,
        partial_gate_fn=gate_fn,
        window_gate_fn=gate_fn,
        diarize_fn=lambda _wave: [],
        asr_fn=asr_fn,
        map_speaker_fn=lambda _tenant_id, diar_label, _wave: diar_label,
    )
    engine = RealtimeEngine(cfg=cfg, processor=processor)

    first_eight_seconds = np.ones(8 * cfg.sr, dtype=np.int16).tobytes()
    next_two_seconds = np.ones(2 * cfg.sr, dtype=np.int16).tobytes()
    right_context = np.ones(cfg.sr, dtype=np.int16).tobytes()

    assert [result.text for result in engine.feed("s", first_eight_seconds, tenant_id="t1")] == ["partial"]
    assert engine.feed("s", next_two_seconds, tenant_id="t1") == []
    assert asr_calls == [(8000, True)]

    final_results = engine.feed("s", right_context, tenant_id="t1")
    assert [result.text for result in final_results] == ["final"]
    assert asr_calls == [(8000, True), (11000, False)]


def test_rtservice_slow_decode_does_not_reset_lag_as_client_idle(monkeypatch):
    """Internal inference time must keep PARTIAL overload suppression active."""

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(engine_module.time, "time", lambda: clock.now)
    asr_calls = []

    def asr_fn(wave, lang, start_sample, partial):
        asr_calls.append(partial)
        if not partial:
            clock.now += 8.0
        return _candidate("partial" if partial else "final", wave, lang, start_sample, partial)

    cfg, engine = _lag_guard_engine(asr_fn)

    full_window = np.ones(cfg.sr, dtype=np.int16).tobytes()
    assert [result.text for result in engine.feed("s", full_window, tenant_id="t1")] == ["final"]

    next_audio = np.ones(int(0.2 * cfg.sr), dtype=np.int16).tobytes()
    assert engine.feed("s", next_audio, tenant_id="t1") == []
    assert asr_calls == [False]


def test_rtservice_genuine_client_idle_still_resets_lag(monkeypatch):
    """Time after a completed feed remains a valid client-pause signal."""

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(engine_module.time, "time", lambda: clock.now)

    def asr_fn(wave, lang, start_sample, partial):
        return _candidate(f"samples={wave.size}", wave, lang, start_sample, partial)

    cfg, engine = _lag_guard_engine(asr_fn)
    audio = np.ones(int(0.2 * cfg.sr), dtype=np.int16).tobytes()

    assert [result.text for result in engine.feed("s", audio, tenant_id="t1")] == ["samples=200"]
    clock.now += 4.0
    assert [result.text for result in engine.feed("s", audio, tenant_id="t1")] == ["samples=400"]


def test_rtservice_engine_isolates_tenants_same_session_id():
    """Same session_id in different tenants must not share buffers."""

    cfg = RealtimeConfig(sr=16000, window_sec=1.0, overlap_sec=0.0, finalize_min_dur_sec=0.1, key_resolution_sec=0.1)

    def gate_fn(_w):
        return True

    def diarize_fn(_w):
        return [{"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]

    def asr_fn(chunk, lang, start_sample, partial):
        return _candidate("ok", chunk, lang, start_sample, partial)

    def map_speaker_fn(_tenant_id, diar_label, _chunk):
        return diar_label

    processor = RealtimeSessionProcessor(
        cfg=cfg,
        partial_gate_fn=gate_fn,
        window_gate_fn=gate_fn,
        diarize_fn=diarize_fn,
        asr_fn=asr_fn,
        map_speaker_fn=map_speaker_fn,
    )

    engine = RealtimeEngine(cfg=cfg, processor=processor)

    pcm16 = (np.ones(int(cfg.sr * cfg.window_sec), dtype=np.int16)).tobytes()
    out_t1 = engine.feed("same", pcm16, tenant_id="t1", lang="en")
    assert len(out_t1) == 1

    # other tenant only half window -> must not be influenced by t1 buffer
    pcm16_half = (np.ones(int(cfg.sr * cfg.window_sec / 2), dtype=np.int16)).tobytes()
    out_t2 = engine.feed("same", pcm16_half, tenant_id="t2", lang="en")
    assert out_t2 == []
