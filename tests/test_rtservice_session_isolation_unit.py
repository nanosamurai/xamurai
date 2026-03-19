import numpy as np

from rtservice.engine import (
    RealtimeConfig,
    RealtimeEngine,
    RealtimeSessionProcessor,
)


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

    def asr_fn(_chunk, _lang):
        return "hello"

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

    def asr_fn(_chunk, _lang):
        return "hello"

    def map_speaker_fn(_tenant_id, diar_label, _chunk):
        return diar_label

    class _FakeBundle:
        def gate_window(self, w):
            return gate_fn(w)

        def gate_speech(self, w):
            return gate_fn(w)

        def diarize_window(self, w):
            return diarize_fn(w)

        def asr_text(self, w, lang):
            return asr_fn(w, lang)

        def map_speaker_label(self, tenant_id, diar_label, chunk):
            return map_speaker_fn(tenant_id, diar_label, chunk)

    # IMPORTANT: we do NOT inject a fixed processor here, because per-session
    # overrides are implemented by selecting a processor per effective cfg.
    engine = RealtimeEngine(cfg=cfg, model_bundle=_FakeBundle())

    # Feed 0.3s worth of audio; should trigger partial for emit_every_sec=0.2.
    pcm16_03 = (np.ones(int(cfg.sr * 0.3), dtype=np.int16)).tobytes()

    out_a = engine.feed("sA", pcm16_03, tenant_id="t1", lang="en", rt_emit_every_sec=0.2)
    out_b = engine.feed("sB", pcm16_03, tenant_id="t1", lang="en", rt_emit_every_sec=0.6)

    assert any((not r.is_final) for r in out_a), "Expected PARTIAL for session A"
    assert not any((not r.is_final) for r in out_b), "Did not expect PARTIAL for session B"


def test_rtservice_engine_isolates_tenants_same_session_id():
    """Same session_id in different tenants must not share buffers."""

    cfg = RealtimeConfig(sr=16000, window_sec=1.0, overlap_sec=0.0, finalize_min_dur_sec=0.1, key_resolution_sec=0.1)

    def gate_fn(_w):
        return True

    def diarize_fn(_w):
        return [{"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]

    def asr_fn(_chunk, _lang):
        return "ok"

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
