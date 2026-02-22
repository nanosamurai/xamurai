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
        gate_fn=gate_fn,
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
        gate_fn=gate_fn,
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
