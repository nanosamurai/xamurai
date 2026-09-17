"""Run inside the Nemotron image with a GPU, model cache and synthetic WAV."""

import wave
from pathlib import Path

import numpy as np
import pytest

from nemotron_rtservice.native import NativeNemotronBackend


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not Path("/opt/nemo-speech/models/silero-v6.2.0.gguf").exists(),
                       reason="requires the native image, GPU and model cache"),
]


@pytest.mark.parametrize("diarization", [False, True])
def test_native_vad_endpoints_before_eof_and_rearms(monkeypatch, diarization):
    monkeypatch.setenv("NEMOTRON_DIARIZATION", str(diarization).lower())
    monkeypatch.setenv("NEMOTRON_ENDPOINTING_SILENCE_MS", "2000")
    with wave.open(str(Path(__file__).parent / "data/test_cs.wav"), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    # Trim leading silence and use a short excerpt, keeping the first endpoint
    # well below the independent 30-second hard limit even if VAD fails.
    start = np.flatnonzero(np.abs(samples.astype(np.int32)) > 200)[0]
    speech = samples[start:start + 8 * 16000].tobytes()
    silence = bytes(6 * 32000)
    backend = NativeNemotronBackend(maximum_sessions=1)
    delays = {}
    try:
        for silence_ms in (800, 0, 3000):  # zero inherits the 2000 ms default
            session = backend.open(f"vad-{silence_ms}", "cs-CZ", endpointing_silence_ms=silence_ms)
            try:
                for offset in range(0, len(silence), 640):
                    assert not any(u.final and u.text for u in session.push(silence[offset:offset + 640], 16000))
                for utterance in range(2):
                    for offset in range(0, len(speech), 640):
                        session.push(speech[offset:offset + 640], 16000)
                    finals = []
                    for offset in range(0, len(silence), 640):
                        for update in session.push(silence[offset:offset + 640], 16000):
                            if update.final and update.text:
                                finals.append((offset / 32000 + 0.02, update))
                    assert len(finals) == 1, "expected one endpoint during the pause, before EOF"
                    delay, final = finals[0]
                    expected = (silence_ms or 2000) / 1000
                    assert expected - 0.5 <= delay <= expected + 1.0
                    if diarization:
                        assert final.words and any(word.speaker >= 0 for word in final.words)
                    if utterance == 0:
                        delays[silence_ms] = delay
                    print(f"vad_check=ok diarization={diarization} silence_ms={silence_ms or 2000} "
                          f"utterance={utterance + 1} pause_to_final_s={delay:.1f}")
                # All meaningful finals must already have arrived via push().
                assert not any(u.final and u.text for u in session.finish())
            finally:
                session.close()
        assert delays[800] < delays[0] < delays[3000]
    finally:
        backend.close()
