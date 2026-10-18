"""Run inside the Nemotron image with a GPU, model cache and synthetic WAV."""

import wave
from pathlib import Path

import pytest

from nemotron_rtservice.native import NativeNemotronBackend


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not Path("/opt/nemo-speech/models/silero-v6.2.0.gguf").exists(),
                       reason="requires the native image, GPU and model cache"),
]


@pytest.fixture
def speech():
    with wave.open(str(Path(__file__).parent / "data/test_cs.wav"), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        # Skip the fixture's opening non-speech audio.
        wav.setpos(3 * 16000)
        return wav.readframes(8 * 16000)


@pytest.mark.parametrize("diarization,soft_after,soft_silence", [
    (False, 90, 500), (True, 90, 500), (False, 1, 30000),
])
def test_native_vad_endpoints_before_eof_and_rearms(monkeypatch, diarization, soft_after, soft_silence, speech):
    monkeypatch.setenv("NEMOTRON_DIARIZATION", str(diarization).lower())
    monkeypatch.setenv("NEMOTRON_ENDPOINTING_SILENCE_MS", "2000")
    monkeypatch.setenv("NEMOTRON_ENDPOINTING_SOFT_AFTER_SECONDS", str(soft_after))
    monkeypatch.setenv("NEMOTRON_ENDPOINTING_SOFT_SILENCE_MS", str(soft_silence))
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


@pytest.mark.parametrize("soft_after,maximum", [(90, 120), (12, 20)])
def test_duration_policy_keeps_short_pauses_then_relaxes_and_rearms(monkeypatch, soft_after, maximum, speech):
    monkeypatch.setenv("NEMOTRON_DIARIZATION", "true")
    monkeypatch.setenv("NEMOTRON_ENDPOINTING_SOFT_AFTER_SECONDS", str(soft_after))
    monkeypatch.setenv("NEMOTRON_MAX_UTTERANCE_SECONDS", str(maximum))
    monkeypatch.delenv("NEMOTRON_ENDPOINTING_SOFT_SILENCE_MS", raising=False)
    backend = NativeNemotronBackend(1)
    try:
        session = backend.open("duration-pause", "cs-CZ", endpointing_silence_ms=3000)
        try:
            # A long initial silence must not spend the new utterance's budget.
            for _ in range((maximum + 1) * 10):
                assert not any(u.final for u in session.push(bytes(3200), 16000))
            # One-second pauses cannot end an utterance before the soft age.
            cycles = soft_after // 9 + 1
            for cycle in range(cycles):
                finals = []
                pcm = speech + bytes(32000)
                for offset in range(0, len(pcm), 640):
                    finals.extend(u for u in session.push(pcm[offset:offset + 640], 16000) if u.final)
                assert bool(finals) == (cycle == cycles - 1)
            assert len(finals) == 1 and finals[0].text and finals[0].words
            # A new utterance restores the normal three-second silence setting.
            for offset in range(0, len(pcm), 640):
                assert not any(u.final for u in session.push(pcm[offset:offset + 640], 16000))
            assert any(u.final and u.text for u in session.finish())
        finally:
            session.close()
        print(f"duration_pause_check=ok soft_after_s={soft_after} max_s={maximum}")
    finally:
        backend.close()


def test_emergency_endpoint_during_continuous_speech(monkeypatch, speech):
    monkeypatch.setenv("NEMOTRON_DIARIZATION", "false")
    monkeypatch.setenv("NEMOTRON_ENDPOINTING_SOFT_AFTER_SECONDS", "90")
    monkeypatch.setenv("NEMOTRON_MAX_UTTERANCE_SECONDS", "120")
    monkeypatch.setenv("NEMOTRON_ENDPOINTING_SOFT_SILENCE_MS", "30000")
    backend = NativeNemotronBackend(1)
    session = backend.open("duration-emergency", "cs-CZ", endpointing_silence_ms=30000)
    try:
        finals = []
        pcm = speech * 16
        for offset in range(0, len(pcm), 640):
            for update in session.push(pcm[offset:offset + 640], 16000):
                if update.final:
                    finals.append((offset / 32000 + 0.02, update))
        assert len(finals) == 1 and finals[0][1].text
        assert 120 <= finals[0][0] <= 121
        assert any(u.final and u.text for u in session.finish())
        print(f"emergency_check=ok final_at_s={finals[0][0]:.2f}")
    finally:
        session.close()
        backend.close()
