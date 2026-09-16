"""Real pinned native models; run inside the Parakeet finalizer image with a GPU."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not Path("/opt/nemo-speech/lib/libnemo_speech_asr_c.so").exists(),
    reason="requires the Parakeet native image",
)]


@pytest.fixture(scope="module")
def pipeline():
    from finalizer_worker.parakeet import Parakeet

    model = Parakeet()
    yield model
    model.close()


def test_real_speech_has_timed_speakers_and_fresh_request_state(pipeline, tmp_path):
    audio, sr = sf.read(Path(__file__).parent / "data/test_cs.wav", dtype="float32")
    path = tmp_path / "speech.wav"
    sf.write(path, audio[:12 * sr], sr, subtype="PCM_16")
    first = pipeline(path, tenant="first", lang="cs")
    text, segments = first
    assert text.strip() and segments
    assert any(segment["speaker"].startswith("SPEAKER_") for segment in segments)
    words = [word for segment in segments for word in segment["words"]]
    assert words and all(0 <= w["start_s"] <= w["end_s"] <= 12 for w in words)
    assert [w["start_s"] for w in words] == sorted(w["start_s"] for w in words)
    assert "".join(text.split()) == "".join(w["text"].replace(" ", "") for w in words)
    # Reusing the models must not reuse the preceding recording's speaker/session state.
    assert pipeline(path, tenant="second", lang="en") == first


def test_silence_empty_and_invalid_audio(pipeline, tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    assert pipeline(path) == ("", [])
    sf.write(path, np.zeros(0, dtype=np.float32), 16000, subtype="PCM_16")
    assert pipeline(path) == ("", [])
    sf.write(path, np.zeros((16000, 2), dtype=np.float32), 16000, subtype="PCM_16")
    with pytest.raises(ValueError, match="mono 16 kHz"):
        pipeline(path)
