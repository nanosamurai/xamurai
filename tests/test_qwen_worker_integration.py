"""Real vLLM/pyannote integration; run in the Qwen worker image with a GPU."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    importlib.util.find_spec("qwen_asr") is None, reason="requires the Qwen worker image",
)]


@pytest.fixture(scope="module")
def pipeline():
    from qwen_worker.pipeline import Qwen
    return Qwen()


def test_real_batched_speech_and_independent_recordings(pipeline, tmp_path, monkeypatch):
    samples, sr = sf.read(Path(__file__).parent / "data/test_cs.wav", dtype="float32")
    path = tmp_path / "speech.wav"
    samples = np.tile(samples, 3)[:35 * sr]
    sf.write(path, samples, sr, subtype="PCM_16")
    batches = []
    generate = pipeline.model.model.generate

    def observe(requests, *args, **kwargs):
        batches.append(len(requests))
        return generate(requests, *args, **kwargs)

    monkeypatch.setattr(pipeline.model.model, "generate", observe)
    first = pipeline(path, tenant="first", lang="cs")
    text, segments = first
    assert text.strip() and segments
    assert any(1 < size <= pipeline.batch_size for size in batches)
    assert all(0 <= s["start_s"] < s["end_s"] <= 35 and s["speaker"] for s in segments)
    assert all(s["end_s"] - s["start_s"] <= 30 for s in segments)
    assert all(a["end_s"] <= b["start_s"] for a, b in zip(segments, segments[1:]))
    assert text == " ".join(s["text"] for s in segments)
    # GPU reductions need not reproduce identical text. A shorter independent
    # request must restart timing and cannot inherit the previous recording.
    sf.write(path, samples[:10 * sr], sr, subtype="PCM_16")
    second_text, second_segments = pipeline(path, tenant="second", lang="cs")
    assert second_text.strip() and second_segments
    assert all(0 <= s["start_s"] < s["end_s"] <= 10 and s["speaker"] for s in second_segments)


def test_empty_silence_and_invalid_audio(pipeline, tmp_path):
    path = tmp_path / "audio.wav"
    for size in (0, 16000):
        sf.write(path, np.zeros(size), 16000, subtype="PCM_16")
        assert pipeline(path) == ("", [])
    for data, sr in [(np.zeros((16000, 2)), 16000), (np.zeros(8000), 8000)]:
        sf.write(path, data, sr)
        with pytest.raises(ValueError, match="mono 16 kHz"):
            pipeline(path)
    sf.write(path, np.full(16000, np.nan), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="nonfinite"):
        pipeline(path)
