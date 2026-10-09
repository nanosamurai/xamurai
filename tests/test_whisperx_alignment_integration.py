"""Opt-in real GPU parity for concurrent English and Czech alignment."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
import soundfile as sf


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("WHISPERX_SHARED_GPU_TEST") != "1", reason="opt-in GPU qualification")
def test_parallel_alignment_matches_sequential_results_with_resident_models():
    from whisperx_worker import pipeline
    from whisperx_worker.alignment import AlignmentExecutor, WhisperXAlignmentBackend

    pipeline._init_whisperx()
    backend = WhisperXAlignmentBackend(("en", "de", "cs"), pipeline._WHISPERX_DEVICE)
    identities = {language: id(model) for language, (model, _) in backend.models.items()}
    inputs = []
    expected = []
    for language in ("en", "cs"):
        audio, sr = sf.read(Path(__file__).parent / "data" / f"test_{language}.wav", dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        audio = np.tile(audio, 3)
        segments = pipeline._WHISPERX_MODEL.transcribe(audio, batch_size=16, language=language)["segments"]
        assert segments
        inputs.append((segments, audio, language))
        expected.append(backend.align(deepcopy(segments), audio, language))
    executor = AlignmentExecutor(backend, concurrency=2)
    start = Barrier(2)
    def run(args):
        start.wait(10)
        return executor.align(*args)
    try:
        with ThreadPoolExecutor(2) as callers:
            results = [callers.submit(run, args) for args in inputs]
            actual = [result.result(120) for result in results]
        for reference, result in zip(expected, actual):
            # Compare the public text/timing result, including word association.
            assert result["segments"] == reference["segments"]
            assert any(s.get("words") for s in result["segments"])
        assert executor.snapshot()["alignment_peak_active"] == 2
        assert {language: id(model) for language, (model, _) in backend.models.items()} == identities
    finally:
        executor.close()
        for thread in executor.threads:
            thread.join(10)
