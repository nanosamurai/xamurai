"""Opt-in GPU parity check using one model for native and shared ASR calls."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
import soundfile as sf


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("WHISPERX_SHARED_GPU_TEST") != "1", reason="opt-in GPU qualification")
def test_shared_asr_matches_native_chunks_and_preserves_request_results():
    from whisperx_worker import pipeline
    from whisperx_worker.batch_backend import WhisperXBackend
    from whisperx_worker.batching import ModelOwner
    pipeline._init_whisperx()
    model = pipeline._WHISPERX_MODEL
    audio, sr = sf.read(Path(__file__).parent / "data" / "test_en.wav", dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    inputs = [audio, np.tile(audio, 3)]
    native = [model.transcribe(a, batch_size=16, language="en") for a in inputs]
    owner = ModelOwner(WhisperXBackend(model), wait_ms=200)
    barrier = Barrier(2)
    def run(index):
        barrier.wait(timeout=10)
        return owner.transcribe(inputs[index], stage=("refinement", "finalization")[index], language="en")
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(run, i) for i in range(2)]
            shared = [f.result(120) for f in futures]
        for expected, actual in zip(native, shared):
            assert expected["language"] == actual["language"]
            assert [(s["text"], s["start"], s["end"]) for s in expected["segments"]] == [
                (s["text"], s["start"], s["end"]) for s in actual["segments"]]
        assert pipeline._WHISPERX_MODEL is model
        assert owner.mixed_batch_count >= 1
    finally:
        owner.close()
        owner.thread.join(10)
