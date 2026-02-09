import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterable, List

import importlib.util

import grpc
import numpy as np
import pytest
import soundfile as sf

from proto_gen import stream_pb2
from proto_gen import stream_pb2_grpc

from rtservice.server import create_realtime_asr_server

# This module requires heavy deps (torch/pyannote/faster-whisper) and HF_TOKEN.
_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_PYANNOTE = importlib.util.find_spec("pyannote") is not None
_HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None
_HAS_HF_TOKEN = bool(os.getenv("HF_TOKEN"))

pytestmark = pytest.mark.skipif(
    not (_HAS_TORCH and _HAS_PYANNOTE and _HAS_FASTER_WHISPER and _HAS_HF_TOKEN),
    reason="rtservice integration test requires torch + pyannote + faster-whisper + HF_TOKEN",
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SR = 16000
CHUNK_SAMPLES = 2048


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_audio_mono_16k(path: str | Path) -> np.ndarray:
    """Load a WAV as mono 16k float32."""
    path = Path(path)
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        # simple resample for tests
        import librosa  # only for tests; minor dependency
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x.astype("float32")


def gen_chunks(
    session_id: str,
    audio: np.ndarray,
    lang: str = "cs",
    chunk_samples: int = CHUNK_SAMPLES,
) -> Iterable[stream_pb2.AudioChunk]:
    """
    Turn a mono 16k float32 waveform into a stream of AudioChunk messages.
    """
    pcm16 = (audio * 32768.0).clip(-32768, 32767).astype("<i2")
    n = pcm16.shape[0]
    seq = 0
    for start in range(0, n, chunk_samples):
        end = min(start + chunk_samples, n)
        frame = pcm16[start:end]
        seq += 1
        msg = stream_pb2.AudioChunk(
            session_id=session_id,
            seq=seq,
            t0_ns=0,
            sample_rate=SR,
            pcm16_le=frame.tobytes(),
            lang=lang or "",
        )
        logger.debug(
            "[test] sending chunk seq=%d samples=%d (%.3fs-%.3fs)",
            seq,
            frame.size,
            start / SR,
            end / SR,
        )
        yield msg

    yield stream_pb2.AudioChunk(
        session_id=session_id,
        seq=seq + 1,
        t0_ns=0,
        sample_rate=SR,
        pcm16_le=b"",
        lang=lang or "",
    )
    logger.debug("[test] sent final chunk seq=%d", seq + 1)


# ---------------------------------------------------------------------------
# Fixtures: start/stop server for tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grpc_server():
    """
    Start the RealtimeASR gRPC server in a background thread, yield its port,
    and stop it after the tests in this module finish.
    """
    server = create_realtime_asr_server(port=50052)
    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    logger.info("[test] RealtimeASR server started at localhost:50052")
    # tiny sleep to let server bind the port
    time.sleep(1.0)
    try:
        yield server
    finally:
        logger.info("[test] RealtimeASR server stopping")
        server.stop(grace=5.0)
        t.join(timeout=5.0)
        logger.info("[test] RealtimeASR server stopped")


@pytest.fixture(scope="module")
def grpc_channel(grpc_server):
    """Create a gRPC channel connected to the test server."""
    with grpc.insecure_channel("localhost:50052") as channel:
        yield channel


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_realtime_asr_stream_basic(grpc_channel):
    """
    Sanity check: send a full test WAV and ensure at least one AsrEvent with text.
    """
    stub = stream_pb2_grpc.RealtimeASRStub(grpc_channel)

    wav_path = Path(__file__).parent / "data" / "test_cs.wav"
    audio = load_audio_mono_16k(wav_path)

    chunks_iter = gen_chunks("test-grpc-basic", audio, lang="cs")

    def wrapped_gen():
        try:
            for msg in gen_chunks("test-grpc-basic", audio):
                print("[debug] sending chunk")
                yield msg
        except Exception as e:
            print("[debug] gen_chunks raised:", repr(e))
            raise

    events: List[stream_pb2.AsrEvent] = list(stub.Stream(wrapped_gen()))

    for ev in events:
        logger.debug(
            "[test-basic] got AsrEvent session=%s [%.2f, %.2f] speaker=%s text=%r",
            ev.session_id,
            ev.start_s,
            ev.end_s,
            ev.speaker,
            ev.text,
        )

    assert len(events) > 0, "Expected at least one AsrEvent from basic stream"
    total_chars = sum(len(ev.text) for ev in events)
    assert total_chars > 0, f"Expected non-empty text, got total_chars={total_chars}"


@pytest.mark.integration
def test_realtime_asr_stream(grpc_channel):
    """
    More 'streamy' test: same as basic, but log everything explicitly and give
    clearer failure if we somehow get an empty transcript.
    """
    stub = stream_pb2_grpc.RealtimeASRStub(grpc_channel)

    wav_path = Path(__file__).parent / "data" / "test_cs.wav"
    audio = load_audio_mono_16k(wav_path)

    chunks_iter = gen_chunks("test-grpc-stream", audio, lang="cs")

    events: List[stream_pb2.AsrEvent] = []
    for ev in stub.Stream(chunks_iter):
        logger.debug(
            "[test-stream] got AsrEvent session=%s [%.2f, %.2f] speaker=%s text=%r",
            ev.session_id,
            ev.start_s,
            ev.end_s,
            ev.speaker,
            ev.text,
        )
        events.append(ev)

    assert events, "Expected some AsrEvents, got none"
    total_chars = sum(len(ev.text) for ev in events)
    logger.info("[test-stream] total AsrEvents=%d total_chars=%d", len(events), total_chars)
    assert total_chars > 0, f"Expected non-empty text, got total_chars={total_chars}"
