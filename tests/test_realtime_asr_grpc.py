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
_HAS_BOTO3 = importlib.util.find_spec("boto3") is not None

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
    tenant_id: str = "t-test",
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
            tenant_id=tenant_id,
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
        tenant_id=tenant_id,
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


@pytest.mark.integration
def test_realtime_asr_stream_emits_partial_and_final(grpc_channel):
    """Plan C regression test: rtservice should emit PARTIAL updates before FINAL.

    We don't assert exact timings (wall clock) because inference cost varies.
    Instead, we send enough audio for at least one partial emission and at
    least one finalized window.
    """

    stub = stream_pb2_grpc.RealtimeASRStub(grpc_channel)

    wav_path = Path(__file__).parent / "data" / "test_cs.wav"
    audio = load_audio_mono_16k(wav_path)

    # Send only a short prefix to keep the test runtime bounded.
    # Default rtservice window is 5s, so 6s is enough to finalize at least one window.
    audio = audio[: int(6.0 * SR)]

    events: List[stream_pb2.AsrEvent] = list(stub.Stream(gen_chunks("test-grpc-partial", audio, lang="cs")))
    assert events, "Expected some AsrEvents"

    partial_idx = [i for i, e in enumerate(events) if e.type == stream_pb2.PARTIAL and e.text.strip()]
    final_idx = [i for i, e in enumerate(events) if e.type == stream_pb2.FINAL and e.text.strip()]

    assert partial_idx, "Expected at least one PARTIAL AsrEvent"
    assert final_idx, "Expected at least one FINAL AsrEvent"
    assert min(partial_idx) < min(final_idx), "Expected PARTIAL to arrive before FINAL"


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_HF_TOKEN, reason="requires HF_TOKEN")
@pytest.mark.skipif(not _HAS_BOTO3, reason="boto3 not installed")
def test_realtime_asr_stream_s3_enrollment(localstack_s3, tmp_path):
    """Integration test: rtservice loads enrolled speakers from S3 (LocalStack).

    This ensures multi-tenant enrollment works without relying on local filesystem.

    Requires Docker (LocalStack). Uses in-process gRPC server.
    """

    # Upload tenant enrollment to LocalStack
    from tests._s3_enrollment_testdata import (
        make_enrollment_wav_from_test_audio,
        upload_tenant_enrollment_to_s3,
    )

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=localstack_s3["endpoint_url"],
        region_name=localstack_s3["region"],
        aws_access_key_id=localstack_s3["access_key"],
        aws_secret_access_key=localstack_s3["secret_key"],
    )

    tenant = "t-test"
    test_wav_path = Path(__file__).parent / "data" / "test_cs.wav"

    # Create enrollment wav from the same test audio
    enroll_wav = tmp_path / "enroll" / "Miro-cz.wav"
    make_enrollment_wav_from_test_audio(src_wav_path=test_wav_path, out_wav_path=enroll_wav, seconds=5.0)

    upload_tenant_enrollment_to_s3(
        s3_client=s3,
        bucket=localstack_s3["bucket"],
        prefix=localstack_s3["prefix"],
        tenant_id=tenant,
        speaker_id="spk-1",
        label="Miro-cz",
        sample_wav_path=enroll_wav,
    )

    # Configure rtservice enrollment backend via env (engine reads env at init)
    os.environ["ENROLL_BACKEND"] = "s3_manifest"
    os.environ["ENROLL_S3_BUCKET"] = localstack_s3["bucket"]
    os.environ["ENROLL_S3_PREFIX"] = localstack_s3["prefix"]
    os.environ["ENROLL_S3_ENDPOINT"] = localstack_s3["endpoint_url"]
    os.environ["ENROLL_S3_REGION"] = localstack_s3["region"]
    os.environ["ENROLL_S3_ACCESS_KEY"] = localstack_s3["access_key"]
    os.environ["ENROLL_S3_SECRET_KEY"] = localstack_s3["secret_key"]
    os.environ["ENROLL_S3_FORCE_PATH_STYLE"] = "true"

    # diarization model compatible with pyannote 3.x
    os.environ["RT_DIAR_MODEL"] = "pyannote/speaker-diarization-3.1"

    # make mapping permissive
    os.environ["ENROLL_SIM_THRESHOLD"] = "-1.0"

    # Start gRPC server with a freshly initialized engine
    server = create_realtime_asr_server(port=50053)
    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    time.sleep(1.0)

    try:
        with grpc.insecure_channel("localhost:50053") as channel:
            stub = stream_pb2_grpc.RealtimeASRStub(channel)

            audio = load_audio_mono_16k(test_wav_path)
            events: List[stream_pb2.AsrEvent] = list(stub.Stream(gen_chunks("test-grpc-s3", audio, tenant_id=tenant)))

        assert events, "Expected some AsrEvents"
        speakers = {e.speaker for e in events if e.speaker}
        assert "Miro-cz" in speakers, f"Expected Miro-cz in speakers, got {sorted(speakers)}"
    finally:
        server.stop(grace=5.0)
        t.join(timeout=5.0)
