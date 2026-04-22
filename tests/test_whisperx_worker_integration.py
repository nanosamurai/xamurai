# tests/test_whisperx_worker_integration.py

import os
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
from confluent_kafka import Producer, Consumer

from tests.conftest import _ensure_topics

import importlib.util

from proto_gen import stream_pb2

_HAS_WHISPERX = importlib.util.find_spec("whisperx") is not None
_HAS_HF_TOKEN = bool(os.getenv("HF_TOKEN"))
_HAS_BOTO3 = importlib.util.find_spec("boto3") is not None

SR = 16000  # must match whisperx_worker.SR


def _make_producer(bootstrap: str) -> Producer:
    return Producer(
        {
            "bootstrap.servers": bootstrap,
            "client.id": "test-whisperx-producer",
            "compression.type": "zstd",
            "linger.ms": 5,
            "batch.size": 128_000,
        }
    )


def _make_consumer(bootstrap: str, group_id: str) -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )

@pytest.mark.timeout(900)           # WhisperX model load can be sloooow on first run
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_WHISPERX, reason="whisperx not installed in this env")
def test_whisperx_worker_end_to_end_real(kafka_bootstrap):
    """
    True integration test:

    - Uses a *real* Kafka broker (kafka_bootstrap fixture points to localhost:9092).
    - Starts the real whisperx_worker.main() in a background thread.
    - Sends real audio (test WAV) to a test audio topic.
    - Waits for a RefinedEvent on a test refined topic.

    This is more of a smoke/E2E test and may be slow, especially on first model download.
    """

    # ------------------------------------------------------------------ #
    # 1) Configure env *before* importing the worker
    # ------------------------------------------------------------------ #
    topic_audio = "audio.raw.test"
    topic_refined = "transcripts.refined.test"

    # Ensure topics exist (Kafka in tests may have auto-create disabled)
    _ensure_topics(kafka_bootstrap, [topic_audio, topic_refined])

    os.environ["KAFKA_BOOTSTRAP"] = kafka_bootstrap
    os.environ["KAFKA_TOPIC_AUDIO"] = topic_audio
    os.environ["KAFKA_TOPIC_REFINED"] = topic_refined
    # Use shorter slices than 60s so test doesn't need a 1-minute clip
    os.environ["WHISPERX_SLICE_SECONDS"] = "10.0"

    # Import AFTER env is set so module-level constants pick them up
    import importlib
    from whisperx_worker import whisperx_worker  # adjust if package name differs

    importlib.reload(whisperx_worker)

    # Sanity: log which topics the worker thinks it uses
    print(
        f"[test] worker topics: audio={whisperx_worker.TOPIC_AUDIO}, "
        f"refined={whisperx_worker.TOPIC_REFINED}, bootstrap={whisperx_worker.KAFKA_BOOTSTRAP}"
    )

    # ------------------------------------------------------------------ #
    # 2) Start the worker in a background thread
    # ------------------------------------------------------------------ #
    worker_thread = threading.Thread(
        target=whisperx_worker.main,
        name="whisperx-worker-main",
        daemon=True,
    )
    worker_thread.start()

    # Give the worker time to init WhisperX + subscribe to Kafka.
    # First run may include downloading models, so this might need to be generous.
    print("[test] waiting 30s for whisperx_worker to start up…")
    time.sleep(30.0)

    # ------------------------------------------------------------------ #
    # 3) Produce real audio to TOPIC_AUDIO
    # ------------------------------------------------------------------ #
    session_id = "integration-whisperx-real-1"

    # Use a real test WAV that has speech, so WhisperX is likely to produce text.
    test_wav_path = Path(__file__).parent / "data" / "test_cs.wav"
    assert test_wav_path.exists(), f"Missing test WAV at {test_wav_path}"

    # Load as 16k mono (or let your own helper handle this if you have one)
    import soundfile as sf

    wav_data, wav_sr = sf.read(test_wav_path)
    if wav_data.ndim > 1:
        wav_data = wav_data[:, 0]  # take first channel

    if wav_sr != SR:
        # naive resample: for tests it's fine, or you can use librosa/scipy if available
        ratio = SR / float(wav_sr)
        new_len = int(len(wav_data) * ratio)
        x_old = np.linspace(0, 1, len(wav_data), endpoint=False)
        x_new = np.linspace(0, 1, new_len, endpoint=False)
        wav_data = np.interp(x_new, x_old, wav_data)

    wav_data = wav_data.astype("float32")
    # Ensure we have at least slice length + a bit
    min_seconds = 11.0
    if len(wav_data) < min_seconds * SR:
        reps = int(np.ceil((min_seconds * SR) / len(wav_data)))
        wav_data = np.tile(wav_data, reps)

    wav_data = wav_data[: int(min_seconds * SR)]
    # Convert float32 [-1,1] to PCM16 little endian
    wav_data = np.clip(wav_data, -1.0, 1.0)
    pcm16 = (wav_data * 32767.0).astype("<i2")

    audio_chunk = stream_pb2.AudioChunk(
        session_id=session_id,
        t0_ns=0,
        pcm16_le=pcm16.tobytes(),
        sample_rate=SR,
        lang="cs",
    )

    producer = _make_producer(kafka_bootstrap)
    producer.produce(
        topic=topic_audio,
        key=session_id.encode("utf-8"),
        value=audio_chunk.SerializeToString(),
    )
    producer.flush(30_000)
    print(f"[test] produced AudioChunk to {topic_audio} for session {session_id}")

    # ------------------------------------------------------------------ #
    # 4) Consume from TOPIC_REFINED and assert we get at least one event
    # ------------------------------------------------------------------ #
    consumer = _make_consumer(kafka_bootstrap, group_id="test-whisperx-consumer-real")
    consumer.subscribe([topic_refined])

    received_events: List[stream_pb2.RefinedEvent] = []
    deadline = time.time() + 180.0  # generous, first WhisperX run may be slow

    try:
        while time.time() < deadline:
            msg = consumer.poll(5.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[test] Kafka consumer error: {msg.error()}")
                continue

            ev = stream_pb2.RefinedEvent()
            ev.ParseFromString(msg.value())
            print(
                f"[test] got RefinedEvent: session={ev.session_id}, "
                f"{ev.start_s:.2f}-{ev.end_s:.2f}s, "
                f"segments={len(ev.segments)}, full_text_len={len(ev.full_text)}"
            )
            received_events.append(ev)
            # For this test, one event is enough
            break
    finally:
        consumer.close()

    assert received_events, "Did not receive any RefinedEvent from whisperx_worker"

    # Optional extra sanity checks:
    non_empty = [e for e in received_events if (e.full_text.strip() or e.text.strip())]
    assert non_empty, "Received RefinedEvent(s) but all had empty text"


@pytest.mark.timeout(1200)
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_WHISPERX, reason="whisperx not installed in this env")
@pytest.mark.skipif(not _HAS_HF_TOKEN, reason="HF_TOKEN not provided; diarization is disabled")
def test_whisperx_worker_diarization_and_enrollment_on_test_wav(kafka_bootstrap, tmp_path, monkeypatch):
    """Stricter E2E test:

    - runs WhisperX worker against real Kafka
    - enables diarization (requires HF_TOKEN)
    - enrolls ONLY one speaker (Miro-cz.wav) to make mapping deterministic
    - asserts at least one emitted RefinedEvent contains a segment with speaker == "Miro-cz"

    This ensures diarization+enrollment mapping actually runs on the provided sample.
    """

    topic_audio = "audio.raw.test.diar"
    topic_refined = "transcripts.refined.test.diar"
    _ensure_topics(kafka_bootstrap, [topic_audio, topic_refined])

    # Build a valid enrollment WAV on the fly to keep the test self-contained.
    # The repo's enrolled_speakers/ currently contains placeholder 0-byte wavs.
    enroll_dir = tmp_path / "enrolled_speakers"
    enroll_dir.mkdir(parents=True, exist_ok=True)

    # Use the first few seconds of the test wav as the enrollment sample.
    # This makes it very likely that at least one diarization cluster maps to this embedding.
    test_wav_path = Path(__file__).parent / "data" / "test_cs.wav"
    assert test_wav_path.exists(), f"Missing test WAV at {test_wav_path}"

    import soundfile as sf

    x, sr = sf.read(test_wav_path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]

    # save ~5 seconds as enrollment sample
    enroll_seconds = 5.0
    n = int(enroll_seconds * SR)

    # resample if needed (match worker SR)
    if sr != SR:
        ratio = SR / float(sr)
        new_len = int(len(x) * ratio)
        x_old = np.linspace(0, 1, len(x), endpoint=False)
        x_new = np.linspace(0, 1, new_len, endpoint=False)
        x = np.interp(x_new, x_old, x)

    x = x[:n]
    sf.write(enroll_dir / "Miro-cz.wav", x, SR, subtype="PCM_16")

    os.environ["KAFKA_BOOTSTRAP"] = kafka_bootstrap
    os.environ["KAFKA_TOPIC_AUDIO"] = topic_audio
    os.environ["KAFKA_TOPIC_REFINED"] = topic_refined
    os.environ["WHISPERX_SLICE_SECONDS"] = "10.0"
    os.environ["WHISPERX_ENABLE_DIARIZATION"] = "true"
    os.environ["WHISPERX_DIAR_MODEL"] = "pyannote/speaker-diarization-3.1"

    # Use legacy enrollment dir for deterministic local mapping.
    os.environ["ENROLL_BACKEND"] = "legacy_dir"
    os.environ["ENROLL_DIR"] = str(enroll_dir)

    # reduce flakiness: allow any similarity to map (we assert the label itself)
    os.environ["ENROLL_SIM_THRESHOLD"] = "-1.0"

    # make sure worker re-reads env
    import importlib
    from whisperx_worker import whisperx_worker

    importlib.reload(whisperx_worker)

    worker_thread = threading.Thread(
        target=whisperx_worker.main,
        name="whisperx-worker-main-diar",
        daemon=True,
    )
    worker_thread.start()

    print("[test] waiting 30s for whisperx_worker (diar) to start up…")
    time.sleep(30.0)

    session_id = "integration-whisperx-diar-1"

    # Reuse the test wav as the produced audio
    wav_data, wav_sr = sf.read(test_wav_path)
    if wav_data.ndim > 1:
        wav_data = wav_data[:, 0]

    if wav_sr != SR:
        ratio = SR / float(wav_sr)
        new_len = int(len(wav_data) * ratio)
        x_old = np.linspace(0, 1, len(wav_data), endpoint=False)
        x_new = np.linspace(0, 1, new_len, endpoint=False)
        wav_data = np.interp(x_new, x_old, wav_data)

    wav_data = wav_data.astype("float32")

    min_seconds = 11.0
    if len(wav_data) < min_seconds * SR:
        reps = int(np.ceil((min_seconds * SR) / len(wav_data)))
        wav_data = np.tile(wav_data, reps)

    wav_data = wav_data[: int(min_seconds * SR)]
    wav_data = np.clip(wav_data, -1.0, 1.0)
    pcm16 = (wav_data * 32767.0).astype("<i2")

    audio_chunk = stream_pb2.AudioChunk(
        session_id=session_id,
        t0_ns=0,
        pcm16_le=pcm16.tobytes(),
        sample_rate=SR,
        lang="cs",
        tenant_id="default",
    )

    producer = _make_producer(kafka_bootstrap)
    producer.produce(
        topic=topic_audio,
        key=session_id.encode("utf-8"),
        value=audio_chunk.SerializeToString(),
    )
    producer.flush(30_000)

    consumer = _make_consumer(kafka_bootstrap, group_id="test-whisperx-consumer-real-diar")
    consumer.subscribe([topic_refined])

    deadline = time.time() + 240.0
    seen_speakers = []

    try:
        while time.time() < deadline:
            msg = consumer.poll(5.0)
            if msg is None:
                continue
            if msg.error():
                continue

            ev = stream_pb2.RefinedEvent()
            ev.ParseFromString(msg.value())
            for seg in ev.segments:
                if seg.speaker:
                    seen_speakers.append(seg.speaker)
            print(
                f"[test] refined segments={len(ev.segments)} speakers={sorted(set(seen_speakers))[:5]} "
                f"full_text_len={len(ev.full_text)}"
            )

            if any((seg.speaker == "Miro-cz") for seg in ev.segments):
                return
    finally:
        consumer.close()

    raise AssertionError(
        f"Did not see expected speaker label 'Miro-cz'. Seen speakers: {sorted(set(seen_speakers))}"
    )


@pytest.mark.timeout(1500)
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_WHISPERX, reason="whisperx not installed in this env")
@pytest.mark.skipif(not _HAS_HF_TOKEN, reason="HF_TOKEN not provided; diarization is disabled")
@pytest.mark.skipif(not _HAS_BOTO3, reason="boto3 not installed")
def test_whisperx_worker_diarization_and_s3_enrollment_on_test_wav(
    kafka_bootstrap, tmp_path, localstack_s3
):
    """Strict E2E test with S3-backed enrollment (LocalStack).

    This validates:
    - diarization actually runs
    - enrollment cache loads from S3 manifest backend
    - emitted RefinedEvent contains a segment whose speaker is the enrolled human label

    Requires Docker (LocalStack + Kafka).
    """

    from tests._s3_enrollment_testdata import (
        make_enrollment_wav_from_test_audio,
        upload_tenant_enrollment_to_s3,
    )

    # Kafka topics
    topic_audio = "audio.raw.test.diar.s3"
    topic_refined = "transcripts.refined.test.diar.s3"
    _ensure_topics(kafka_bootstrap, [topic_audio, topic_refined])

    # Build enrollment wav + upload to LocalStack S3
    test_wav_path = Path(__file__).parent / "data" / "test_cs.wav"
    enroll_wav = tmp_path / "enroll" / "Miro-cz.wav"
    make_enrollment_wav_from_test_audio(src_wav_path=test_wav_path, out_wav_path=enroll_wav, seconds=5.0)

    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=localstack_s3["endpoint_url"],
        region_name=localstack_s3["region"],
        aws_access_key_id=localstack_s3["access_key"],
        aws_secret_access_key=localstack_s3["secret_key"],
    )

    tenant = "t-test"
    upload_tenant_enrollment_to_s3(
        s3_client=s3,
        bucket=localstack_s3["bucket"],
        prefix=localstack_s3["prefix"],
        tenant_id=tenant,
        speaker_id="spk-1",
        label="Miro-cz",
        sample_wav_path=enroll_wav,
    )

    # Configure worker
    os.environ["KAFKA_BOOTSTRAP"] = kafka_bootstrap
    os.environ["KAFKA_TOPIC_AUDIO"] = topic_audio
    os.environ["KAFKA_TOPIC_REFINED"] = topic_refined
    os.environ["WHISPERX_SLICE_SECONDS"] = "10.0"
    os.environ["WHISPERX_ENABLE_DIARIZATION"] = "true"
    os.environ["WHISPERX_DIAR_MODEL"] = "pyannote/speaker-diarization-3.1"

    os.environ["ENROLL_BACKEND"] = "s3_manifest"
    os.environ["ENROLL_S3_BUCKET"] = localstack_s3["bucket"]
    os.environ["ENROLL_S3_PREFIX"] = localstack_s3["prefix"]
    os.environ["ENROLL_S3_ENDPOINT"] = localstack_s3["endpoint_url"]
    os.environ["ENROLL_S3_REGION"] = localstack_s3["region"]
    os.environ["ENROLL_S3_ACCESS_KEY"] = localstack_s3["access_key"]
    os.environ["ENROLL_S3_SECRET_KEY"] = localstack_s3["secret_key"]
    os.environ["ENROLL_S3_FORCE_PATH_STYLE"] = "true"

    # make mapping permissive
    os.environ["ENROLL_SIM_THRESHOLD"] = "-1.0"

    # make sure worker re-reads env
    import importlib
    from whisperx_worker import whisperx_worker

    importlib.reload(whisperx_worker)

    worker_thread = threading.Thread(
        target=whisperx_worker.main,
        name="whisperx-worker-main-diar-s3",
        daemon=True,
    )
    worker_thread.start()

    print("[test] waiting 30s for whisperx_worker (diar+s3) to start up…")
    time.sleep(30.0)

    session_id = "integration-whisperx-diar-s3-1"

    import soundfile as sf

    wav_data, wav_sr = sf.read(test_wav_path)
    if wav_data.ndim > 1:
        wav_data = wav_data[:, 0]

    if wav_sr != SR:
        ratio = SR / float(wav_sr)
        new_len = int(len(wav_data) * ratio)
        x_old = np.linspace(0, 1, len(wav_data), endpoint=False)
        x_new = np.linspace(0, 1, new_len, endpoint=False)
        wav_data = np.interp(x_new, x_old, wav_data)

    wav_data = wav_data.astype("float32")

    min_seconds = 11.0
    if len(wav_data) < min_seconds * SR:
        reps = int(np.ceil((min_seconds * SR) / len(wav_data)))
        wav_data = np.tile(wav_data, reps)

    wav_data = wav_data[: int(min_seconds * SR)]
    wav_data = np.clip(wav_data, -1.0, 1.0)
    pcm16 = (wav_data * 32767.0).astype("<i2")

    audio_chunk = stream_pb2.AudioChunk(
        session_id=session_id,
        t0_ns=0,
        pcm16_le=pcm16.tobytes(),
        sample_rate=SR,
        lang="cs",
        tenant_id=tenant,
    )

    producer = _make_producer(kafka_bootstrap)
    producer.produce(
        topic=topic_audio,
        key=session_id.encode("utf-8"),
        value=audio_chunk.SerializeToString(),
    )
    producer.flush(30_000)

    consumer = _make_consumer(kafka_bootstrap, group_id="test-whisperx-consumer-real-diar-s3")
    consumer.subscribe([topic_refined])

    deadline = time.time() + 240.0
    seen_speakers = []

    try:
        while time.time() < deadline:
            msg = consumer.poll(5.0)
            if msg is None:
                continue
            if msg.error():
                continue

            ev = stream_pb2.RefinedEvent()
            ev.ParseFromString(msg.value())
            for seg in ev.segments:
                if seg.speaker:
                    seen_speakers.append(seg.speaker)
            print(
                f"[test] refined segments={len(ev.segments)} speakers={sorted(set(seen_speakers))[:5]} "
                f"full_text_len={len(ev.full_text)}"
            )

            if any((seg.speaker == "Miro-cz") for seg in ev.segments):
                return
    finally:
        consumer.close()

    raise AssertionError(
        f"Did not see expected speaker label 'Miro-cz'. Seen speakers: {sorted(set(seen_speakers))}"
    )
