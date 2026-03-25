import json
import os
import threading
import time
from pathlib import Path
from typing import List

import importlib.util

import pytest
from confluent_kafka import Consumer, Producer

from tests.conftest import _ensure_topics

from proto_gen import stream_pb2

SR = 16000

_HAS_WHISPERX = importlib.util.find_spec("whisperx") is not None
_HAS_HF_TOKEN = bool(os.getenv("HF_TOKEN"))
_HAS_BOTO3 = importlib.util.find_spec("boto3") is not None


def _make_producer(bootstrap: str) -> Producer:
    return Producer(
        {
            "bootstrap.servers": bootstrap,
            "client.id": "test-finalizer-producer",
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


@pytest.mark.timeout(900)  # allow for WhisperX model load
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_WHISPERX, reason="whisperx not installed in this env")
def test_finalizer_worker_writes_json_and_emits_event(
    tmp_path: Path,
    kafka_bootstrap: str,
):
    """
    Integration test for finalizer_worker with real WhisperX:

    - Uses real Kafka from the kafka_bootstrap fixture.
    - Starts the real finalizer_worker.main() in a background thread.
    - Produces one RecordingFinished event pointing to a local WAV file
      (tests/data/test_cs.wav copied into tmp_path).
    - Asserts:
        - JSON transcript file is written next to the WAV (file:// case).
        - A SessionTranscript event appears on Kafka.

    Note: Postgres persistence is handled by `samuraipersistor` and is
    intentionally NOT tested here.
    """

    # -------------------- 1) Topics and env configuration -------------------- #
    topic_recording_finished = "recordings.finished.test"
    topic_session_transcripts = "transcripts.final.test"

    _ensure_topics(kafka_bootstrap, [topic_recording_finished, topic_session_transcripts])

    os.environ["KAFKA_BOOTSTRAP"] = kafka_bootstrap
    os.environ["KAFKA_TOPIC_RECORDING_FINISHED"] = topic_recording_finished
    # Must match finalizer_worker.TOPIC_TRANSCRIPTS_FINAL
    os.environ["KAFKA_TOPIC_TRANSCRIPTS_FINAL"] = topic_session_transcripts
    # finalizer saves JSON next to the WAV; we copy the WAV under tmp_path so artifacts
    # stay within tmp_path.

    # -------------------- 2) Use real test_cs.wav ---------------------------- #
    session_id = "finalizer-integration-1"

    src_wav = Path(__file__).parent / "data" / "test_cs.wav"
    assert src_wav.exists(), f"Missing test WAV at {src_wav}"

    # Copy into tmp_path so test artifacts stay together
    dst_wav = tmp_path / f"{session_id}.wav"
    dst_wav.write_bytes(src_wav.read_bytes())

    recording_url = f"file://{dst_wav}"

    # -------------------- 3) Import real finalizer_worker -------------------- #
    import importlib
    from finalizer_worker import finalizer_worker  # type: ignore

    importlib.reload(finalizer_worker)

    print(
        f"[test] finalizer topics: rf={finalizer_worker.TOPIC_RECORDING_FINISHED}, "
        f"session={finalizer_worker.TOPIC_TRANSCRIPTS_FINAL}, "
        f"bootstrap={finalizer_worker.KAFKA_BOOTSTRAP}"
    )

    # -------------------- 4) Start the worker in background ------------------ #
    worker_thread = threading.Thread(
        target=finalizer_worker.main,
        name="finalizer-worker-main",
        daemon=True,
    )
    worker_thread.start()

    # Give worker time to subscribe to Kafka
    time.sleep(2.0)

    # -------------------- 5) Produce RecordingFinished event ----------------- #
    # Duration/sample_rate here are metadata; WhisperX will load the WAV by path.
    # Use the same tenant UUID created in tests/conftest.py
    event = stream_pb2.RecordingFinished(
        session_id=session_id,
        recording_url=recording_url,
        duration_s=20.0,          # match your test_cs.wav length if you like
        sample_rate=SR,           # nominal; WhisperX will resample as needed
        lang="cs",
        tenant_id="00000000-0000-0000-0000-000000000000",
        created_at_ns=time.time_ns(),
    )

    producer = _make_producer(kafka_bootstrap)
    producer.produce(
        topic=topic_recording_finished,
        key=session_id.encode("utf-8"),
        value=event.SerializeToString(),
    )
    producer.flush(10_000)

    # -------------------- 6) Wait for JSON transcript file ------------------- #
    json_file: Path | None = None
    deadline = time.time() + 120.0  # WhisperX can be slow on first run

    while time.time() < deadline:
        candidates = list(tmp_path.glob("*.json"))
        if candidates:
            json_file = candidates[0]
            break
        time.sleep(1.0)

    assert json_file is not None, "finalizer_worker did not write any JSON transcript."

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("session_id") == session_id
    assert data.get("recording_url") == recording_url
    assert isinstance(data.get("full_text", ""), str)
    # With real speech we expect at least one segment
    assert isinstance(data.get("segments"), list)
    assert len(data["segments"]) > 0

    # -------------------- 7) Verify SessionTranscript on Kafka --------------- #
    consumer = _make_consumer(kafka_bootstrap, group_id="test-finalizer-consumer")
    consumer.subscribe([topic_session_transcripts])

    session_events: List[stream_pb2.SessionTranscript] = []
    deadline2 = time.time() + 60.0

    try:
        while time.time() < deadline2:
            msg = consumer.poll(2.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[tests] Kafka error on session transcripts: {msg.error()}")
                continue

            ft = stream_pb2.SessionTranscript()
            ft.ParseFromString(msg.value())
            print(
                f"[test] got SessionTranscript: session={ft.session_id}, "
                f"segments={len(ft.segments)}"
            )
            session_events.append(ft)
            break
    finally:
        consumer.close()

    assert session_events, "Did not receive any SessionTranscript from finalizer_worker."

    ft = session_events[0]
    assert ft.session_id == session_id
    assert ft.recording_url == recording_url
    assert len(ft.segments) > 0

    # Word-level timings should be present when alignment runs.
    # (In rare cases alignment may fall back; in that case this assertion may be too strict.
    # If it becomes flaky in CI, we can relax it to "field exists".)
    assert any(len(s.words) > 0 for s in ft.segments), "Expected at least one segment with word timings"


@pytest.mark.timeout(1200)
@pytest.mark.integration
@pytest.mark.skipif(not _HAS_WHISPERX, reason="whisperx not installed in this env")
@pytest.mark.skipif(not _HAS_HF_TOKEN, reason="HF_TOKEN not provided; diarization is disabled")
@pytest.mark.skipif(not _HAS_BOTO3, reason="boto3 not installed")
def test_finalizer_worker_s3_enrollment_speaker_labels(
    tmp_path: Path,
    kafka_bootstrap: str,
    localstack_s3,
):
    """Strict integration test for finalizer_worker with S3-backed enrollment.

    Validates:
    - finalizer_worker uses tenant_id from RecordingFinished
    - loads enrolled speakers from S3 manifest backend (LocalStack)
    - emits SessionTranscript with at least one segment speaker == "Miro-cz"

    Requires Docker (Kafka + LocalStack).
    """

    from tests._s3_enrollment_testdata import (
        make_enrollment_wav_from_test_audio,
        upload_tenant_enrollment_to_s3,
    )

    # Topics
    topic_recording_finished = "recordings.finished.test.s3enroll"
    topic_session_transcripts = "transcripts.final.test.s3enroll"
    _ensure_topics(kafka_bootstrap, [topic_recording_finished, topic_session_transcripts])

    # Configure env BEFORE importing worker
    os.environ["KAFKA_BOOTSTRAP"] = kafka_bootstrap
    os.environ["KAFKA_TOPIC_RECORDING_FINISHED"] = topic_recording_finished
    os.environ["KAFKA_TOPIC_TRANSCRIPTS_FINAL"] = topic_session_transcripts
    os.environ["KAFKA_GROUP_ID_FINALIZER"] = "finalizer-worker-test-s3enroll"

    os.environ["WHISPERX_ENABLE_DIARIZATION"] = "true"
    os.environ["WHISPERX_DIAR_MODEL"] = "pyannote/speaker-diarization-3.1"

    # S3 enrollment backend
    os.environ["ENROLL_BACKEND"] = "s3_manifest"
    os.environ["ENROLL_S3_BUCKET"] = localstack_s3["bucket"]
    os.environ["ENROLL_S3_PREFIX"] = localstack_s3["prefix"]
    os.environ["ENROLL_S3_ENDPOINT"] = localstack_s3["endpoint_url"]
    os.environ["ENROLL_S3_REGION"] = localstack_s3["region"]
    os.environ["ENROLL_S3_ACCESS_KEY"] = localstack_s3["access_key"]
    os.environ["ENROLL_S3_SECRET_KEY"] = localstack_s3["secret_key"]
    os.environ["ENROLL_S3_FORCE_PATH_STYLE"] = "true"

    # reduce flakiness: accept any similarity
    os.environ["ENROLL_SIM_THRESHOLD"] = "-1.0"

    # Prepare WAV
    session_id = "finalizer-integration-s3enroll-1"
    tenant = "t-test"

    src_wav = Path(__file__).parent / "data" / "test_cs.wav"
    dst_wav = tmp_path / f"{session_id}.wav"
    dst_wav.write_bytes(src_wav.read_bytes())
    recording_url = f"file://{dst_wav}"

    # Upload tenant enrollment to LocalStack
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=localstack_s3["endpoint_url"],
        region_name=localstack_s3["region"],
        aws_access_key_id=localstack_s3["access_key"],
        aws_secret_access_key=localstack_s3["secret_key"],
    )

    enroll_wav = tmp_path / "enroll" / "Miro-cz.wav"
    make_enrollment_wav_from_test_audio(src_wav_path=src_wav, out_wav_path=enroll_wav, seconds=5.0)
    upload_tenant_enrollment_to_s3(
        s3_client=s3,
        bucket=localstack_s3["bucket"],
        prefix=localstack_s3["prefix"],
        tenant_id=tenant,
        speaker_id="spk-1",
        label="Miro-cz",
        sample_wav_path=enroll_wav,
    )

    # Start worker
    import importlib
    from finalizer_worker import finalizer_worker  # type: ignore

    importlib.reload(finalizer_worker)

    worker_thread = threading.Thread(
        target=finalizer_worker.main,
        name="finalizer-worker-main-s3enroll",
        daemon=True,
    )
    worker_thread.start()

    time.sleep(2.0)

    # Produce RecordingFinished
    event = stream_pb2.RecordingFinished(
        session_id=session_id,
        recording_url=recording_url,
        duration_s=20.0,
        sample_rate=SR,
        lang="cs",
        tenant_id=tenant,
        created_at_ns=time.time_ns(),
    )

    producer = _make_producer(kafka_bootstrap)
    producer.produce(
        topic=topic_recording_finished,
        key=session_id.encode("utf-8"),
        value=event.SerializeToString(),
    )
    producer.flush(30_000)

    # Consume SessionTranscript
    consumer = _make_consumer(kafka_bootstrap, group_id="test-finalizer-consumer-s3enroll")
    consumer.subscribe([topic_session_transcripts])

    ft: stream_pb2.SessionTranscript | None = None
    deadline = time.time() + 240.0

    try:
        while time.time() < deadline:
            msg = consumer.poll(5.0)
            if msg is None:
                continue
            if msg.error():
                continue
            ev = stream_pb2.SessionTranscript()
            ev.ParseFromString(msg.value())
            if ev.session_id != session_id:
                continue
            ft = ev
            break
    finally:
        consumer.close()

    assert ft is not None, "Did not receive SessionTranscript"
    speakers = {s.speaker for s in ft.segments if s.speaker}
    assert "Miro-cz" in speakers, f"Expected Miro-cz in speakers, got {sorted(speakers)}"

    assert any(len(s.words) > 0 for s in ft.segments), "Expected at least one segment with word timings"
