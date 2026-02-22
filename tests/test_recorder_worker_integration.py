# tests/test_recorder_worker_integration.py

import os
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
from confluent_kafka import Producer, Consumer

from tests.conftest import _ensure_topics

from proto_gen import stream_pb2  # or from proto import stream_pb2 if that's your layout

SR = 16000  # must match AudioChunk.sample_rate your system uses


def _make_producer(bootstrap: str) -> Producer:
    return Producer(
        {
            "bootstrap.servers": bootstrap,
            "client.id": "test-recorder-producer",
            "compression.type": "none",
            "linger.ms": 0,
        }
    )


@pytest.mark.timeout(300)
@pytest.mark.integration
def test_recorder_worker_records_wav_and_emits_finished(kafka_bootstrap: str):
    """
    End-to-end-ish test for recorder_worker:

    - Uses real Kafka (kafka_bootstrap fixture).
    - Configures recorder_worker to use 'fs' backend.
    - Starts recorder_worker.main() in a background thread.
    - Produces one AudioChunk to the audio topic.
    - Asserts that a .wav file appears in the default 'recordings/' directory.
    """

    # --------------------- 1) Configure env & paths --------------------- #
    topic_audio = "audio.raw.test"
    topic_finished = "recordings.finished.test"

    os.environ["KAFKA_BOOTSTRAP"] = kafka_bootstrap
    os.environ["KAFKA_TOPIC_AUDIO"] = topic_audio
    os.environ["KAFKA_TOPIC_RECORDING_FINISHED"] = topic_finished

    # Force local backend (your worker already logs "backend=local")
    os.environ["RECORDER_STORAGE_BACKEND"] = "fs"

    # The worker clearly uses a default 'recordings/' directory.
    # We'll align the test with that instead of trying to override it.
    output_dir = Path("recordings")
    output_dir.mkdir(exist_ok=True)

    session_id = "recorder-integration-1"

    # Clean old files for this session so we don't pick up stale ones
    for f in output_dir.glob(f"{session_id}_*.wav"):
        try:
            f.unlink()
        except OSError:
            pass

    # --------------------- 2) Ensure topics exist ------------------------- #
    _ensure_topics(kafka_bootstrap, [topic_audio, topic_finished])

    # --------------------- 3) Import & start worker ---------------------- #
    import importlib
    from recorder_worker import recorder_worker  # type: ignore

    importlib.reload(recorder_worker)

    worker_thread = threading.Thread(
        target=recorder_worker.main,
        name="recorder-worker-main",
        daemon=True,
    )
    worker_thread.start()

    # Give the worker a moment to connect & subscribe
    time.sleep(2.0)

    # --------------------- 3) Produce one AudioChunk --------------------- #
    # 0.5 s of silence at 16kHz
    n_samples = int(0.5 * SR)
    pcm = np.zeros(n_samples, dtype="<i2")

    chunk = stream_pb2.AudioChunk(
        session_id=session_id,
        t0_ns=0,
        pcm16_le=pcm.tobytes(),
        sample_rate=SR,
        lang="en",
    )

    producer = _make_producer(kafka_bootstrap)
    producer.produce(
        topic=topic_audio,
        key=session_id.encode("utf-8"),
        value=chunk.SerializeToString(),
    )
    producer.flush(10_000)

    # --------------------- 4) Wait for recorder & assert WAV -------------- #
    deadline = time.time() + 20.0

    wav_files: List[Path] = []
    while time.time() < deadline:
        wav_files = list(output_dir.glob(f"{session_id}_*.wav"))
        if wav_files:
            break
        time.sleep(1.5)

    assert wav_files, "Recorder worker did not produce any .wav files in 'recordings/'"

    out_path = wav_files[0]
    assert out_path.stat().st_size > 0, "Recorded WAV file is empty"

    # --------------------- 5) Assert RecordingFinished event -------------- #
    consumer = Consumer(
        {
            "bootstrap.servers": kafka_bootstrap,
            "group.id": "test-recorder-finished-consumer",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([topic_finished])

    finished: stream_pb2.RecordingFinished | None = None
    deadline2 = time.time() + 30.0

    try:
        while time.time() < deadline2:
            msg = consumer.poll(2.0)
            if msg is None:
                continue
            if msg.error():
                continue
            ev = stream_pb2.RecordingFinished()
            ev.ParseFromString(msg.value())
            if ev.session_id != session_id:
                continue
            finished = ev
            break
    finally:
        consumer.close()

    assert finished is not None, "Did not receive RecordingFinished for this session"
    assert finished.session_id == session_id
    assert finished.recording_url.startswith("file://")
    assert finished.sample_rate == SR
    assert finished.duration_s > 0.0
