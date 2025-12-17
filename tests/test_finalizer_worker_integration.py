import json
import os
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
from confluent_kafka.admin import AdminClient, NewTopic

import stream_pb2
from conftest import _make_producer, _make_consumer  # adjust if needed

SR = 16000


@pytest.mark.timeout(900)  # allow for WhisperX model load
@pytest.mark.integration
def test_finalizer_worker_writes_json_and_emits_event(tmp_path: Path, kafka_bootstrap: str):
    """
    Integration test for finalizer_worker with real WhisperX:

    - Uses real Kafka from the kafka_bootstrap fixture.
    - Starts the real finalizer_worker.main() in a background thread.
    - Produces one RecordingFinished event pointing to a local WAV file
      (tests/data/test_cs.wav copied into tmp_path).
    - Asserts:
        - JSON transcript file is written to TRANSCRIPTS_DIR.
        - A SessionTranscript event appears on transcripts.final.
    """

    # -------------------- 1) Topics and env configuration -------------------- #
    topic_recording_finished = "recordings.finished.test"
    topic_session_transcripts = "transcripts.final.test"

    admin = AdminClient({"bootstrap.servers": kafka_bootstrap})
    existing = admin.list_topics(timeout=10).topics.keys()

    new_topics: List[NewTopic] = []
    for t in [topic_recording_finished, topic_session_transcripts]:
        if t not in existing:
            new_topics.append(
                NewTopic(
                    topic=t,
                    num_partitions=1,
                    replication_factor=1,
                )
            )
    if new_topics:
        fs = admin.create_topics(new_topics)
        for t, f in fs.items():
            try:
                f.result()
            except Exception as e:
                # topic may already exist due to race; not fatal in tests
                print(f"[tests] create_topics warning for {t}: {e}")

    os.environ["KAFKA_BOOTSTRAP"] = kafka_bootstrap
    os.environ["KAFKA_TOPIC_RECORDING_FINISHED"] = topic_recording_finished
    # Must match finalizer_worker.TOPIC_FULL_TRANSCRIPTS
    os.environ["KAFKA_TOPIC_FULL_TRANSCRIPTS"] = topic_session_transcripts
    os.environ["TRANSCRIPTS_DIR"] = str(tmp_path)

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
    event = stream_pb2.RecordingFinished(
        session_id=session_id,
        recording_url=recording_url,
        duration_s=20.0,          # match your test_cs.wav length if you like
        sample_rate=SR,           # nominal; WhisperX will resample as needed
        lang="cs",
        tenant_id="tenant-1",
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
