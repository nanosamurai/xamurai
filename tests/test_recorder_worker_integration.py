"""Exercise the recorder process against real Kafka, including ordered completion."""
import io
import os
import subprocess
import sys
import time
import uuid
import wave
from urllib.parse import urlparse

import pytest
from confluent_kafka import Consumer, Producer

from proto_gen import stream_pb2 as pb
from tests.conftest import _ensure_topics


@pytest.mark.integration
@pytest.mark.timeout(90)
@pytest.mark.parametrize("end_marker", [True, False])
@pytest.mark.parametrize("storage", ["local", "s3"])
def test_recorder_completion(kafka_bootstrap, tmp_path, end_marker, storage, request):
    token = uuid.uuid4().hex
    topic, finished_topic = f"audio.recorder.{token}", f"finished.recorder.{token}"
    _ensure_topics(kafka_bootstrap, [topic, finished_topic])
    env = dict(os.environ, KAFKA_BOOTSTRAP=kafka_bootstrap, KAFKA_TOPIC_AUDIO=topic,
               KAFKA_TOPIC_RECORDING_FINISHED=finished_topic, KAFKA_GROUP_ID_RECORDER=token,
               RECORDING_STORAGE_BACKEND=storage, RECORDING_DIR=str(tmp_path),
               RECORDER_IDLE_SECONDS="30" if end_marker else "3",
               PYTHONPATH=os.pathsep.join(sys.path), OTEL_SDK_DISABLED="true")
    if storage == "s3":
        import boto3

        connection = request.getfixturevalue("localstack_s3")
        env.update(S3_ENDPOINT=connection["endpoint_url"], S3_BUCKET=connection["bucket"],
                   S3_REGION=connection["region"], S3_ACCESS_KEY=connection["access_key"],
                   S3_SECRET_KEY=connection["secret_key"], S3_FORCE_PATH_STYLE="true",
                   S3_PREFIX=token)
        s3 = boto3.client("s3", endpoint_url=connection["endpoint_url"],
                          region_name=connection["region"],
                          aws_access_key_id=connection["access_key"],
                          aws_secret_access_key=connection["secret_key"])
    consumer = Consumer({"bootstrap.servers": kafka_bootstrap, "group.id": token,
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    consumer.subscribe([finished_topic])
    producer = Producer({"bootstrap.servers": kafka_bootstrap, "enable.idempotence": True})
    pcm = bytes(range(256)) * 125
    chunk = pb.AudioChunk(session_id=token, tenant_id="tenant", sample_rate=16000, lang="en")
    headers = [("x-final-tracks", b"qwen"), ("x-store-recording", b"false")]
    with (tmp_path / "worker.log").open("w") as log:
        worker = subprocess.Popen([sys.executable, "-m", "recorder_worker.recorder_worker"],
                                  env=env, stdout=log, stderr=log)
        try:
            for start in range(0, len(pcm), 6400):
                chunk.pcm16_le = pcm[start:start + 6400]
                producer.produce(topic, key=token, value=chunk.SerializeToString(), headers=headers)
                if start == 0 and end_marker:
                    invalid = pb.AudioChunk(session_id=token, tenant_id="foreign", sample_rate=16000)
                    producer.produce(topic, key=token, value=invalid.SerializeToString(),
                                     headers=headers + [("x-audio-end", b"true")])
                    invalid.tenant_id = "tenant"
                    producer.produce(topic, key="wrong-key", value=invalid.SerializeToString(),
                                     headers=headers + [("x-audio-end", b"true")])
            chunk.pcm16_le = b""
            marker = chunk.SerializeToString()
            if end_marker:
                producer.produce(topic, key=token, value=marker,
                                 headers=headers + [("x-audio-end", b"true")])
            assert producer.flush(10) == 0
            sent = time.monotonic()
            deadline = sent + (15 if end_marker else 20)
            message = None
            while time.monotonic() < deadline:
                assert worker.poll() is None, (tmp_path / "worker.log").read_text()
                message = consumer.poll(.1)
                if message is not None:
                    assert not message.error(), message.error()
                    break
            assert message is not None, (tmp_path / "worker.log").read_text()
            elapsed = time.monotonic() - sent
            if not end_marker:
                assert elapsed >= 3
            event = pb.RecordingFinished.FromString(message.value())
            assert event.session_id == token and message.key() == token.encode()
            assert (event.sample_rate, event.duration_s, event.tenant_id) == (16000, 1, "tenant")
            assert dict(message.headers())["x-final-tracks"] == b"qwen"
            assert dict(message.headers())["x-store-recording"] == b"false"
            if storage == "s3":
                location = urlparse(event.recording_url)
                assert location.scheme == "s3" and location.netloc == connection["bucket"]
                assert location.path.startswith(f"/{token}/tenant/")
                response = s3.get_object(Bucket=location.netloc, Key=location.path.lstrip("/"))
                with response["Body"] as body:
                    recording = io.BytesIO(body.read())
            else:
                recording = str(next(tmp_path.glob("*.wav")))
            with wave.open(recording, "rb") as wav:
                assert wav.readframes(wav.getnframes()) == pcm
            if end_marker:
                # Repeated markers and markers for an empty session must create no recording.
                producer.produce(topic, key=token, value=marker, headers=[("x-audio-end", b"true")])
                chunk.session_id = "empty-" + token
                producer.produce(topic, key=chunk.session_id, value=chunk.SerializeToString(),
                                 headers=[("x-audio-end", b"true")])
                assert producer.flush(10) == 0
                assert consumer.poll(2) is None
                if storage == "s3":
                    assert s3.list_objects_v2(Bucket=connection["bucket"], Prefix=token)["KeyCount"] == 1
                else:
                    assert len(list(tmp_path.glob("*.wav"))) == 1
        finally:
            worker.terminate()
            worker.wait(timeout=10)
            consumer.close()


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_recorder_converts_uploaded_source(kafka_bootstrap, tmp_path, localstack_s3):
    """An uploaded file becomes a normal 16 kHz mono recording and its source is deleted."""
    import shutil

    import boto3

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    token = uuid.uuid4().hex
    session_id = str(uuid.uuid4())
    uploads_topic, finished_topic = f"uploaded.recorder.{token}", f"finished.recorder.{token}"
    _ensure_topics(kafka_bootstrap, [f"audio.recorder.{token}", uploads_topic, finished_topic])
    connection = localstack_s3
    s3 = boto3.client("s3", endpoint_url=connection["endpoint_url"], region_name=connection["region"],
                      aws_access_key_id=connection["access_key"],
                      aws_secret_access_key=connection["secret_key"])
    source_key = f"{token}/tenant/uploads/{session_id}/source"
    stereo = io.BytesIO()
    with wave.open(stereo, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\x01\x00\x02\x00" * 44100)
    s3.put_object(Bucket=connection["bucket"], Key=source_key, Body=stereo.getvalue())
    env = dict(os.environ, KAFKA_BOOTSTRAP=kafka_bootstrap, KAFKA_TOPIC_AUDIO=f"audio.recorder.{token}",
               KAFKA_TOPIC_RECORDING_FINISHED=finished_topic, KAFKA_TOPIC_RECORDING_UPLOADED=uploads_topic,
               KAFKA_GROUP_ID_RECORDER=token, RECORDING_STORAGE_BACKEND="s3",
               S3_ENDPOINT=connection["endpoint_url"], S3_BUCKET=connection["bucket"],
               S3_REGION=connection["region"], S3_ACCESS_KEY=connection["access_key"],
               S3_SECRET_KEY=connection["secret_key"], S3_FORCE_PATH_STYLE="true", S3_PREFIX=token,
               PYTHONPATH=os.pathsep.join(sys.path), OTEL_SDK_DISABLED="true")
    consumer = Consumer({"bootstrap.servers": kafka_bootstrap, "group.id": token,
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    consumer.subscribe([finished_topic])
    producer = Producer({"bootstrap.servers": kafka_bootstrap, "enable.idempotence": True})
    event = pb.RecordingFinished(session_id=session_id, tenant_id="tenant", lang="en",
                                 recording_url=f"s3://{connection['bucket']}/{source_key}")
    headers = [("x-outputs", b"final"), ("x-final-tracks", b"whisperx"), ("x-store-recording", b"true")]
    producer.produce(uploads_topic, key=session_id, value=event.SerializeToString(), headers=headers)
    assert producer.flush(10) == 0
    with (tmp_path / "worker.log").open("w") as log:
        worker = subprocess.Popen([sys.executable, "-m", "recorder_worker.recorder_worker"],
                                  env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 60
            message = None
            while time.monotonic() < deadline and message is None:
                assert worker.poll() is None, (tmp_path / "worker.log").read_text()
                message = consumer.poll(.2)
            assert message is not None, (tmp_path / "worker.log").read_text()
            assert not message.error(), message.error()
            finished = pb.RecordingFinished.FromString(message.value())
            assert (finished.session_id, finished.tenant_id, finished.sample_rate) == (session_id, "tenant", 16000)
            assert abs(finished.duration_s - 1.0) < 0.05
            assert dict(message.headers())["x-final-tracks"] == b"whisperx"
            location = urlparse(finished.recording_url)
            assert location.path.startswith(f"/{token}/tenant/") and location.path.endswith("/audio.wav")
            body = s3.get_object(Bucket=location.netloc, Key=location.path.lstrip("/"))["Body"].read()
            with wave.open(io.BytesIO(body), "rb") as wav:
                assert (wav.getnchannels(), wav.getframerate()) == (1, 16000)
            assert s3.list_objects_v2(Bucket=connection["bucket"],
                                      Prefix=f"{token}/tenant/uploads/")["KeyCount"] == 0
        finally:
            worker.terminate()
            worker.wait(timeout=10)
            consumer.close()
