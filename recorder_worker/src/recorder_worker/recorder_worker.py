import logging
import os
import time
import wave
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import numpy as np
from confluent_kafka import Consumer, Producer, KafkaException

from proto import stream_pb2


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_AUDIO = os.getenv("KAFKA_TOPIC_AUDIO", "audio.raw")
TOPIC_RECORDING_FINISHED = os.getenv(
    "KAFKA_TOPIC_RECORDING_FINISHED", "recordings.finished"
)
GROUP_ID = os.getenv("KAFKA_GROUP_ID_RECORDER", "recorder-worker")

# Where to store recordings locally (if backend=local)
RECORDING_DIR = os.getenv("RECORDING_DIR", "recordings")

# Storage backend: "local" or "s3"
RECORDING_STORAGE_BACKEND = os.getenv("RECORDING_STORAGE_BACKEND", "local").lower()

# S3 config (if backend=s3)
RECORDING_S3_BUCKET = os.getenv("RECORDING_S3_BUCKET", "")
RECORDING_S3_PREFIX = os.getenv("RECORDING_S3_PREFIX", "")  # e.g. "prod/"

# Idle timeout (seconds) after which we consider the session finished
SESSION_IDLE_SEC = float(os.getenv("RECORDER_IDLE_SECONDS", "30.0"))

DEFAULT_SR = 16000
NUM_CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("recorder_worker")


# --------------------------------------------------------------------------- #
# Kafka helpers
# --------------------------------------------------------------------------- #

def make_consumer() -> Consumer:
    logger.info("Creating Kafka consumer for recorder_worker")
    return Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "max.partition.fetch.bytes": 5_000_000,
        "fetch.wait.max.ms": 50,
    })


def make_producer() -> Producer:
    logger.info("Creating Kafka producer for recorder_worker")
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "client.id": "recorder-worker",
        "compression.type": "zstd",
        "linger.ms": 10,
        "batch.size": 131072,
    })


# --------------------------------------------------------------------------- #
# Storage abstraction
# --------------------------------------------------------------------------- #

class BaseRecordingWriter(Protocol):
    """Simple interface for “append PCM16 samples” + “close & return URL”."""

    def append_pcm16(self, pcm16: bytes) -> None: ...
    def close_and_get_url(self) -> str: ...


class LocalFileWriter:
    """Write directly to a local WAV file and return file:// URL."""

    def __init__(self, path: str, sample_rate: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._path = path
        self._wave = wave.open(self._path, "wb")
        self._wave.setnchannels(NUM_CHANNELS)
        self._wave.setsampwidth(SAMPLE_WIDTH_BYTES)
        self._wave.setframerate(sample_rate)

    def append_pcm16(self, pcm16: bytes) -> None:
        if pcm16:
            self._wave.writeframes(pcm16)

    def close_and_get_url(self) -> str:
        try:
            self._wave.close()
        except Exception:
            logger.exception("Error closing local WAV %s", self._path)
        abs_path = os.path.abspath(self._path)
        return f"file://{abs_path}"


class S3FileWriter:
    """
    Write to a local temp WAV, then upload to S3 on close.

    URL format: s3://<bucket>/<key>
    """

    def __init__(self, bucket: str, key: str, sample_rate: int):
        if not bucket:
            raise RuntimeError("S3 bucket not configured for S3FileWriter.")
        import tempfile

        self._bucket = bucket
        self._key = key

        # Local temp file
        self._tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._tmp_path = self._tmp.name
        self._tmp.close()

        self._wave = wave.open(self._tmp_path, "wb")
        self._wave.setnchannels(NUM_CHANNELS)
        self._wave.setsampwidth(SAMPLE_WIDTH_BYTES)
        self._wave.setframerate(sample_rate)

    def append_pcm16(self, pcm16: bytes) -> None:
        if pcm16:
            self._wave.writeframes(pcm16)

    def close_and_get_url(self) -> str:
        import boto3

        try:
            self._wave.close()
        except Exception:
            logger.exception("Error closing temp WAV %s", self._tmp_path)

        s3 = boto3.client("s3")
        logger.info("Uploading recording to s3://%s/%s", self._bucket, self._key)
        s3.upload_file(self._tmp_path, self._bucket, self._key)

        try:
            os.unlink(self._tmp_path)
        except Exception:
            logger.warning("Failed to delete temp file %s", self._tmp_path)

        return f"s3://{self._bucket}/{self._key}"


# --------------------------------------------------------------------------- #
# Per-session recording state
# --------------------------------------------------------------------------- #

@dataclass
class SessionRecording:
    session_id: str
    tenant_id: str
    sample_rate: int
    lang: str
    writer: BaseRecordingWriter

    total_samples: int = 0
    last_activity: float = 0.0

    def append_pcm16(self, pcm16: bytes) -> None:
        self.writer.append_pcm16(pcm16)
        self.last_activity = time.time()
        self.total_samples += len(pcm16) // SAMPLE_WIDTH_BYTES

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.total_samples) / float(self.sample_rate)

    def close_and_get_url(self) -> str:
        return self.writer.close_and_get_url()


def make_writer_for_session(session_id: str,
                            tenant_id: str,
                            sample_rate: int) -> BaseRecordingWriter:
    if RECORDING_STORAGE_BACKEND == "s3":
        # Build an S3 key like: <prefix>/<tenant>/<YYYY>/<MM>/<DD>/<session>/audio.wav
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        y = now.year
        m = now.month
        d = now.day

        tenant = tenant_id or "default"
        prefix = RECORDING_S3_PREFIX.strip("/")
        base = f"{prefix}/{tenant}/{y:04d}/{m:02d}/{d:02d}/{session_id}".strip("/")

        key = f"{base}/audio.wav"
        logger.info("Using S3 backend for session %s → s3://%s/%s",
                    session_id, RECORDING_S3_BUCKET, key)
        return S3FileWriter(RECORDING_S3_BUCKET, key, sample_rate)

    # Default: local
    os.makedirs(RECORDING_DIR, exist_ok=True)
    now_ns = time.time_ns()
    filename = f"{session_id}_{now_ns}.wav"
    path = os.path.join(RECORDING_DIR, filename)
    logger.info("Using local backend for session %s → %s", session_id, path)
    return LocalFileWriter(path, sample_rate)


# --------------------------------------------------------------------------- #
# Finalization
# --------------------------------------------------------------------------- #

def finalize_session(session_id: str,
                     rec: SessionRecording,
                     producer: Producer) -> None:
    url = rec.close_and_get_url()
    logger.info(
        "Finalized session %s: url=%s dur=%.2fs sr=%d lang=%s tenant=%s",
        session_id, url, rec.duration_s, rec.sample_rate, rec.lang, rec.tenant_id
    )

    event = stream_pb2.RecordingFinished(
        session_id=session_id,
        recording_url=url,
        duration_s=rec.duration_s,
        sample_rate=rec.sample_rate,
        lang=rec.lang,
        tenant_id=rec.tenant_id,
        created_at_ns=time.time_ns(),
    )

    producer.produce(
        topic=TOPIC_RECORDING_FINISHED,
        key=session_id.encode("utf-8"),
        value=event.SerializeToString(),
    )
    producer.poll(0)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main():
    logger.info(
        "Starting recorder_worker with backend=%s", RECORDING_STORAGE_BACKEND
    )

    if RECORDING_STORAGE_BACKEND == "s3" and not RECORDING_S3_BUCKET:
        raise RuntimeError(
            "RECORDING_STORAGE_BACKEND=s3 but RECORDING_S3_BUCKET is not set"
        )

    consumer = make_consumer()
    producer = make_producer()
    consumer.subscribe([TOPIC_AUDIO])

    # session_id -> SessionRecording
    sessions: Dict[str, SessionRecording] = {}

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            now = time.time()

            # 1) Idle session finalization
            idle_sessions = [
                sid for sid, rec in sessions.items()
                if now - rec.last_activity > SESSION_IDLE_SEC
            ]
            for sid in idle_sessions:
                rec = sessions.pop(sid, None)
                if rec is not None:
                    finalize_session(sid, rec, producer)

            if msg is None:
                continue

            if msg.error():
                logger.error("Kafka error: %s", msg.error())
                raise KafkaException(msg.error())

            audio_chunk = stream_pb2.AudioChunk()
            audio_chunk.ParseFromString(msg.value())

            session_id = audio_chunk.session_id or ""
            if not session_id:
                logger.warning("AudioChunk without session_id, skipping.")
                consumer.commit(msg, asynchronous=True)
                continue

            sr = getattr(audio_chunk, "sample_rate", DEFAULT_SR) or DEFAULT_SR
            lang = getattr(audio_chunk, "lang", "") or ""
            tenant_id = getattr(audio_chunk, "tenant_id", "") or "default"

            rec = sessions.get(session_id)
            if rec is None:
                writer = make_writer_for_session(session_id, tenant_id, sr)
                rec = SessionRecording(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    sample_rate=sr,
                    lang=lang,
                    writer=writer,
                    last_activity=time.time(),
                )
                sessions[session_id] = rec

            pcm_bytes = audio_chunk.pcm16_le
            if not pcm_bytes:
                logger.debug("Empty pcm16_le for session %s; skipping.", session_id)
            else:
                rec.append_pcm16(pcm_bytes)

            consumer.commit(msg, asynchronous=True)

    except KeyboardInterrupt:
        logger.info("Stopping recorder_worker (KeyboardInterrupt)")
    finally:
        # Finalize all active sessions on shutdown
        for sid, rec in list(sessions.items()):
            finalize_session(sid, rec, producer)
        sessions.clear()

        try:
            consumer.close()
        except Exception:
            pass
        try:
            producer.flush(2.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
