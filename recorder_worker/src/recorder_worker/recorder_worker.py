import logging
import os
import time
import wave
from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import numpy as np
from confluent_kafka import Consumer, Producer, KafkaException

from proto_gen import stream_pb2

from drsynth_common.otel_setup import setup_otel
from drsynth_common.otel_kafka import extracted_context_from_headers, with_current_trace_context

try:
    from opentelemetry import trace
except Exception:  # pragma: no cover
    trace = None  # type: ignore[assignment]


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
# In containers/k8s, mount a volume to /data/recordings and set RECORDING_DIR accordingly.
RECORDING_DIR = os.getenv("RECORDING_DIR", "recordings")

# Storage backend: "local" or "s3"
RECORDING_STORAGE_BACKEND = os.getenv("RECORDING_STORAGE_BACKEND", "local").lower()

# S3 config (if backend=s3)
# Standardized env vars (works with AWS S3 and S3-compatible storage like Ceph/MinIO)
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "").strip()  # optional
S3_BUCKET = os.getenv("S3_BUCKET", "").strip()
S3_REGION = os.getenv("S3_REGION", "").strip()  # optional for some providers
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "").strip()
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "").strip()
S3_FORCE_PATH_STYLE = os.getenv("S3_FORCE_PATH_STYLE", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
)

# Optional prefix (not in the standardized list, but useful for multi-env layouts)
S3_PREFIX = os.getenv("S3_PREFIX", os.getenv("RECORDING_S3_PREFIX", "")).strip()  # e.g. "prod/"

# Backwards-compat: if someone still uses RECORDING_S3_BUCKET
if not S3_BUCKET:
    S3_BUCKET = os.getenv("RECORDING_S3_BUCKET", "").strip()

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
    return Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": GROUP_ID,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.partition.fetch.bytes": 5_000_000,
            "fetch.wait.max.ms": 50,
        }
    )


def make_producer() -> Producer:
    logger.info("Creating Kafka producer for recorder_worker")
    return Producer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "client.id": "recorder-worker",
            "compression.type": "zstd",
            "linger.ms": 10,
            "batch.size": 131072,
        }
    )


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
    """Write to a local temp WAV, then upload to S3 on close.

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
        from botocore.config import Config

        try:
            self._wave.close()
        except Exception:
            logger.exception("Error closing temp WAV %s", self._tmp_path)

        # Credentials:
        # - If S3_ACCESS_KEY / S3_SECRET_KEY are set, use them.
        # - Otherwise, fall back to boto3's default credential chain (IRSA / Pod Identity / env / instance role).
        session_kwargs = {}
        if S3_REGION:
            session_kwargs["region_name"] = S3_REGION

        if S3_ACCESS_KEY and S3_SECRET_KEY:
            session_kwargs["aws_access_key_id"] = S3_ACCESS_KEY
            session_kwargs["aws_secret_access_key"] = S3_SECRET_KEY

        addressing_style = "path" if S3_FORCE_PATH_STYLE else "virtual"
        cfg = Config(s3={"addressing_style": addressing_style})

        client_kwargs = {"config": cfg}
        if S3_ENDPOINT:
            client_kwargs["endpoint_url"] = S3_ENDPOINT

        s3 = boto3.client("s3", **session_kwargs, **client_kwargs)
        logger.info(
            "Uploading recording to s3://%s/%s (endpoint=%s, path_style=%s)",
            self._bucket,
            self._key,
            S3_ENDPOINT or "aws",
            addressing_style,
        )
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

    # Kafka trace context headers captured from the last consumed AudioChunk.
    # We store them so finalization can publish RecordingFinished with the
    # *same* traceparent even if it happens later (idle timeout).
    trace_headers: Optional[list[tuple[str, Optional[bytes]]]] = None

    total_samples: int = 0
    last_activity: float = 0.0

    # Optional long-lived span for the whole recording session.
    # This reduces span explosion vs per-audio-chunk spans.
    session_span_cm: object = None

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


def make_writer_for_session(session_id: str, tenant_id: str, sample_rate: int) -> BaseRecordingWriter:
    if RECORDING_STORAGE_BACKEND == "s3":
        # Build an S3 key like: <prefix>/<tenant>/<YYYY>/<MM>/<DD>/<session>/audio.wav
        from datetime import datetime, timezone

        if not S3_BUCKET:
            raise RuntimeError("S3_BUCKET not configured for S3 backend")

        now = datetime.now(timezone.utc)
        y = now.year
        m = now.month
        d = now.day

        tenant = tenant_id or "default"
        prefix = S3_PREFIX.strip("/")
        base = f"{prefix}/{tenant}/{y:04d}/{m:02d}/{d:02d}/{session_id}".strip("/")

        key = f"{base}/audio.wav"
        logger.info("Using S3 backend for session %s → s3://%s/%s", session_id, S3_BUCKET, key)
        return S3FileWriter(S3_BUCKET, key, sample_rate)

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

def finalize_session(session_id: str, rec: SessionRecording, producer: Producer) -> None:
    # Close long-lived recording span (best-effort) before producing finished event.
    try:
        if rec.session_span_cm is not None:
            rec.session_span_cm.__exit__(None, None, None)
            rec.session_span_cm = None
    except Exception:
        pass

    url = rec.close_and_get_url()
    logger.info(
        "Finalized session %s: url=%s dur=%.2fs sr=%d lang=%s tenant=%s",
        session_id,
        url,
        rec.duration_s,
        rec.sample_rate,
        rec.lang,
        rec.tenant_id,
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

    # Re-attach trace context captured from audio.raw consumption.
    with extracted_context_from_headers(rec.trace_headers):
        span_cm = None
        if trace is not None:
            tracer = trace.get_tracer("recorder_worker")
            span_cm = tracer.start_as_current_span("recorder.publish_recording_finished")
            span_cm.__enter__()
            try:
                span = trace.get_current_span()
                if hasattr(span, "set_attribute"):
                    span.set_attribute("nanosamurai.session_id", session_id)
                    if rec.tenant_id:
                        span.set_attribute("nanosamurai.tenant_id", rec.tenant_id)
                    span.set_attribute("nanosamurai.duration_s", float(rec.duration_s))
                    span.set_attribute("nanosamurai.sample_rate", int(rec.sample_rate))
            except Exception:
                pass

        try:
            producer.produce(
                topic=TOPIC_RECORDING_FINISHED,
                key=session_id.encode("utf-8"),
                value=event.SerializeToString(),
                headers=with_current_trace_context(),
            )
            producer.poll(0)
        finally:
            if span_cm is not None:
                span_cm.__exit__(None, None, None)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main():
    # Initialize OTEL SDK (no-op if deps missing)
    setup_otel(service_name=os.getenv("OTEL_SERVICE_NAME", "recorder-worker"))

    logger.info("Starting recorder_worker with backend=%s", RECORDING_STORAGE_BACKEND)

    if RECORDING_STORAGE_BACKEND == "s3" and not S3_BUCKET:
        raise RuntimeError("RECORDING_STORAGE_BACKEND=s3 but S3_BUCKET is not set")

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
                sid for sid, rec in sessions.items() if now - rec.last_activity > SESSION_IDLE_SEC
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

            # Extract upstream context (if present) from Kafka headers.
            with extracted_context_from_headers(msg.headers()):
                try:
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

                        # Start a single long-lived span for the whole recording session.
                        if trace is not None:
                            try:
                                tracer = trace.get_tracer("recorder_worker")
                                rec.session_span_cm = tracer.start_as_current_span("recorder.session")
                                rec.session_span_cm.__enter__()
                                span = trace.get_current_span()
                                if hasattr(span, "set_attribute"):
                                    span.set_attribute("nanosamurai.session_id", session_id)
                                    if tenant_id:
                                        span.set_attribute("nanosamurai.tenant_id", tenant_id)
                                    span.set_attribute("nanosamurai.sample_rate", int(sr))
                                    if lang:
                                        span.set_attribute("nanosamurai.lang", str(lang))
                                    span.set_attribute("nanosamurai.storage_backend", str(RECORDING_STORAGE_BACKEND))
                            except Exception:
                                rec.session_span_cm = None

                    # Keep latest kafka headers for end-to-end propagation.
                    rec.trace_headers = msg.headers() or rec.trace_headers

                    pcm_bytes = audio_chunk.pcm16_le
                    if not pcm_bytes:
                        logger.debug("Empty pcm16_le for session %s; skipping.", session_id)
                    else:
                        rec.append_pcm16(pcm_bytes)

                    consumer.commit(msg, asynchronous=True)
                finally:
                    pass

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
