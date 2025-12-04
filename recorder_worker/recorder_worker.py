import logging
import os
import time
import wave
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from confluent_kafka import Consumer, Producer, KafkaException

import stream_pb2


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_AUDIO = os.getenv("KAFKA_TOPIC_AUDIO", "audio.raw")
TOPIC_RECORDING_FINALIZED = os.getenv("KAFKA_TOPIC_RECORDING_FINALIZED",
                                      "recordings.finalized")
GROUP_ID = os.getenv("KAFKA_GROUP_ID_RECORDER", "recorder-worker")

# Where to store recordings locally for now
RECORDING_DIR = os.getenv("RECORDING_DIR", "recordings")

# Idle timeout (seconds). If no new chunk for this session in this time, finalize.
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
logger = logging.getLogger(__name__)


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
# Recorder state
# --------------------------------------------------------------------------- #

class SessionRecording:
    """
    Holds state for one active recording:
    - WAV file handle
    - path
    - sample_rate
    - lang
    - total_samples written
    - last_activity timestamp
    """

    def __init__(self, session_id: str, sample_rate: int, lang: str, base_dir: str):
        self.session_id = session_id
        self.sample_rate = sample_rate or DEFAULT_SR
        self.lang = lang or ""
        self.base_dir = base_dir

        os.makedirs(self.base_dir, exist_ok=True)

        # File name: {session_id}_{start_ns}.wav (local path)
        now_ns = time.time_ns()
        self.path = os.path.join(
            self.base_dir,
            f"{self.session_id}_{now_ns}.wav",
        )

        logger.info("Opening WAV for session %s at %s (sr=%d, lang=%s)",
                    self.session_id, self.path, self.sample_rate, self.lang)

        self._wave = wave.open(self.path, "wb")
        self._wave.setnchannels(NUM_CHANNELS)
        self._wave.setsampwidth(SAMPLE_WIDTH_BYTES)
        self._wave.setframerate(self.sample_rate)

        self.total_samples = 0
        self.last_activity = time.time()

    def append_pcm16(self, pcm16: bytes):
        """Append raw PCM16 (little-endian) frames to the WAV."""
        if not pcm16:
            return
        self._wave.writeframes(pcm16)
        self.last_activity = time.time()
        self.total_samples += len(pcm16) // SAMPLE_WIDTH_BYTES

    def close(self):
        try:
            self._wave.close()
        except Exception:
            logger.exception("Error closing WAV for session %s", self.session_id)

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.total_samples) / float(self.sample_rate)

    @property
    def recording_url(self) -> str:
        # For now: file:// URL. Later: s3://bucket/... etc.
        abs_path = os.path.abspath(self.path)
        return f"file://{abs_path}"


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #

def finalize_session(session_id: str,
                     rec: SessionRecording,
                     producer: Producer):
    """Close WAV and emit RecordingFinalized for this session."""
    logger.info(
        "Finalizing session %s: path=%s dur=%.2fs sr=%d lang=%s",
        session_id, rec.path, rec.duration_s, rec.sample_rate, rec.lang
    )
    rec.close()

    event = stream_pb2.RecordingFinalized(
        session_id=session_id,
        recording_url=rec.recording_url,
        duration_s=rec.duration_s,
        sample_rate=rec.sample_rate,
        lang=rec.lang,
        created_at_ns=time.time_ns(),
    )

    producer.produce(
        topic=TOPIC_RECORDING_FINALIZED,
        key=session_id.encode("utf-8"),
        value=event.SerializeToString(),
    )
    producer.poll(0)


def main():
    logger.info("Starting recorder_worker")

    consumer = make_consumer()
    producer = make_producer()
    consumer.subscribe([TOPIC_AUDIO])

    # session_id -> SessionRecording
    sessions: Dict[str, SessionRecording] = {}

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            now = time.time()

            # 1) Idle session finalization check
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
            lang = getattr(audio_chunk, "lang", "")

            # 2) Get/create session recording
            rec = sessions.get(session_id)
            if rec is None:
                rec = SessionRecording(
                    session_id=session_id,
                    sample_rate=sr,
                    lang=lang,
                    base_dir=RECORDING_DIR,
                )
                sessions[session_id] = rec

            # 3) Append PCM16 data
            pcm_bytes = audio_chunk.pcm16_le
            if not pcm_bytes:
                logger.debug("Empty pcm16_le for session %s; skipping.", session_id)
            else:
                rec.append_pcm16(pcm_bytes)

            # 4) Commit Kafka offset
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
