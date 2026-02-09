# finalizer_worker.py
import json
import logging
import os
import uuid
from typing import Optional, List, Tuple

from confluent_kafka import Consumer, Producer, KafkaException

from proto_gen import stream_pb2
from whisperx_worker.whisperx_worker import run_whisperx


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_RECORDING_FINISHED = os.getenv(
    "KAFKA_TOPIC_RECORDING_FINISHED", "recordings.finished"
)
TOPIC_TRANSCRIPTS_FINAL = os.getenv(
    "KAFKA_TOPIC_TRANSCRIPTS_FINAL", "transcripts.final"
)
GROUP_ID = os.getenv("KAFKA_GROUP_ID_FINALIZER", "finalizer-worker")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finalizer_worker")


# --------------------------------------------------------------------------- #
# Kafka helpers
# --------------------------------------------------------------------------- #

def make_consumer() -> Consumer:
    logger.info("Creating Kafka consumer for finalizer_worker")
    return Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": GROUP_ID,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.partition.fetch.bytes": 10_000_000,
            "fetch.wait.max.ms": 50,
        }
    )


def make_producer() -> Producer:
    logger.info("Creating Kafka producer for finalizer_worker")
    return Producer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "client.id": "finalizer-worker",
            "compression.type": "zstd",
            "linger.ms": 10,
            "batch.size": 131072,
        }
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_local_wav_from_url(url: str) -> str:
    """
    For file:// URLs, return local path.

    For now we only support local file recordings. S3 etc. can be added later.
    """
    if url.startswith("file://"):
        return url[len("file://") :]
    raise RuntimeError(f"Unsupported recording_url scheme for finalizer: {url}")


def _save_transcript_json(transcript: stream_pb2.SessionTranscript) -> Optional[str]:
    """
    Persist a JSON representation of the transcript next to the WAV file
    (only for file:// URLs). Returns path to the JSON file, or None if skipped.
    """
    url = transcript.recording_url
    if not url.startswith("file://"):
        logger.info(
            "Recording URL is not file:// (%s); skipping local JSON save for now.",
            url,
        )
        return None

    wav_path = url[len("file://") :]
    base, _ = os.path.splitext(wav_path)
    json_path = base + ".json"

    data = {
        "session_id": transcript.session_id,
        "recording_url": transcript.recording_url,
        "lang": transcript.lang,
        "duration_s": transcript.duration_s,
        "tenant_id": transcript.tenant_id,
        "created_at_ns": transcript.created_at_ns,
        "full_text": transcript.full_text,
        "segments": [
            {
                "start_s": seg.start_s,
                "end_s": seg.end_s,
                "text": seg.text,
                "speaker": seg.speaker,
            }
            for seg in transcript.segments
        ],
    }

    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Saved JSON transcript to %s", json_path)
    return json_path


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main():
    logger.info("Starting finalizer_worker")

    consumer = make_consumer()
    producer = make_producer()
    consumer.subscribe([TOPIC_RECORDING_FINISHED])

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Kafka error: %s", msg.error())
                raise KafkaException(msg.error())

            rf = stream_pb2.RecordingFinished()
            rf.ParseFromString(msg.value())

            logger.info(
                "Processing RecordingFinished: session=%s url=%s dur=%.2fs sr=%d lang=%s tenant=%s",
                rf.session_id,
                rf.recording_url,
                rf.duration_s,
                rf.sample_rate,
                rf.lang,
                rf.tenant_id,
            )

            try:
                wav_path = _load_local_wav_from_url(rf.recording_url)
            except Exception as e:
                logger.exception(
                    "Cannot resolve recording_url=%s, skipping: %s",
                    rf.recording_url,
                    e,
                )
                consumer.commit(msg, asynchronous=True)
                continue

            # Full-session WhisperX with alignment
            full_text, segments = run_whisperx(
                wav_path,
                lang=rf.lang or None,
                use_alignment=True,
                alignment_min_coverage=0.7,
            )

            transcript = stream_pb2.SessionTranscript(
                session_id=rf.session_id,
                recording_url=rf.recording_url,
                lang=rf.lang,
                duration_s=rf.duration_s,
                full_text=full_text,
                tenant_id=rf.tenant_id,
                created_at_ns=rf.created_at_ns,
            )

            for (s0, s1, text, speaker) in segments:
                seg = transcript.segments.add()
                seg.start_s = s0
                seg.end_s = s1
                seg.text = text
                seg.speaker = speaker or ""

            # 1) Persist JSON next to WAV
            _ = _save_transcript_json(transcript)

            # 2) Publish to Kafka for downstream consumers (persistence handled by samuraipersistor)
            producer.produce(
                topic=TOPIC_TRANSCRIPTS_FINAL,
                key=rf.session_id.encode("utf-8"),
                value=transcript.SerializeToString(),
            )
            producer.poll(0)

            consumer.commit(msg, asynchronous=True)

    except KeyboardInterrupt:
        logger.info("Stopping finalizer_worker (KeyboardInterrupt)")
    finally:
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
