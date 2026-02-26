# finalizer_worker.py
import json
import logging
import os
import time
from threading import Event
from typing import Optional

from confluent_kafka import Consumer, Producer, KafkaException

from proto_gen import stream_pb2
from whisperx_worker.whisperx_worker import run_whisperx_diarized

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
TOPIC_RECORDING_FINISHED = os.getenv(
    "KAFKA_TOPIC_RECORDING_FINISHED", "recordings.finished"
)
TOPIC_TRANSCRIPTS_FINAL = os.getenv(
    "KAFKA_TOPIC_TRANSCRIPTS_FINAL", "transcripts.final"
)
GROUP_ID = os.getenv("KAFKA_GROUP_ID_FINALIZER", "finalizer-worker")

# Finalizer can do *very* long work per message (full session + alignment model downloads),
# so the default Kafka client max.poll.interval.ms (5 minutes) is too low.
# If exceeded, the consumer leaves the group mid-processing.
MAX_POLL_INTERVAL_MS = int(
    os.getenv(
        "FINALIZER_MAX_POLL_INTERVAL_MS",
        os.getenv("KAFKA_MAX_POLL_INTERVAL_MS", "1800000"),
    )
)

# Only commit the input offset after we know the output message was delivered.
FINALIZER_PRODUCE_ACK_TIMEOUT_S = float(
    os.getenv("FINALIZER_PRODUCE_ACK_TIMEOUT_S", "30.0")
)
FINALIZER_PRODUCE_RETRIES = int(os.getenv("FINALIZER_PRODUCE_RETRIES", "8"))
FINALIZER_PRODUCE_RETRY_BACKOFF_S = float(
    os.getenv("FINALIZER_PRODUCE_RETRY_BACKOFF_S", "1.0")
)


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
            # Must be higher than worst-case processing time for a single RecordingFinished
            # message (including first-run model downloads).
            "max.poll.interval.ms": MAX_POLL_INTERVAL_MS,
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
    """For file:// URLs, return local path.

    For now we only support local file recordings. S3 etc. can be added later.
    """

    if url.startswith("file://"):
        return url[len("file://") :]
    raise RuntimeError(f"Unsupported recording_url scheme for finalizer: {url}")


def _save_transcript_json(transcript: stream_pb2.SessionTranscript) -> Optional[str]:
    """Persist transcript JSON next to WAV for file:// URLs."""

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


def _produce_with_ack(
    producer: Producer,
    *,
    topic: str,
    key: bytes,
    value: bytes,
    headers: Optional[list[tuple[str, Optional[bytes]]]],
    timeout_s: float,
    retries: int,
    base_backoff_s: float,
) -> None:
    """Produce a message and wait until Kafka acks it.

    This is critical for finalizer_worker semantics: we must not commit the input
    offset until we are sure the output message has been accepted by Kafka.

    Notes:
    - confluent_kafka's `produce()` is async.
    - delivery callbacks run when we call `poll()` / `flush()`.
    """

    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        delivered = Event()
        delivery_err: dict[str, object] = {}

        def _on_delivery(err, _msg):
            delivery_err["err"] = err
            delivered.set()

        # Try producing, handle local queue being full.
        while True:
            try:
                producer.produce(
                    topic=topic,
                    key=key,
                    value=value,
                    headers=headers,
                    on_delivery=_on_delivery,
                )
                break
            except BufferError as e:
                # Local producer queue is full; poll to clear delivery reports.
                last_err = e
                producer.poll(0.2)

        # Let producer start IO.
        producer.poll(0)

        # Wait for delivery callback.
        if not delivered.wait(timeout_s):
            # Force IO + callbacks; after flush returns, any deliverable messages
            # should have triggered callbacks.
            producer.flush(timeout_s)
            producer.poll(0)

        err = delivery_err.get("err")
        if delivered.is_set() and err is None:
            return

        # Failed or timed out.
        if not delivered.is_set():
            last_err = TimeoutError(
                f"Kafka produce delivery timed out after {timeout_s:.1f}s"
            )
        else:
            # `err` is typically a KafkaError instance.
            last_err = KafkaException(err)  # type: ignore[arg-type]

        backoff_s = base_backoff_s * (2 ** (attempt - 1))
        logger.warning(
            "Produce to topic=%s failed (attempt %d/%d): %s. Retrying in %.1fs",
            topic,
            attempt,
            retries,
            last_err,
            backoff_s,
        )
        try:
            producer.flush(5.0)
        except Exception:
            pass
        time.sleep(backoff_s)

    raise RuntimeError(
        f"Failed to produce to topic={topic} after {retries} attempts: {last_err}"
    )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main():
    # Initialize OTEL SDK (no-op if deps missing)
    setup_otel(service_name=os.getenv("OTEL_SERVICE_NAME", "finalizer-worker"))

    logger.info("Starting finalizer_worker")

    import torch

    if not torch.cuda.is_available():
        logger.warning(
            "finalizer_worker: CUDA not available; running on CPU (torch=%s torch.version.cuda=%s)",
            getattr(torch, "__version__", "unknown"),
            getattr(getattr(torch, "version", None), "cuda", None),
        )
    else:
        logger.info(
            "finalizer_worker: CUDA available; will use GPU (torch=%s torch.version.cuda=%s)",
            getattr(torch, "__version__", "unknown"),
            getattr(getattr(torch, "version", None), "cuda", None),
        )

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

            # Extract upstream trace context from recordings.finished.
            with extracted_context_from_headers(msg.headers()):
                if trace is not None:
                    tracer = trace.get_tracer("finalizer_worker")
                    span_cm = tracer.start_as_current_span(
                        f"kafka.consume {TOPIC_RECORDING_FINISHED}"
                    )
                else:
                    span_cm = None

                if span_cm is not None:
                    span_cm.__enter__()

                try:
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
                        # Bad event is not retriable, commit and move on.
                        consumer.commit(msg, asynchronous=True)
                        continue

                    # Full-session WhisperX with alignment + optional diarization/enrollment.
                    # Enrollment backend is configured via env (ENROLL_BACKEND=...).
                    full_text, segments = run_whisperx_diarized(
                        wav_path,
                        tenant=(rf.tenant_id or None),
                        lang=(rf.lang or None),
                        use_alignment=True,
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
                    # IMPORTANT: commit the input offset only after Kafka acked this publish.
                    try:
                        _produce_with_ack(
                            producer,
                            topic=TOPIC_TRANSCRIPTS_FINAL,
                            key=rf.session_id.encode("utf-8"),
                            value=transcript.SerializeToString(),
                            headers=with_current_trace_context(),
                            timeout_s=FINALIZER_PRODUCE_ACK_TIMEOUT_S,
                            retries=FINALIZER_PRODUCE_RETRIES,
                            base_backoff_s=FINALIZER_PRODUCE_RETRY_BACKOFF_S,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to publish final transcript for session=%s; will NOT commit input offset.",
                            rf.session_id,
                        )
                        # Avoid a hot loop if Kafka is down.
                        time.sleep(1.0)
                        continue

                    # If commit fails, we may reprocess (at-least-once). That's preferred over
                    # committing before the output exists.
                    try:
                        consumer.commit(msg, asynchronous=False)
                    except Exception:
                        logger.exception(
                            "Offset commit failed after successful publish (session=%s). "
                            "Worker may reprocess this RecordingFinished.",
                            rf.session_id,
                        )

                finally:
                    if span_cm is not None:
                        span_cm.__exit__(None, None, None)

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
