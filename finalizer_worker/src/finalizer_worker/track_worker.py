"""Shared source-topic finalizer: one deployment/group per track and profile."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from confluent_kafka import Consumer, Producer, KafkaException, TopicPartition
from proto_gen import stream_pb2 as pb
from drsynth_common.final_tracks import FINAL_TRACK_TOPIC, MAX_EVENT_BYTES, ContractError, read_plan, require_name
from drsynth_common.final_track_artifacts import S3Artifacts, s3_client
from drsynth_common.kafka_ack import produce_acked
from drsynth_common.logging_setup import setup_logging
from finalizer_worker.track_processing import process_recording, outcome_headers
from drsynth_common import refinement_tracks

logger = logging.getLogger(__name__)


def kafka_config():
    """Use the existing operator-owned Kafka security settings."""
    config = {"bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")}
    protocol = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    config["security.protocol"] = protocol
    if os.getenv("KAFKA_SSL_CA_LOCATION"):
        config["ssl.ca.location"] = os.environ["KAFKA_SSL_CA_LOCATION"]
    return config


def handle_message(message, *, producer, stage="final", **processing):
    """Publish immutable canonical and primary records before allowing a commit."""
    if message.value() is None or len(message.value()) > MAX_EVENT_BYTES:
        raise ContractError("recording_event_too_large")
    window = pb.RefinementWindow.FromString(message.value()) if stage == "refined" else None
    event = window.recording if window is not None else pb.RecordingFinished.FromString(message.value())
    if read_plan(message.headers(), event.tenant_id, event.session_id) is None:
        if event.HasField("final_plan") or event.HasField("source"):
            raise ContractError("missing_plan_headers")
        return
    if message.key() != event.session_id.encode():
        raise ContractError("recording_key_mismatch")
    accepted = process_recording(event, message.headers(), window=window, **processing)
    if accepted is None:
        return
    outcome, canonical, primary = accepted
    headers = outcome_headers(outcome)
    headers.extend((key, value) for key, value in (message.headers() or [])
                   if key in {"traceparent", "tracestate"})
    key = event.session_id.encode()
    produce_acked(producer, topic=refinement_tracks.RESULT_TOPIC if window is not None else FINAL_TRACK_TOPIC,
                  key=key, value=canonical, headers=headers)
    if primary is not None:
        produce_acked(producer, topic="transcripts.refined" if window is not None else "transcripts.final",
                      key=key, value=primary, headers=headers)


def consume(consumer, handler, *, stopped=lambda: False, topic=None):
    """Keep polling during inference; fence commits when assignment changes.

    One work item per process bounds GPU and memory use. All consumer operations
    stay on this thread. A failure exits without advancing an uncertain offset.
    """
    generation = 0
    active = None

    def assigned(client, partitions):
        nonlocal generation
        generation += 1
        client.assign(partitions)
        if active is not None:
            client.pause(partitions)

    def revoked(_client, _partitions):
        nonlocal generation
        generation += 1

    consumer.subscribe([topic or os.getenv("KAFKA_TOPIC_RECORDING_FINISHED", "recordings.finished")],
                       on_assign=assigned, on_revoke=revoked, on_lost=revoked)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="final-track") as pool:
        try:
            while not stopped():
                message = consumer.poll(0.1)
                if message is not None:
                    if message.error():
                        raise KafkaException(message.error())
                    if active is not None:
                        consumer.seek(TopicPartition(message.topic(), message.partition(), message.offset()))
                        consumer.pause(consumer.assignment())
                    else:
                        consumer.pause(consumer.assignment())
                        active = (pool.submit(handler, message), message, generation)
                if active is not None and active[0].done():
                    future, original, owned_generation = active
                    future.result()
                    if owned_generation == generation:
                        consumer.commit(message=original, asynchronous=False)
                    else:
                        logger.info("Assignment changed; leaving final track input for replay")
                    active = None
                    assignment = consumer.assignment()
                    if assignment:
                        consumer.resume(assignment)
        finally:
            consumer.close()


def main():
    """Run an allowlisted composite or explicitly enabled qualification provider."""
    setup_logging(default_level="INFO")
    stage = os.getenv("ASR_TRACK_STAGE", "final")
    if stage not in ("final", "refined"):
        raise ContractError("unsupported_stage")
    refined = stage == "refined"
    prefix = "REFINEMENT" if refined else "FINAL"
    track_id = require_name(os.getenv(f"{prefix}_TRACK_ID", "whisperx"))
    profile_id = require_name(os.getenv(f"{prefix}_PROFILE_ID", f"whisperx-medium-{'refined' if refined else 'final'}-r1"))
    if profile_id == f"whisperx-medium-{'refined' if refined else 'final'}-r1":
        from finalizer_worker.whisperx_track import WhisperXTrack
        provider = WhisperXTrack(profile_id=profile_id)
    elif profile_id == f"test-{'refined' if refined else 'final'}-r1" and os.getenv(f"{prefix}_TRACK_TEST_PROFILE_ENABLED") == "true":
        from finalizer_worker.test_track import TestTrack
        provider = TestTrack(profile_id=profile_id)
    else:
        raise ContractError("unsupported_profile")
    config = kafka_config()
    consumer = Consumer(dict(config, **{
        "group.id": f"{stage}-track.{track_id}.{profile_id}",
        "enable.auto.commit": False, "enable.auto.offset.store": False,
        "auto.offset.reset": "earliest", "max.poll.interval.ms": 300000,
        "max.partition.fetch.bytes": 1000000}))
    producer = Producer(dict(config, **{"enable.idempotence": True, "acks": "all",
                                        "delivery.timeout.ms": 25000}))
    store = S3Artifacts(s3_client(), os.environ["S3_BUCKET"],
                        recording_prefix="refinement-windows" if refined else os.getenv("FINAL_RECORDING_PREFIX", "recordings"),
                        result_prefix="refined-tracks" if refined else "final-tracks")
    logger.info("Starting final track track=%s profile=%s", track_id, profile_id)
    try:
        consume(consumer, lambda message: handle_message(
            message, producer=producer, track_id=track_id, profile_id=profile_id,
            provider=provider, store=store, stage=stage),
            topic=refinement_tracks.WINDOW_TOPIC if refined else None)
    except Exception as error:
        logger.error("Final track stopped without committing uncertain work kind=%s", type(error).__name__)
        raise SystemExit(1) from None
    finally:
        producer.flush(2)


if __name__ == "__main__":
    main()
