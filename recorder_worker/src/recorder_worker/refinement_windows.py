"""Reconstructible fixed refinement windows; one model-free Kafka consumer group.

Open sessions pin their first partition offset. Only acknowledged immutable
windows and a durable terminal boundary allow that offset to advance. A crash
or rebalance discards local buffers and reconstructs from retained audio.raw.
"""

import base64
import logging
import os
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from confluent_kafka import Consumer, Producer, KafkaException, TopicPartition
from proto_gen import stream_pb2 as pb
from drsynth_common.final_tracks import (
    ContractError, MAX_AUDIO_SECONDS, canonical_json, plan_headers, plan_proto, read_plan,
)
from drsynth_common.refinement_tracks import WINDOW_TOPIC, source_id, validate_window
from drsynth_common.final_track_artifacts import S3Artifacts, s3_client
from drsynth_common.kafka_ack import produce_acked
from drsynth_common.logging_setup import setup_logging
from drsynth_common.stream_controls import parse_stream_controls_from_kafka_headers
from finalizer_worker.track_worker import kafka_config

logger = logging.getLogger(__name__)


@dataclass
class Session:
    plan: dict
    partition: int
    first_offset: int
    lang: str
    origin: str
    headers: list
    closed: dict | None
    last_seq: int = 0
    total: int = 0
    start: int = 0
    pending: bytearray = field(default_factory=bytearray)
    last_activity: float = field(default_factory=time.monotonic)


class WindowProducer:
    """Bounded session assembly independent of the Kafka polling lifecycle."""

    def __init__(self, store, publish, max_sessions=32):
        self.store, self.publish = store, publish
        self.max_sessions = max_sessions
        self.sessions = {}

    def close_key(self, plan):
        return f"refinement-windows/{plan['tenant_id']}/{plan['session_id']}/{plan['plan_id']}.closed.json"

    def put_once(self, key, value):
        """Choose the first manifest without overwriting an accepted boundary."""
        from botocore.exceptions import ClientError
        try:
            self.store.client.put_object(Bucket=self.store.bucket, Key=key,
                                         Body=canonical_json(value), IfNoneMatch="*",
                                         ContentType="application/json")
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in ("PreconditionFailed", "412"):
                raise
        return self.store._read_json(key, 131072)

    def emit(self, state, count, reason):
        """Make the WAV and event durable, then publish the accepted event bytes."""
        window = pb.RefinementWindow(
            start_sample=state.start, end_sample=state.start + count,
            window_samples=state.plan["refinement_window_samples"],
            flush_reason=reason, bff_origin_uri=state.origin)
        identity = source_id(state.plan, window)
        with tempfile.TemporaryDirectory(prefix="refinement-window-") as directory:
            path = Path(directory) / "audio.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(bytes(state.pending[:count * 2]))
            source = self.store.put_audio(path, state.plan["tenant_id"], state.plan["session_id"], identity)
        window.recording.CopyFrom(pb.RecordingFinished(
            session_id=state.plan["session_id"], tenant_id=state.plan["tenant_id"],
            recording_url=source.storage_uri, duration_s=count / 16000, sample_rate=16000,
            lang=state.lang, source=source, final_plan=plan_proto(state.plan), created_at_ns=time.time_ns()))
        validate_window(window, state.headers)
        data = window.SerializeToString(deterministic=True)
        key = self.store.source_key(state.plan["tenant_id"], state.plan["session_id"], identity) + ".event.json"
        accepted = self.put_once(key, {"event": base64.b64encode(data).decode()})
        data = base64.b64decode(accepted["event"], validate=True)
        existing = pb.RefinementWindow.FromString(data)
        window.recording.created_at_ns = existing.recording.created_at_ns
        if window != existing:
            raise ContractError("window_replay_conflict")
        self.publish(state.plan["session_id"], data, state.headers)
        del state.pending[:count * 2]
        state.start += count

    def finish(self, key, reason):
        """Freeze the terminal sequence/sample boundary before publishing its tail."""
        state = self.sessions[key]
        boundary = {"last_seq": state.last_seq, "sample_count": state.total, "reason": reason}
        accepted = self.put_once(self.close_key(state.plan), boundary)
        if accepted != boundary:
            raise ContractError("closed_generation_conflict")
        if state.pending:
            self.emit(state, len(state.pending) // 2, reason)
        del self.sessions[key]

    def accept(self, message):
        """Validate selection, sequence and source continuity before buffering audio."""
        raw = message.value()
        if raw is None or len(raw) > 1048576:
            raise ContractError("invalid_audio_chunk_size")
        chunk = pb.AudioChunk.FromString(raw)
        headers = message.headers()
        plan = read_plan(headers, chunk.tenant_id, chunk.session_id)
        if plan is None or not plan.get("refinement_tracks"):
            return
        controls = parse_stream_controls_from_kafka_headers(headers)
        if not controls.want_refined or not controls.store_recording:
            raise ContractError("unsupported_retention_or_outputs")
        eof_headers = [v for k, v in headers if k == "x-audio-eof"]
        eof = eof_headers == [b"true"]
        if (message.key() != chunk.session_id.encode() or chunk.sample_rate != 16000
                or len(chunk.pcm16_le) % 2 or (eof_headers and not eof)
                or (eof and chunk.pcm16_le) or (not eof and not chunk.pcm16_le)):
            raise ContractError("invalid_audio_geometry_or_eof")
        key = (chunk.tenant_id, chunk.session_id)
        state = self.sessions.get(key)
        if state is None:
            if chunk.seq != 1 or len(self.sessions) >= self.max_sessions:
                raise ContractError("missing_session_start_or_capacity")
            close = self.store._read_json(self.close_key(plan), 1024)
            if close is not None and (set(close) != {"last_seq", "sample_count", "reason"}
                    or type(close["last_seq"]) is not int or close["last_seq"] < 1
                    or type(close["sample_count"]) is not int
                    or not 0 <= close["sample_count"] <= MAX_AUDIO_SECONDS * 16000
                    or close["reason"] not in ("eof", "idle")):
                raise ContractError("invalid_close_manifest")
            out_headers = plan_headers(plan) + [("x-outputs", b"refined"), ("x-store-recording", b"true")]
            out_headers.extend((k, v) for k, v in headers if k in ("traceparent", "tracestate"))
            state = Session(plan, message.partition(), message.offset(), chunk.lang,
                            chunk.bff_origin_uri, out_headers, close)
            self.sessions[key] = state
        if (state.plan != plan or state.partition != message.partition()
                or state.lang != chunk.lang or state.origin != chunk.bff_origin_uri
                or chunk.seq != state.last_seq + 1):
            raise ContractError("audio_continuity_mismatch")
        state.last_seq = chunk.seq
        state.total += len(chunk.pcm16_le) // 2
        if state.total > MAX_AUDIO_SECONDS * 16000:
            raise ContractError("recording_limit_exceeded")
        state.pending.extend(chunk.pcm16_le)
        state.last_activity = time.monotonic()
        size = plan["refinement_window_samples"]
        # Retain one complete window until the next sample or closure, so an
        # exact-multiple session still has a terminal window without a fake tail.
        while len(state.pending) > size * 2:
            self.emit(state, size, "slice")
        if state.closed is not None and chunk.seq == state.closed["last_seq"]:
            if state.total != state.closed["sample_count"] or eof != (state.closed["reason"] == "eof"):
                raise ContractError("closed_generation_conflict")
            self.finish(key, state.closed["reason"])
        elif eof:
            self.finish(key, "eof")

    def idle(self, seconds):
        """Close only live sessions; replay must reach its already frozen boundary."""
        for key, state in list(self.sessions.items()):
            if state.closed is None and time.monotonic() - state.last_activity >= seconds:
                self.finish(key, "idle")

    def frontier(self, partition, next_offset):
        """Do not commit past any open session's first message in this partition."""
        return min([next_offset, *(s.first_offset for s in self.sessions.values()
                                  if s.partition == partition)])


def main():
    """Poll and commit on one thread, dropping local ownership on rebalance."""
    setup_logging(default_level="INFO")
    config = kafka_config()
    consumer = Consumer(dict(config, **{
        "group.id": "refinement-window-producer", "enable.auto.commit": False,
        "enable.auto.offset.store": False, "auto.offset.reset": "earliest",
        "max.poll.interval.ms": 300000}))
    producer = Producer(dict(config, **{"enable.idempotence": True, "acks": "all",
                                       "delivery.timeout.ms": 25000}))
    store = S3Artifacts(s3_client(), os.environ["S3_BUCKET"], recording_prefix="refinement-windows")
    windows = WindowProducer(store, lambda key, value, headers: produce_acked(
        producer, topic=WINDOW_TOPIC, key=key.encode(), value=value, headers=headers))
    topic = os.getenv("KAFKA_TOPIC_AUDIO", "audio.raw")
    frontiers, committed = {}, {}

    def revoke(_client, partitions):
        revoked = {p.partition for p in partitions}
        windows.sessions = {k: s for k, s in windows.sessions.items() if s.partition not in revoked}
        for partition in revoked:
            frontiers.pop(partition, None)
            committed.pop(partition, None)

    consumer.subscribe([topic], on_revoke=revoke, on_lost=revoke)
    try:
        while True:
            message = consumer.poll(0.2)
            if message is not None:
                if message.error():
                    raise KafkaException(message.error())
                windows.accept(message)
                frontiers[message.partition()] = message.offset() + 1
            else:
                # Empty polls also occur while fetching; check lag before using
                # wall-clock idle, otherwise backlog could truncate a session.
                caught_up = all(consumer.get_watermark_offsets(p, timeout=5)[1] <= p.offset
                                for p in consumer.position(consumer.assignment()))
                if caught_up:
                    windows.idle(float(os.getenv("REFINEMENT_IDLE_SECONDS", "30")))
            offsets = [TopicPartition(topic, p, windows.frontier(p, end)) for p, end in frontiers.items()
                       if windows.frontier(p, end) != committed.get(p)]
            if offsets:
                consumer.commit(offsets=offsets, asynchronous=False)
                committed.update({p.partition: p.offset for p in offsets})
    except KeyboardInterrupt:
        pass
    except Exception as error:
        logger.error("Refinement window producer stopped without committing uncertain work kind=%s",
                     type(error).__name__)
        raise SystemExit(1) from None
    finally:
        consumer.close()
        producer.flush(2)


if __name__ == "__main__":
    main()
