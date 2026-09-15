"""Buffer audio per tenant/session while one inference thread processes windows.

The poll thread alone owns the consumer. Keep each active session's first offset
until its idle tail and all queued windows are acknowledged. Replays can then
reconstruct sample-relative timing without a checkpoint store or message changes.
"""

import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from confluent_kafka import KafkaError, KafkaException, TopicPartition

from drsynth_common.stream_controls import (
    parse_stream_controls_from_kafka_headers,
    parse_refinement_window_sec_from_kafka_headers,
)

logger = logging.getLogger(__name__)


def run_decoupled(*, consumer, topic_audio, slice_seconds, sample_rate,
                  session_idle_sec, track_id, parse_audio_chunk,
                  run_inference_and_publish):
    """Run the existing audio-to-refinement loop with replay-safe commits.

    The inference callback must return only after Kafka acknowledges its event.
    Exceptions stop this worker without committing its unfinished session audio.
    Revocation discards local buffers; the next owner rebuilds them from Kafka.
    """
    sessions = {}
    ready = deque()
    consumed = {}
    committed = {}
    caught_up = set()
    paused = set()
    queue_limit = max(1, int(os.getenv("WHISPERX_READY_QUEUE_MAX", "256")))
    running = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisperx.inference")

    def revoke(_consumer, partitions):
        """Invalidate revoked buffers and completions without advancing offsets."""
        revoked = {(p.topic, p.partition) for p in partitions}
        for key, state in list(sessions.items()):
            if state["partition"] in revoked:
                del sessions[key]
        for partition in revoked:
            consumed.pop(partition, None)
            committed.pop(partition, None)
        caught_up.difference_update(revoked)
        paused.difference_update(revoked)
        logger.info("Refinement partitions revoked: track=%s count=%d", track_id, len(revoked))

    def enqueue(state, size, reason):
        """Cut one sample-counted window and queue its existing inference payload."""
        parts = []
        remaining = size
        while remaining:
            chunk = state["buffer"][0]
            take = min(remaining, chunk.size)
            parts.append(chunk[:take])
            if take == chunk.size:
                state["buffer"].popleft()
            else:
                state["buffer"][0] = chunk[take:]
            remaining -= take
        job = dict(state["metadata"], pcm16=np.concatenate(parts),
                   base_start_s=state["samples"] / sample_rate,
                   slice_index=state["index"], flush_reason=reason)
        state["samples"] += size
        state["buffered"] -= size
        state["index"] += 1
        state["pending"] += 1
        ready.append((state, job))

    consumer.subscribe([topic_audio], on_revoke=revoke, on_lost=revoke)
    logger.info("Refinement worker started: track=%s", track_id)
    try:
        while True:
            if running and running[1].done():
                state, future = running
                # A revoked owner may finish publishing, but can never commit.
                future.result()
                state["pending"] -= 1
                running = None
            while ready and running is None:
                state, job = ready.popleft()
                if sessions.get(state["key"]) is state:
                    running = (state, executor.submit(run_inference_and_publish, job))

            now = time.monotonic()
            for key, state in list(sessions.items()):
                # Never call a backlog or a paused partition an idle session.
                if (state["partition"] in caught_up and state["partition"] not in paused
                        and now - state["last_audio"] >= session_idle_sec):
                    if state["buffered"] and len(ready) < queue_limit:
                        enqueue(state, state["buffered"], "idle")
                    if not state["buffered"] and not state["pending"]:
                        del sessions[key]

            # Pin each partition behind its earliest active session. In particular,
            # an unselected message or another session's completion cannot jump it.
            offsets = []
            for partition, end in consumed.items():
                safe = min((s["first_offset"] for s in sessions.values()
                            if s["partition"] == partition), default=end)
                if safe > committed.get(partition, -1):
                    offsets.append(TopicPartition(*partition, safe))
            if offsets:
                consumer.commit(offsets=offsets, asynchronous=False)
                committed.update({(p.topic, p.partition): p.offset for p in offsets})

            assigned = {(p.topic, p.partition) for p in consumer.assignment()}
            if len(ready) >= queue_limit:
                to_pause = assigned - paused
                if to_pause:
                    consumer.pause([TopicPartition(*p) for p in to_pause])
                    paused.update(to_pause)
            elif paused:
                consumer.resume([TopicPartition(*p) for p in paused & assigned])
                # Require a fresh EOF before an idle flush after backpressure.
                caught_up.difference_update(paused)
                paused.clear()

            msg = consumer.poll(0.1)
            if msg is None:
                continue
            partition = (msg.topic(), msg.partition())
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    caught_up.add(partition)
                    continue
                raise KafkaException(msg.error())
            caught_up.discard(partition)
            consumed[partition] = msg.offset() + 1
            headers = msg.headers() or None
            controls = parse_stream_controls_from_kafka_headers(headers)
            if not controls.want_refined or track_id not in controls.refinement_tracks:
                continue

            audio = parse_audio_chunk(msg.value())
            key = (audio.tenant_id, audio.session_id)
            if audio.sample_rate != sample_rate or len(audio.pcm16_le) % 2:
                raise ValueError("Refinement requires whole PCM16 samples at the configured sample rate")
            if not audio.pcm16_le:
                continue
            if key not in sessions:
                window = parse_refinement_window_sec_from_kafka_headers(headers, default_sec=slice_seconds)
                sessions[key] = {
                    "key": key, "partition": partition, "first_offset": msg.offset(),
                    "buffer": deque(), "buffered": 0, "samples": 0, "index": 0, "pending": 0,
                    "window_samples": max(1, round(window * sample_rate)),
                    "metadata": {"session_id": audio.session_id, "tenant_id": audio.tenant_id,
                                 "lang": audio.lang, "bff_origin_uri": audio.bff_origin_uri,
                                 "trace_headers": headers, "window_sec": window},
                }
            state = sessions[key]
            if state["partition"] != partition:
                raise ValueError("Session audio must retain its Kafka partition")
            state["last_audio"] = time.monotonic()
            chunk = np.frombuffer(audio.pcm16_le, dtype="<i2")
            state["buffer"].append(chunk)
            state["buffered"] += chunk.size
            while state["buffered"] >= state["window_samples"]:
                enqueue(state, state["window_samples"], "slice")
    finally:
        # Auto commit is disabled. Buffered/queued work must be replayed, not
        # flushed with newly invented boundaries while a consumer is closing.
        consumer.close()
        executor.shutdown(wait=True, cancel_futures=True)
