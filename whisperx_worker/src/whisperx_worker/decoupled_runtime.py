"""Decoupled runtime for whisperx_worker.

Goal: keep Kafka polling responsive even when WhisperX inference is slow.

This module is intentionally small/isolated so we can evolve it without making
`whisperx_worker.py` (already large) even larger.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from queue import Empty, Queue
from typing import Callable, Deque, Dict, List, Optional, TypedDict

import numpy as np
from confluent_kafka import Consumer, KafkaException, TopicPartition

from drsynth_common.stream_controls import (
    parse_stream_controls_from_kafka_headers,
    parse_refinement_window_sec_from_kafka_headers,
)


KafkaHeader = tuple[str, Optional[bytes]]

logger = logging.getLogger(__name__)


class SliceJob(TypedDict):
    session_id: str
    tenant_id: Optional[str]
    lang: Optional[str]
    bff_origin_uri: Optional[str]
    trace_headers: Optional[list[KafkaHeader]]

    # Refinement window config for this job.
    window_sec: float

    pcm16: np.ndarray
    base_start_s: float
    slice_index: int
    flush_reason: str  # "slice" | "idle"

    topic: str
    partition: int
    offset: int


class CommitReq(TypedDict):
    topic: str
    partition: int
    offset: int  # next offset to consume


def _queue_get_many(q: "Queue[SliceJob]", *, max_items: int, max_wait_ms: int) -> List[SliceJob]:
    if max_items <= 1:
        return [q.get()]

    first = q.get()
    out: List[SliceJob] = [first]
    deadline = time.time() + (max_wait_ms / 1000.0)

    while len(out) < max_items:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            out.append(q.get(timeout=remaining))
        except Empty:
            break
    return out


def _cut_samples_from_session_buffer(
    session_id: str,
    *,
    need_samples: int,
    buffers: Dict[str, Deque[np.ndarray]],
    buf_samples: Dict[str, int],
) -> np.ndarray:
    parts: List[np.ndarray] = []
    need = need_samples
    while need > 0 and buffers[session_id]:
        ch = buffers[session_id][0]
        if ch.size <= need:
            parts.append(buffers[session_id].popleft())
            need -= ch.size
            buf_samples[session_id] -= ch.size
        else:
            parts.append(ch[:need])
            buffers[session_id][0] = ch[need:]
            buf_samples[session_id] -= need
            need = 0
    if not parts:
        return np.zeros((0,), dtype="<i2")
    return np.concatenate(parts).astype("<i2")


def run_decoupled(
    *,
    consumer: Consumer,
    topic_audio: str,
    slice_seconds: float,
    sample_rate: int,
    session_idle_sec: float,
    should_evict_idle_session,
    parse_audio_chunk: Callable[[bytes], object],
    build_job: Callable[..., SliceJob],
    run_inference_and_publish: Callable[[SliceJob], None],
) -> None:
    """Run the decoupled worker loop.

    Threading model:
    - Poll thread owns Kafka Consumer (thread-affinity).
    - Main thread owns inference + producer.
    - Commit-after-produce requests are executed by poll thread.
    """

    # env knobs
    commit_after_produce = os.getenv("WHISPERX_COMMIT_AFTER_PRODUCE", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    batch_max_items = int(os.getenv("WHISPERX_BATCH_MAX_ITEMS", "1"))
    batch_max_wait_ms = int(os.getenv("WHISPERX_BATCH_MAX_WAIT_MS", "30"))
    batch_mode = os.getenv("WHISPERX_BATCH_MODE", "sequential").strip().lower()
    ready_q_max = int(os.getenv("WHISPERX_READY_QUEUE_MAX", "256"))
    commit_q_max = int(os.getenv("WHISPERX_COMMIT_QUEUE_MAX", "1024"))

    ready_q: "Queue[SliceJob]" = Queue(maxsize=ready_q_max)
    commit_q: "Queue[CommitReq]" = Queue(maxsize=commit_q_max)
    stop_event = threading.Event()

    buffers: Dict[str, Deque[np.ndarray]] = defaultdict(deque)
    buf_samples: Dict[str, int] = defaultdict(int)
    slice_index: Dict[str, int] = defaultdict(int)
    last_activity: Dict[str, float] = defaultdict(lambda: 0.0)
    session_lang: Dict[str, Optional[str]] = defaultdict(lambda: None)
    session_bff_uri: Dict[str, Optional[str]] = defaultdict(lambda: None)
    session_tenant: Dict[str, Optional[str]] = defaultdict(lambda: None)
    session_headers: Dict[str, Optional[list[KafkaHeader]]] = defaultdict(lambda: None)

    # Per-session refinement window override.
    session_slice_seconds: Dict[str, float] = defaultdict(lambda: float(slice_seconds))
    session_slice_samples: Dict[str, int] = defaultdict(lambda: int(round(float(slice_seconds) * sample_rate)))

    pending_by_session: Dict[str, int] = defaultdict(int)
    state_lock = threading.Lock()
    last_poll_s: float = time.time()
    last_committed: Dict[tuple[str, int], int] = {}

    default_slice_samples = int(round(float(slice_seconds) * sample_rate))

    def poll_loop() -> None:
        nonlocal last_poll_s
        logger.info(
            "whisperx_worker decoupled poll loop started: commit_after_produce=%s ready_q_max=%d",
            commit_after_produce,
            ready_q_max,
        )

        while not stop_event.is_set():
            now_s = time.time()
            last_poll_s = now_s

            # Commit requests from inference thread.
            if commit_after_produce:
                while True:
                    try:
                        req = commit_q.get_nowait()
                    except Empty:
                        break
                    key = (req["topic"], req["partition"])
                    prev = last_committed.get(key, -1)
                    if req["offset"] <= prev:
                        continue
                    consumer.commit(
                        offsets=[TopicPartition(req["topic"], req["partition"], req["offset"])],
                        asynchronous=False,
                    )
                    last_committed[key] = req["offset"]

            # Idle eviction. Only when no pending jobs for session.
            idle_sessions = [
                sid
                for sid, _ts in list(last_activity.items())
                if should_evict_idle_session(
                    sid,
                    now_s=now_s,
                    last_activity_s=last_activity,
                    last_poll_s=last_poll_s,
                    idle_sec=session_idle_sec,
                )
            ]

            for sid in idle_sessions:
                with state_lock:
                    if pending_by_session.get(sid, 0) > 0:
                        continue
                if buf_samples.get(sid, 0) <= 0:
                    # nothing buffered, just evict state
                    buffers.pop(sid, None)
                    buf_samples.pop(sid, None)
                    slice_index.pop(sid, None)
                    last_activity.pop(sid, None)
                    session_lang.pop(sid, None)
                    session_bff_uri.pop(sid, None)
                    session_tenant.pop(sid, None)
                    session_headers.pop(sid, None)
                    session_slice_seconds.pop(sid, None)
                    session_slice_samples.pop(sid, None)
                    continue

                pcm = _cut_samples_from_session_buffer(
                    sid,
                    need_samples=buf_samples.get(sid, 0),
                    buffers=buffers,
                    buf_samples=buf_samples,
                )
                base_start = slice_index[sid] * float(session_slice_seconds.get(sid, float(slice_seconds)))
                slice_idx = slice_index[sid]
                slice_index[sid] += 1

                job: SliceJob = {
                    "session_id": sid,
                    "tenant_id": session_tenant.get(sid),
                    "lang": session_lang.get(sid),
                    "bff_origin_uri": session_bff_uri.get(sid),
                    "trace_headers": session_headers.get(sid),

                    "window_sec": float(session_slice_seconds.get(sid, float(slice_seconds))),
                    "pcm16": pcm,
                    "base_start_s": float(base_start),
                    "slice_index": int(slice_idx),
                    "flush_reason": "idle",
                    "topic": topic_audio,
                    "partition": 0,
                    "offset": -1,
                }

                with state_lock:
                    pending_by_session[sid] += 1
                ready_q.put(job)

                # Evict state after enqueue.
                buffers.pop(sid, None)
                buf_samples.pop(sid, None)
                slice_index.pop(sid, None)
                last_activity.pop(sid, None)
                session_lang.pop(sid, None)
                session_bff_uri.pop(sid, None)
                session_tenant.pop(sid, None)
                session_headers.pop(sid, None)
                session_slice_seconds.pop(sid, None)
                session_slice_samples.pop(sid, None)

            msg = consumer.poll(timeout=0.5)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            controls = parse_stream_controls_from_kafka_headers(msg.headers() or None)
            from drsynth_common.refinement_tracks import selected_audio
            if not controls.want_refined or selected_audio(msg.value(), msg.headers()):
                # Skip refined jobs entirely; still commit offsets so group progresses.
                if commit_after_produce:
                    consumer.commit(msg, asynchronous=False)
                else:
                    consumer.commit(msg, asynchronous=True)
                continue

            # Parse protobuf in caller to avoid dependency cycle.
            value = msg.value()
            headers = msg.headers() or None
            topic = msg.topic()
            partition = msg.partition()
            offset = msg.offset()

            audio = parse_audio_chunk(value)

            # AudioChunk-like interface (protobuf)
            sid = getattr(audio, "session_id")

            # Resolve per-session refinement window from headers.
            if sid not in session_slice_seconds:
                win_sec = parse_refinement_window_sec_from_kafka_headers(
                    headers,
                    default_sec=float(slice_seconds),
                )
                session_slice_seconds[sid] = float(win_sec)
                session_slice_samples[sid] = int(round(float(win_sec) * sample_rate))

            slice_samples = int(session_slice_samples.get(sid, default_slice_samples))
            lang = getattr(audio, "lang", "") or None
            bff_uri = getattr(audio, "bff_origin_uri", "") or None
            tenant = getattr(audio, "tenant_id", "") or None

            session_headers[sid] = headers or session_headers.get(sid)
            if lang:
                session_lang[sid] = lang
            if bff_uri:
                session_bff_uri[sid] = bff_uri
            if tenant:
                session_tenant[sid] = tenant
            last_activity[sid] = now_s

            pcm16_le = getattr(audio, "pcm16_le")
            arr = np.frombuffer(pcm16_le, dtype="<i2")
            buffers[sid].append(arr)
            buf_samples[sid] += arr.size

            while buf_samples[sid] >= slice_samples:
                pcm = _cut_samples_from_session_buffer(
                    sid,
                    need_samples=slice_samples,
                    buffers=buffers,
                    buf_samples=buf_samples,
                )
                # base_start must use the per-session window.
                base_start = slice_index[sid] * float(session_slice_seconds.get(sid, float(slice_seconds)))
                slice_idx = slice_index[sid]
                slice_index[sid] += 1

                job2 = build_job(
                    session_id=sid,
                    pcm16=pcm,
                    base_start_s=base_start,
                    slice_index=slice_idx,
                    flush_reason="slice",
                    window_sec=float(session_slice_seconds.get(sid, float(slice_seconds))),
                    lang=session_lang.get(sid),
                    bff_uri=session_bff_uri.get(sid),
                    tenant=session_tenant.get(sid),
                    trace_headers=session_headers.get(sid),
                    msg_topic=topic,
                    msg_partition=int(partition),
                    msg_offset=int(offset),
                )
                with state_lock:
                    pending_by_session[sid] += 1
                ready_q.put(job2)

            # at-most-once fallback
            if not commit_after_produce:
                consumer.commit(msg, asynchronous=True)

    poll_thread = threading.Thread(target=poll_loop, name="whisperx.poll", daemon=True)
    poll_thread.start()

    logger.info(
        "whisperx_worker decoupled mode enabled: batch_max_items=%d batch_max_wait_ms=%d batch_mode=%s",
        batch_max_items,
        batch_max_wait_ms,
        batch_mode,
    )

    try:
        while True:
            batch = _queue_get_many(ready_q, max_items=max(1, batch_max_items), max_wait_ms=batch_max_wait_ms)
            if batch_mode not in ("sequential", "none"):
                logger.warning("Unknown WHISPERX_BATCH_MODE=%s; using sequential", batch_mode)

            for job in batch:
                try:
                    run_inference_and_publish(job)
                finally:
                    with state_lock:
                        pending_by_session[job["session_id"]] = max(0, pending_by_session.get(job["session_id"], 1) - 1)

                if commit_after_produce and job["offset"] >= 0:
                    commit_q.put(
                        {
                            "topic": job["topic"],
                            "partition": job["partition"],
                            "offset": int(job["offset"]) + 1,
                        }
                    )
    finally:
        stop_event.set()
        try:
            poll_thread.join(timeout=2.0)
        except Exception:
            pass
