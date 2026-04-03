# Plan: whisperx_worker decouple Kafka I/O from inference + dynamic batching

This document scopes **Phase 2** follow-up work for `whisperx_worker`.

## Implementation status (as of 2026-04-03)

This plan started as a design document. The following section is an explicit
mapping to what is already implemented on branch `fix-whisperx-idle-eviction`.

Related docs:
- `docs/whisperx-worker-phase2-status.md` (status + evidence of test runs)
- `docs/whisperx-worker-scalability.md` (long-form rationale + operational notes)

### Implemented

#### Consume/infer decoupling (rollout step 1)

- ✅ Added an opt-in decoupled runtime in:
  - `whisperx_worker/src/whisperx_worker/decoupled_runtime.py`
- ✅ Wiring in `whisperx_worker.py` behind:
  - `WHISPERX_DECOUPLE_IO` (default `false`)

Runtime model:
- poll thread owns Kafka `Consumer` and continuously polls/buffers/cuts slices
- main thread runs inference+publish for slice jobs

This addresses the original correctness issue (slow inference blocking `poll()`),
and it keeps Kafka consumer health stable under load.

#### Job model

- ✅ Implemented internal job object `SliceJob` (TypedDict) including:
  - session metadata (tenant/lang/bff uri/trace headers)
  - audio payload (`pcm16`, `base_start_s`, `slice_index`)
  - delivery metadata (`topic`, `partition`, `offset`)

Note: field names differ slightly from the plan (`topic/partition/offset` instead
of `kafka_topic/kafka_partition/kafka_offset`), but semantics are the same.

#### Offset commit semantics (commit-after-produce)

- ✅ Implemented `WHISPERX_COMMIT_AFTER_PRODUCE` (default `true`).
- ✅ In decoupled mode, commits are executed in the poll thread via
  `TopicPartition` commits (consumer thread-affinity is respected).

Trade-off is as documented: safer semantics (less chance of losing work) at the
cost of possible reprocessing after crash.

#### Scheduler-level batch collection (groundwork)

- ✅ Added env knobs:
  - `WHISPERX_BATCH_MAX_ITEMS`
  - `WHISPERX_BATCH_MAX_WAIT_MS`
  - `WHISPERX_BATCH_MODE`
- ✅ Implemented `_queue_get_many()` which can collect multiple ready jobs into a
  list before processing.

Important: this is only the *scheduler* foundation. Inference is still executed
sequentially per job.

### Not implemented yet (still part of the plan)

#### True multi-audio GPU batching

- ⛔ No duration bucketization.
- ⛔ No language bucketization.
- ⛔ No single-call list-of-audios batched inference.

Reason: this requires confirming the exact WhisperX backend API we use in our
runtime (and validating accuracy/latency trade-offs). Until then, we keep the
logic correct and safe, and treat the current knobs as groundwork.

#### Publisher flush knob

- ⛔ `WHISPERX_PRODUCER_FLUSH_S` is not implemented yet.

#### Observability spans for batching

- ⛔ No new OTEL spans/attributes for batch-level metrics yet.

## Rollout checklist (recommended)

1) ✅ Decouple consume/infer with batch size 1 (correctness + Kafka health)
2) ⛔ Add true dynamic batching (multi-audio inference) behind env flags
3) ⛔ Measure throughput + P95 latency; tune `max_items` and `max_wait_ms`

Motivation:
- Fix correctness: do not treat sessions as idle purely because inference blocks Kafka polling.
- Improve throughput: enable **dynamic batching** of independent slices across sessions on GPU.

Non-goals (for Phase 2):
- No new Kafka topics.
- No mixing audio across sessions (no concatenation/stitching).
- Do not attempt “more GPU throughput” by running multiple model copies in parallel.

## Current problem

Today, `whisperx_worker` is effectively single-threaded:
- Poll Kafka
- Buffer PCM
- When a 60s slice is ready, run WhisperX inline
- Produce refined events

On slow inference (CPU or overloaded GPU), the loop doesn’t call `poll()` for tens of seconds.
This causes:
- session idle heuristics to become incorrect
- consumer group instability risk (max poll interval)
- poor scalability under multiple concurrent sessions

## Desired pipeline (high level)

Split the worker into three responsibilities:

1) **Kafka I/O thread** (fast loop)
   - Poll Kafka frequently and append to per-session buffers.
   - Cut slices when enough audio accumulates.
   - Enqueue slice jobs into a shared `ready_queue`.
   - Maintain `last_activity` per session based on observed input.

2) **Batcher / GPU owner loop** (single inference worker)
   - Pull many ready slices from the queue.
   - Group into a batch (same model, similar duration, optionally same language).
   - Run *one* batched inference call.
   - Route results back to their originating sessions.

3) **Publisher / offset-commit stage**
   - Publish `RefinedEvent`s.
   - Commit Kafka offsets **only after** successful end-to-end handling.

This matches the “threads for Kafka I/O, not for parallel GPU inference” guidance.

## Job model

Introduce an internal job object (pure Python, no proto changes):

```py
class SliceJob(TypedDict):
    session_id: str
    tenant_id: str | None
    lang: str | None
    bff_origin_uri: str | None
    trace_headers: list[KafkaHeader] | None

    # audio payload
    pcm16: np.ndarray  # shape (n,) dtype <i2, mono 16k
    base_start_s: float
    slice_index: int

    # delivery semantics
    kafka_topic: str
    kafka_partition: int
    kafka_offset: int
```

We store (topic, partition, offset) so commits can be deferred until done.

## Dynamic batching

### Why batching

With Whisper-style models, throughput often improves by increasing batch size.
This keeps **one model copy** in VRAM while transcribing multiple independent audio items.

### Batching constraints

- Never concatenate different sessions into one audio stream.
- Batch at the **input list/tensor** level.
- Avoid mixing very different durations in the same batch (padding waste).

### Simple dynamic batcher algorithm

Parameters:
- `WHISPERX_BATCH_MAX_ITEMS` (e.g. 8)
- `WHISPERX_BATCH_MAX_WAIT_MS` (e.g. 30)
- `WHISPERX_BATCH_BUCKETS` (optional duration buckets)

Pseudo:

```py
while True:
    job = ready_queue.get()  # blocks
    put_into_duration_bucket(job)

    batch = collect_up_to_n(
        prefer_fullest_bucket=True,
        max_items=N,
        max_wait_ms=W,
    )

    audios = [to_float32(job["pcm16"]) for job in batch]
    results = batched_transcribe(audios, langs=[job["lang"] for job in batch])
    for job, result in zip(batch, results):
        publish_refined(job, result)
        mark_done(job)  # enabling commit
```

### Does WhisperX expose multi-audio batching?

We need to verify which WhisperX backend we run in our environment:
- WhisperX commonly uses faster-whisper (CTranslate2) for transcription.
- faster-whisper supports batching via `batch_size` for segments/chunking and has
  ongoing support for batching multiple audio inputs.

Implementation options:
1) **If WhisperX exposes an API to transcribe a list of audios**: use that.
2) If not: implement a thin “batched transcribe” wrapper around faster-whisper
   directly for this worker (still emitting the same `RefinedEvent` proto).

## Session state + eviction in Phase 2

Idle eviction should depend on:
- time since last input (`last_activity`)
- buffer empty
- no queued jobs for the session
- no in-flight jobs for the session

Keep `WHISPERX_IDLE_SECONDS=30` default for UX (don’t hold last chunk hostage).

## Offset commit semantics

Because we decouple polling from processing, we must not auto-commit offsets.

Recommended approach:
- consumer uses `enable.auto.commit=false`
- on consume:
  - append to buffer
  - when a slice is cut, enqueue SliceJob with the offset of the message that
    completed the slice
- after publishing refined events:
  - commit the job’s offset (per partition)

Note: because `audio.raw` ordering per session relies on key partitioning, this
should remain safe as long as a given session_id stays on one partition.

## Observability

Add spans / attributes:
- `whisperx.batch` span with:
  - `batch_size`
  - `batch_wait_ms`
  - `dur_bucket`
- per-job attributes:
  - `session_id`, `tenant_id`, `slice_index`, `slice_start_s`

## Risks

- Batching increases VRAM use; needs tuning per model size.
- Committing offsets too early loses work; too late increases reprocessing after crash.
- Mixing different languages in one batch may reduce accuracy if WhisperX uses
  language-conditioned decoding; we may want to bucket by language too.

## Suggested rollout

1) Implement consume/infer decoupling with **batch size 1** (already fixes polling).
2) Add dynamic batching behind env flags.
3) Measure throughput + P95 latency; tune `max_items` and `max_wait_ms`.
