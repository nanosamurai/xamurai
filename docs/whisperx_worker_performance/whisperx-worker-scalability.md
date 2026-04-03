# whisperx_worker scalability & performance (Phase 2)

This document explains why `whisperx_worker` needed a second runtime mode ("decoupled") and how it improves correctness and scalability when WhisperX inference is slow.

It complements:
- `docs/whisperx-worker-phase2-status.md` (status/handoff)
- `docs/plan-whisperx-worker-batching.md` (original Phase 2 design notes)

## Background: why the original design fails under load

Historically, `whisperx_worker` did everything in one thread:

1. poll Kafka (`audio.raw`)
2. buffer PCM per session
3. when a full slice is ready (`WHISPERX_SLICE_SECONDS`, default 60s), run WhisperX inference inline
4. produce refined events (`transcripts.refined`)

This has two severe problems when inference becomes slow (CPU-only nodes, overloaded GPU, diarization/enrollment enabled, etc.):

### 1) Correctness risk: wall-clock “idle” heuristic becomes wrong

If inference blocks the loop long enough, the worker stops calling `poll()`.
The session can then appear “idle” based on wall-clock time, even though the worker is simply behind.

When the idle eviction fires, the worker flushes and evicts per-session state (`slice_index`, buffers). When more audio is later processed for the same `session_id`, `slice_index` restarts from 0 and refined timestamps reset near 0.

This is the root cause of “refined blocks appear at 0min” even when the audio is at 1min+.

### 2) Kafka consumer health: long gaps between polls

Kafka consumer groups expect consumers to poll reasonably frequently.
If we stop polling for tens of seconds (or minutes), we risk:
- increased consumer lag
- rebalances / partition revocations
- hitting max poll interval (depending on broker + client configs)

## Phase 1 (correctness hotfix)

Phase 1 introduced `_should_evict_idle_session(...)` to prevent eviction while the poll loop itself is blocked longer than the idle threshold.

This mitigates timestamp resets without changing the threading model.

## Phase 2: decouple Kafka I/O from inference

Phase 2 adds an optional "decoupled" runtime:

- **Poll thread (Kafka I/O owner):**
  - owns the Kafka `Consumer` (thread-affinity)
  - continuously polls Kafka and appends audio to per-session buffers
  - cuts slices as soon as enough PCM arrives
  - enqueues independent slice jobs into a `ready_q`
  - performs offset commits requested by the inference loop

- **Inference loop (GPU/CPU compute owner):**
  - dequeues slice jobs
  - runs WhisperX inference and produces refined events
  - requests commits **only after** producing refined events (commit-after-produce)

### Enabling Phase 2

Set:

```bash
WHISPERX_DECOUPLE_IO=true
```

By default, Phase 2 remains opt-in to avoid risk during rollout.

## Offset commit semantics (commit-after-produce)

Environment variable:

```bash
WHISPERX_COMMIT_AFTER_PRODUCE=true  # default
```

### Why commit-after-produce

If the worker commits offsets immediately on consume, then crashes during inference/publish, we lose work.

Commit-after-produce flips the tradeoff:
- **Pros:** avoids message loss when the worker dies mid-processing
- **Cons:** may reprocess messages after crash (at-least-once-ish)

This is more appropriate for transcription pipelines where correctness is more important than avoiding duplicates.

## “Dynamic batching” knobs (current status)

The decoupled runtime contains a scheduler that can collect multiple ready jobs into a list before processing.

Environment variables:
- `WHISPERX_BATCH_MAX_ITEMS` (default 1)
- `WHISPERX_BATCH_MAX_WAIT_MS` (default 30)
- `WHISPERX_BATCH_MODE` (default `sequential`)

### Important: this is NOT true multi-audio GPU batching yet

At the moment, these knobs affect **how many jobs are dequeued together**, but inference still runs sequentially per job.

True multi-audio batching depends on the inference backend API (WhisperX / faster-whisper / CTranslate2). Once we confirm a safe multi-audio API, we can:
- feed a list of audios to a single batched inference call
- then split results back per job

Until then, this scheduler-level batching is mainly groundwork to evolve towards real batching and to reduce per-job queue overhead.

## Refactoring: splitting the large file

To keep `whisperx_worker.py` maintainable and to avoid tool/UI instability when editing large files, Phase 2 introduced:

- `whisperx_worker/src/whisperx_worker/decoupled_runtime.py`

The top-level module `whisperx_worker.py` keeps:
- WhisperX inference functions
- protobuf construction
- end-to-end logic for publishing refined events
- and only small wiring code for selecting runtime mode

## Operational guidance

### Recommended production defaults

If you have GPU:
- keep `WHISPERX_DECOUPLE_IO=true`
- keep `WHISPERX_COMMIT_AFTER_PRODUCE=true`
- keep `WHISPERX_IDLE_SECONDS` safely above worst-case slice processing time (or rely on decoupled mode’s additional protections)

If you are CPU-only:
- decoupled mode still helps Kafka health
- but you likely must increase `WHISPERX_IDLE_SECONDS` to avoid frequent idle flushes

## Security / safety notes

- No ports are exposed by this worker.
- Commit-after-produce can increase duplicate events; downstream consumers must be idempotent or tolerate duplicates.
