# WhisperX Worker Phase 2 (Decouple Kafka I/O + Inference) — Status / Handoff

Last updated: 2026-04-03

This document captures the current state of the WhisperX worker bugfix effort and the intended Phase 2 implementation.

## Problem Summary

### Symptom
In UI, refined transcript blocks appear “misplaced” / time resets (e.g. refined lines that should start around the 1-minute mark appear at the beginning because their timestamps restart near 0 seconds).

### Confirmed Root Cause
`whisperx_worker` evicts per-session slice state mid-session due to an “idle” heuristic based on wall-clock time.

When inference is slow (often CPU / no GPU), the Kafka consume loop is blocked long enough to exceed the idle threshold. The worker then:

1) logs session idle
2) flushes partial
3) evicts state including `slice_index[session_id]`
4) later continues consuming slices for the same `session_id` but with `slice_index` reset to `0`
5) refined timestamps restart from ~0, causing UI ordering issues

This is **not** caused by BFF/UI changes.

## Phase 1 Fix (Already Implemented)

Branch: `fix-whisperx-idle-eviction`

Commits:
- `5699bbf` — Fix idle eviction during blocked poll

### What Phase 1 does
Introduces a helper `_should_evict_idle_session(...)` that refuses to evict a session when the consumer poll loop itself has been blocked longer than the idle threshold.

In other words, “idle” is interpreted as:
- no new audio observed for `WHISPERX_IDLE_SECONDS`, **and**
- the consumer has been polling recently enough (so we are not merely behind)

### Tests
Unit tests were added to cover eviction decision logic.

## Phase 2 Goal (Planned)

Make the worker robust and performant when inference is slow by **decoupling Kafka polling/buffering from WhisperX inference**.

### Design Requirements
1) Kafka polling should continue even while inference is busy, so wall-clock “idle” doesn’t trigger incorrectly.
2) Prevent mid-session eviction while there is buffered audio and/or queued slices for that session.
3) Switch default Kafka delivery semantics to **commit-after-produce** (safer):
   - offsets should be committed only after refined events are produced (and ideally flushed)
   - at the cost of possible reprocessing after crash (at-least-once-ish)

## Planned Implementation Outline

### Runtime Mode Flag
Add an env flag:
- `WHISPERX_DECOUPLE_IO=true|false` (default `false` initially for safety)

When enabled:
- poll/buffering runs in a dedicated thread
- inference+produce runs in another thread (or main thread)

### Commit Semantics
Default to safer semantics:
- `WHISPERX_COMMIT_AFTER_PRODUCE=true` (default `true`)

Implementation approach:
- inference thread enqueues commit requests (topic/partition/offset+1)
- poll thread performs `consumer.commit(TopicPartition(...), asynchronous=False)`
  (keeps Consumer thread-affinity)

### Scheduler-Level “Dynamic Batching” knobs
(Even if underlying WhisperX backend doesn’t truly batch multi-audio in one GPU call, these knobs can still help with micro-batching scheduling.)

- `WHISPERX_BATCH_MAX_ITEMS` (default `1`)
- `WHISPERX_BATCH_MAX_WAIT_MS` (default `30`)
- `WHISPERX_BATCH_MODE` (default `sequential`)

### Additional operational knobs
- `WHISPERX_PRODUCER_FLUSH_S` (default `2.0`)
  - flush producer after producing refined events (best-effort reliability)

### Eviction logic in Phase 2
In decoupled mode, only evict a session when:
- it exceeds idle threshold, **and**
- there is **no buffered audio**, **and**
- there are **no queued/in-flight slice jobs** for that session

This avoids slice-index reset while the session is still being processed.

## Current Working Tree State (IMPORTANT)

Phase 2 is now partially implemented on branch `fix-whisperx-idle-eviction`.

Commits:
- `ba7c612` — Add optional decoupled Kafka poll loop for whisperx worker

### What was implemented
- New module: `whisperx_worker/src/whisperx_worker/decoupled_runtime.py`
  - Poll thread owns Kafka `Consumer`
  - Main thread runs inference/publish
  - Commit-after-produce via `TopicPartition` commits executed in poll thread
- `whisperx_worker.py` wiring:
  - `WHISPERX_DECOUPLE_IO` (default false)
  - `WHISPERX_COMMIT_AFTER_PRODUCE` (default true)
  - When `WHISPERX_DECOUPLE_IO=true`, worker runs `run_decoupled(...)` and returns.

## Cline / Tooling Stability Note

We observed repeated Cline instability and UI lockups due to large state payloads:

> `Large gRPC response: cline.StateService.subscribeToState size=4.2MB`

`apply_patch` returns full `final_file_content`, so editing a large file like `whisperx_worker.py` frequently can push responses into multi-MB.

### Recommended editing strategy for Phase 2
1) Keep `whisperx_worker.py` edits minimal.
2) Move new logic to small new module(s), e.g.:
   - `whisperx_worker/src/whisperx_worker/decoupled_runtime.py`
3) Only add a small wiring call in `whisperx_worker.py` to select mode.
4) Avoid non-ASCII characters in logs / source code.

## Helm / Compose follow-ups

Update charts/compose to expose new env vars:
- `WHISPERX_DECOUPLE_IO`
- `WHISPERX_COMMIT_AFTER_PRODUCE` (default true)
- `WHISPERX_BATCH_MAX_ITEMS`
- `WHISPERX_BATCH_MAX_WAIT_MS`
- `WHISPERX_BATCH_MODE`
- `WHISPERX_PRODUCER_FLUSH_S`

Also consider bumping:
- `WHISPERX_IDLE_SECONDS` to a safer value in non-GPU environments (deployment mitigation)

## Verification Checklist

Functional:
- refined timestamps for a continuous session must be monotonically increasing (no restarts near 0 after 1min+)
- no “idle flush & evict” while audio is still arriving / queued

Operational:
- worker remains responsive under slow inference
- Kafka consumer group lag should not spike purely due to inference blocking poll loop

### Test runs (evidence)

Ran integration suite in the documented conda env (`drsynth-whisperx`):

```bat
conda run -n drsynth-whisperx python -m pytest -q -m integration tests/test_whisperx_worker_integration.py
```

Result (2026-04-03):
- **3 passed** (with warnings from 3rd party libs: testcontainers/pyannote/torchaudio)
