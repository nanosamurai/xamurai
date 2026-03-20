# Plan C: rtservice cumulative refinement (low-latency partials → stabilized final)

This document captures the implementation plan for **Plan C** discussed in the architecting exercise:

> rtservice has two dimensions:
> - timespan it runs on (5s currently)
> - latency: it starts already after ~0.7s, but keeps refining that chunk until the 5s block is finished

Goal: make the realtime stream feel “live”, while still converging to a higher-quality transcript for the same time span.

## Current state (repo: xamurai)

- `rtservice` exposes gRPC `RealtimeASR.Stream(stream AudioChunk) -> stream AsrEvent`.
- `rtservice.engine.RealtimeEngine` currently emits **FINAL** events only.
- Windowing defaults:
  - `window_sec = 5.0`
  - `overlap_sec = 0.5`
- The UI/BFF currently receives rtservice updates and also receives `RefinedEvent` (Kafka) from workers.

## Target behavior

For a given session:
- Start emitting **PARTIAL** updates after `emit_every_sec` (e.g. 0.7s) from session start.
- Continue to emit updated PARTIALs as more audio arrives.
- Once `window_sec` is complete (e.g. 5.0s), emit a **FINAL** event for that window/time span.
- Slide the window and repeat.

**Important:** the UI/BFF must treat PARTIALs as *replaceable* and FINALs as *locking*.

## Partial emission semantics (what "cumulative" means here)

The intended PARTIAL behavior is **cumulative-within-window**:

- For each realtime window (default **5s**), the server emits multiple `PARTIAL` hypotheses on a shorter cadence (default **0.7s**).
- Each subsequent `PARTIAL` is meant to **supersede** the previous one:
  - `start_s` stays constant (the current window start)
  - `end_s` grows as more audio arrives
  - `text` is the best current hypothesis for the span `[start_s..end_s]`
- When the `FINAL` arrives for that window, it becomes the source of truth and any earlier PARTIALs for that time span can be discarded.

This is the UI-friendly pattern:
```
PARTIAL: "hell"
PARTIAL: "hello howa"
PARTIAL: "hello how are you"
FINAL:   "hello how are you doing?"
```

## Configuration

### Phase 1 (no proto changes)

Use rtservice environment variables for defaults:

- `RT_WINDOW_SEC` (default `5.0`)
- `RT_OVERLAP_SEC` (default `0.5`)
- `RT_EMIT_EVERY_SEC` (default `0.7`)
- `RT_PARTIAL_ENABLE` (default `true`)
- `RT_PARTIAL_MODE` (default `cumulative`) - `cumulative` produces monotonic-within-window PARTIALs; `tail` produces non-cumulative lookback PARTIALs.
- `RT_PARTIAL_MIN_BUFFER_SEC` (default `0.7`) - do not emit PARTIAL until this much audio accumulates.
- `RT_PARTIAL_MIN_TRANSCRIBE_SEC` (default `1.5`) - do not run ASR for PARTIAL until at least this much audio is available for the lookback chunk (reduces hallucinations).
- `RT_PARTIAL_STABILITY_REPEATS` (default `1`) - require the same hypothesis to repeat this many times before emitting (set to >1 for extra stability).
- `RT_PARTIAL_LOOKBACK_SEC` (default `2.0`) - only relevant when `RT_PARTIAL_MODE=tail`.

Implementation status (in this repo):
- ✅ PARTIAL emission implemented (ASR-only partials)
- ✅ env knobs implemented and defaulted as above
- ✅ integration test added to assert PARTIAL precedes FINAL

Per-session overrides (optional for phase 1):
- gRPC metadata from `samuraibff` to `rtservice`:
  - `x-rt-window-sec: 5.0`
  - `x-rt-overlap-sec: 0.5`
  - `x-rt-emit-every-sec: 0.7`

Implementation status:
- ✅ `rtservice` reads these from gRPC invocation metadata per stream

### Phase 2 (proto changes)

Move config into `AudioChunk` (and/or introduce a dedicated `SessionConfig` message) once semantics stabilize.

## Replacement semantics (BFF/UI)

**Phase 1 recommended rule (simple):**

- Maintain a per-session “current partial line” and replace it on every PARTIAL.
- When FINAL arrives, append/commit it to the transcript and clear the partial line.

**More robust rule (if needed later):**

- Introduce a stable `segment_id` / `revision` so BFF can replace the correct segment deterministically.

## Implementation in this repo (xamurai)

### Engine changes

1. Extend `RealtimeConfig` with partial emission knobs:
   - `emit_every_sec`
   - `partial_enable`

2. Track additional per-session state:
   - next partial emission threshold (samples)
   - last partial emitted (to avoid spamming identical text)

3. Implement a “partial pass” that is cheaper than finalization:
   - For phase 1: ASR-only partials (speaker empty), FINAL keeps diarization+speaker mapping.

4. Emit PARTIAL AsrResults when conditions met (buffer >= emit_every_sec)
   - cap partial pass to the current window buffer.

5. Keep existing FINAL emission on full window.

### Tests

Update/increase coverage in `tests/test_realtime_asr_grpc.py`:

- Expect at least one PARTIAL quickly.
- Expect a FINAL after enough audio sent.

Current test added:
- `test_realtime_asr_stream_emits_partial_and_final`

## Other repos impacted

### samuraibff

- Needs to forward PARTIAL events to browser UI.
- Needs to implement client-facing merge semantics (replace partials; commit finals).
- Optionally pass rtservice config via gRPC metadata.

### UI/SDK

- Render “live partial line” and replace it.
- When FINAL arrives, commit it and remove partial.

### samuraipersistor

- No immediate change if rtservice events are not persisted.
- If persisted later, must add dedup keys/retention to avoid DB bloat.

## Security / abuse considerations

- PARTIAL emission frequency is a compute amplifier.
- Do not let arbitrary clients set `emit_every_sec` unbounded; validate in BFF.
- Consider per-tenant session quotas.
