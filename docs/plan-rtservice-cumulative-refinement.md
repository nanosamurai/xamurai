# Plan C: rtservice cumulative refinement (low-latency partials → stabilized final)

This document captures the implementation plan for **Plan C** discussed in the architecting exercise:

> rtservice has two dimensions:
> - timespan it runs on (5s currently)
> - latency: it starts after ~1.5s by default, but keeps refining that chunk until the 5s block is finished

Goal: make the realtime stream feel “live”, while still converging to a higher-quality transcript for the same time span.

## Current state (repo: xamurai)

- `rtservice` exposes gRPC `RealtimeASR.Stream(stream AudioChunk) -> stream AsrEvent`.
- `rtservice.engine.RealtimeEngine` emits replaceable **PARTIAL** events and
  word-owned **FINAL** events.
- Windowing defaults:
  - `window_sec = 5.0`
  - `overlap_sec = 0.5`
- The UI/BFF currently receives rtservice updates and also receives `RefinedEvent` (Kafka) from workers.

## Target behavior

For a given session:
- Start emitting **PARTIAL** updates after `emit_every_sec` (1.5s by default) from session start.
- Continue to emit updated PARTIALs as more audio arrives.
- Once `window_sec` plus the configured right context is available, emit a
  **FINAL** event for that committed time span. EOF supplies the terminal
  boundary when future context cannot arrive.
- Slide the window and repeat.

**Important:** the UI/BFF must treat PARTIALs as *replaceable* and FINALs as *locking*.

## Partial emission semantics (what "cumulative" means here)

The intended PARTIAL behavior is **cumulative-within-window**:

- For each realtime window (default **5s**), the server emits multiple `PARTIAL` hypotheses on a shorter cadence (default **1.5s**).
- Each subsequent `PARTIAL` is meant to **supersede** the previous one:
  - `start_s` stays constant (the current window start)
  - `end_s` grows as more audio arrives
  - `text` is the best current hypothesis for the span `[start_s..end_s]`
- PARTIAL attempts stop once the complete committed interval is buffered. The
  processor then waits for right context and runs the FINAL pass instead of
  decoding a redundant full-window PARTIAL that can block audio ingestion.
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
- `RT_EMIT_EVERY_SEC` (default `1.5`)
- `RT_PARTIAL_ENABLE` (default `true`)
- `RT_PARTIAL_MODE` (default `cumulative`) - `cumulative` produces monotonic-within-window PARTIALs; `tail` produces non-cumulative lookback PARTIALs.
- `RT_PARTIAL_MIN_BUFFER_SEC` (default `0.7`) - do not emit PARTIAL until this much audio accumulates.
- `RT_PARTIAL_MIN_TRANSCRIBE_SEC` (default `1.5`) - do not run ASR for PARTIAL until at least this much audio is available for the lookback chunk (reduces hallucinations).
- `RT_PARTIAL_STABILITY_REPEATS` (default `1`) - require the same hypothesis to repeat this many times before emitting (set to >1 for extra stability).
- `RT_PARTIAL_LOOKBACK_SEC` (default `2.0`) - only relevant when `RT_PARTIAL_MODE=tail`.

### Environment variables reference (single source of truth)

The following table documents every rtservice environment variable introduced
for Plan C partials. Runtime owners should pass these settings directly to the
service process. The public Community Edition stack provides a local example.

| Env var | Default | Meaning | Notes / gotchas |
|---|---:|---|---|
| `RT_PARTIAL_ENABLE` | `true` | Enable/disable PARTIAL emission entirely. | Use this as the “kill switch” during rollout. |
| `RT_EMIT_EVERY_SEC` | `1.5` | How often to *attempt* emitting a PARTIAL (cadence). | VAD, unchanged hypotheses, overload protection, the full-interval cutoff, and inference time can reduce the observable event rate. Smaller values amplify compute and should be clamped in BFF for untrusted clients. |
| `RT_WINDOW_SEC` | `5.0` | Duration owned by one FINAL commit interval. | Commit intervals are contiguous and non-overlapping. |
| `RT_OVERLAP_SEC` | `0.5` | Decoder context retained before and awaited after each commit interval. | A steady-state FINAL decode sees `window + 2 * overlap`; word midpoints decide ownership. |
| `RT_PARTIAL_MIN_BUFFER_SEC` | `0.7` | Minimum buffered audio before emitting any PARTIAL. | Avoids extremely-early hallucinations. |
| `RT_PARTIAL_MIN_TRANSCRIBE_SEC` | `1.5` | Minimum audio length we will actually run ASR on for PARTIAL. | Critical hallucination guard. In `tail` mode applies to the lookback chunk; in `cumulative` mode applies to the current window prefix. |
| `RT_PARTIAL_STABILITY_REPEATS` | `1` | Require the same hypothesis to repeat N times before emitting. | Set >1 for extra stability at cost of latency. |
| `RT_PARTIAL_MODE` | `cumulative` | `cumulative` = monotonic within window; `tail` = lookback-only partials. | `tail` is cheaper but produces “jumping” text that’s harder to merge. |
| `RT_PARTIAL_LOOKBACK_SEC` | `2.0` | Lookback duration for `RT_PARTIAL_MODE=tail`. | Ignored in `cumulative` mode. |
| `RT_PARTIAL_MAX_BEHIND_SEC` | `2.0` | Skip PARTIAL emissions when the stream is behind wall clock by more than this threshold. | FINALs still run; prevents runaway lag. |
| `RT_PARTIAL_IDLE_RESET_SEC` | `3.0` | If no chunks arrive for this long, treat it as a pause and reset lag baseline. | Avoids permanently suppressing PARTIALs after a pause. |

Per-stream overrides (gRPC metadata) currently supported:
- `x-rt-window-sec` → `RT_WINDOW_SEC`
- `x-rt-overlap-sec` → `RT_OVERLAP_SEC`
- `x-rt-emit-every-sec` → `RT_EMIT_EVERY_SEC`

Note: the remaining PARTIAL knobs are currently **process defaults** (env-driven), not per-stream.

Implementation status (in this repo):
- ✅ PARTIAL emission implemented (ASR-only partials)
- ✅ env knobs implemented and defaulted as above
- ✅ integration test added to assert PARTIAL precedes FINAL

Per-session overrides (optional for phase 1):
- gRPC metadata from `samuraibff` to `rtservice`:
  - `x-rt-window-sec: 5.0`
  - `x-rt-overlap-sec: 0.5`
  - `x-rt-emit-every-sec: 1.5`

Implementation status:
- ✅ `rtservice` reads these from gRPC invocation metadata per stream

### Phase 2 (proto changes)

Move config into `AudioChunk` (and/or introduce a dedicated `SessionConfig` message) once semantics stabilize.

## Replacement semantics (BFF/UI)

**Phase 1 recommended rule (simple):**

- Maintain a per-session “current partial line” and replace it on every PARTIAL.
- When FINAL arrives, append/commit it to the transcript and clear the partial line.

### Event identity / linking PARTIALs to FINALs

Currently `AsrEvent` does **not** include an explicit `segment_id` / `window_id` / `revision`.
The identity available to the client is:
- `session_id`
- `type` (PARTIAL/FINAL)
- `start_s`, `end_s`
- `speaker` (FINAL may have speaker; PARTIAL is currently speaker-less)

Practical UI/BFF logic (works today):

1) Treat PARTIALs as an ephemeral **single “live line” per session**.
   - In `RT_PARTIAL_MODE=cumulative`, PARTIALs have a stable `start_s` for the current window.
   - Replace the currently displayed partial on each new PARTIAL.

2) Clear the live PARTIAL line when you observe progress into the next window:
   - if the next PARTIAL has a larger `start_s` than the previous (window advanced), drop the old one.

3) Append FINALs in event order. Context is decoded more than once internally,
   but absolute word-midpoint ownership prevents repeated seam text.

Known limitation: FINALs are coalesced adjacent same-speaker word groups
(potentially multiple per commit interval) while PARTIALs are interval-prefix
hypotheses.
So a PARTIAL is not a strict “preview of exactly one FINAL segment”.

Follow-up (recommended): add `window_index` (and optionally `segment_id` + `revision`) into the proto,
so UI can link PARTIALs to a specific committed FINAL deterministically.

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

5. Delay FINAL emission for right context, retain word timestamps internally,
   own each word by a half-open absolute-sample midpoint interval, then join
   owned words to pyannote turns and coalesce adjacent equal speakers.

6. Keep session audio in a bounded PCM16 byte buffer. After each commit retain
   only the left context and uncommitted audio; no session-lifetime dedupe set is
   required. EOF commits every remaining interval and clears the buffer.

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

See also: `docs/rtservice-performance.md` for practical tuning guidance.
