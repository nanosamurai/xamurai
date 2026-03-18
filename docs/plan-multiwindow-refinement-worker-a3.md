# Plan A3: multi-window refinement inside a single worker (per session)

This document captures the implementation plan for **Plan A3** discussed in the architecting exercise:

> Have arbitrary number of workers per session, configurable by session config. User picks at session start how often refinement runs (e.g. 10s/30s/60s chunks).

**Key constraint (per discussion):** do **not** implement “fan-out by window size” via Kafka topics. We keep fan-out as a future mechanism for **model families** (WhisperX vs Voxstral, etc.).

## Current state (repo: xamurai)

- `audio.raw` topic carries `AudioChunk { session_id, seq, pcm16_le, lang, tenant_id, bff_origin_uri }`.
- `whisperx_worker` consumes `audio.raw`, buffers per session, and emits refined segments every `WHISPERX_SLICE_SECONDS` to `transcripts.refined`.
- Refined output is `RefinedEvent` (proto) with timestamped segments.

## Target behavior

For each session, enable **0..N refinement windows** (e.g. 10/30/60) as chosen by user at session start.

Within a single worker process:
- Consume `audio.raw` once.
- Maintain per-session canonical PCM buffer.
- Maintain per-session slicer cursors for each enabled window.
- Enqueue slice jobs for enabled windows when enough samples are accumulated.
- Run WhisperX on each slice and emit refined segments to `transcripts.refined`.

## Session configuration propagation (phase 1)

Use Kafka headers on `audio.raw` (produced by `samuraibff`):

- `x-refine-model: whisperx`  (future-ready)
- `x-refine-windows-sec: 10,30,60` (CSV)
- optional guardrails:
  - `x-refine-max-windows: 3`
  - `x-refine-max-window-sec: 60`

Workers should treat these headers as **untrusted** and apply server-side allowlists/limits.

## Output metadata (required for “longest wins”)

The BFF/UI must be able to tell which refinement layer produced a refined segment.

Phase 1 (no proto change): produce Kafka headers alongside `RefinedEvent`:

- `x-refinement-model: whisperx`
- `x-refinement-window-sec: 10|30|60`

Phase 2 (proto v3): add a typed field to `RefinedEvent`:

```proto
message RefinementInfo {
  string model = 1;              // "whisperx", "voxstral", ...
  double window_sec = 2;         // 10, 30, 60
  uint32 revision = 3;           // optional monotonic revision
}

message RefinedEvent {
  ...existing...
  RefinementInfo refinement = 10;
}
```

## Worker internal design

### State model

- `SessionState` (per session):
  - canonical ring buffer (PCM samples)
  - last activity timestamp
  - `slicers: Dict[window_sec, SlicerState]`

- `SlicerState` (per window):
  - cursor index (how much audio already assigned)
  - slice_index (for absolute base_start)

### Scheduling / fairness

Since GPU is shared, a per-process job queue is needed:

- job: `(session_id, tenant_id, window_sec, pcm_slice, base_start, lang)`
- priorities: smaller windows first (10 > 30 > 60) to preserve responsiveness.
- overload policy: drop/delay the largest windows first.

### Memory bounding

- Keep only up to `max_window_sec + overlap` in memory per session.
- Evict idle sessions after `WHISPERX_IDLE_SECONDS` (already exists).

## Kubernetes / infra implications

- Still a single Deployment (`whisperx-worker`), but now it can do multiple windows per session.
- Scale horizontally by increasing replica count.
- Consider:
  - GPU requests/limits per pod
  - HPA based on GPU utilization (future)

## Other repos impacted

### samuraibff

- Session creation flow must accept user’s desired refinement windows.
- When producing `audio.raw`, add `x-refine-windows-sec` headers.
- When consuming refined callbacks/events, merge segments using “longest wins” semantics.

### UI/SDK

- Allow selecting refinement profile at session start.
- Merge refined layers (prefer longer window segments for overlapping times).

### samuraipersistor

- If refined segments are stored, window/model metadata should be stored too.
- If metadata remains header-only, persistor must capture headers; proto v3 is cleaner.

## Security considerations

- Client-driven window config can amplify compute.
- BFF must enforce allowlisted profiles (e.g. {10}, {30}, {60}, {10,60}).
- Consider tenant quotas.
