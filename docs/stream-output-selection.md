# Stream output selection (Phase 1)

We support multiple transcript “signal layers”:

- **realtime**: gRPC `rtservice` (`AsrEvent` PARTIAL + FINAL)
- **refined**: Kafka `transcripts.refined` (`RefinedEvent`, one per refinement window; `segments[]` carry speaker turns) from `whisperx_worker`
- **final**: Kafka `transcripts.final` (`SessionTranscript`) from `finalizer_worker`

Historically, every session produced all of them. This document describes the
Phase 1 stream-level configuration that lets the caller choose which layers
are produced.

Important properties:

- BFF saves controls before accepting audio and reuses them on reconnect;
  workers receive that snapshot through stream headers/metadata.
- Services must treat all client-provided values as **untrusted**; BFF should
  apply allowlists and quotas.

## Kafka headers (audio.raw)

The BFF produces `AudioChunk` messages to Kafka topic `audio.raw`. It may attach:

### `x-outputs`

CSV set selecting which outputs are enabled for this stream.

Example:

```
x-outputs: realtime,final
```

Recognized tokens:
- `realtime`
- `refined`
- `final`

Backwards compatibility:
- header **missing** ⇒ treat as `realtime,refined,final` (all enabled)
- header **present but empty** ⇒ treat as none (everything disabled)

Current behavior in xamurai services:
- `whisperx_worker` skips buffering/inference/produce when `refined` isn’t selected.
- `recorder_worker` skips recording and therefore prevents finalization when `final` isn’t selected.

### `x-final-tracks`

Ordered CSV track IDs, forwarded unchanged by the recorder to
`recordings.finished`. Omitted means `whisperx`; an explicitly empty header
selects none. BFF accepts a non-empty subset of its configured IDs and keeps
that selection fixed for the session. `x-outputs` still controls stage enablement.

Each finalizer has `FINALIZER_TRACK_ID` and a separate consumer group, defaulting
to `finalizer.<track_id>`. It commits skipped input for unselected tracks and
publishes selected results with `SessionTranscript.track_id` and the actual
`WHISPERX_MODEL` label in the `model` header. Replicas of one track share its group.
Transcript text/segments remain separate Postgres rows via Persistor; finalizers
no longer write local transcript JSON sidecars.

### `x-store-recording`

Boolean controlling whether the recording artifact (file:// WAV or s3:// object)
should be retained after transcription.

This is independent from `final`:

- `final=true` means we produce the final transcript
- `x-store-recording=false` means we delete the recording after successful finalization

Values:
- `true|false|1|0|yes|no|on|off`

Backwards compatibility:
- header missing ⇒ `true` (store)

Current behavior in xamurai services:
- `recorder_worker` propagates `x-store-recording` to `recordings.finished`.
- `finalizer_worker` deletes the recording (best-effort) after successful publish+commit when false.

Multiple final tracks require retention. BFF rejects multiple selected final
tracks with `store_recording=false` before audio starts; finalizers also reject
that unsafe combination. Single-track cleanup behavior is preserved.

### `x-refinement-window-sec`

Optional float controlling how often **whisperx_worker** runs refinement (slice duration).

Example:

```
x-refinement-window-sec: 20
```

Backwards compatibility:
- header missing/unparseable ⇒ worker uses env `WHISPERX_SLICE_SECONDS` (default 60s)

Notes:
- xamurai clamps the value to a safe range (currently `[10, 600]` seconds).
- Expected producer: `samuraibff` (from `/ws/audio` query param `refinement_window_sec`).

## gRPC metadata (rtservice)

When the BFF calls `RealtimeASR.Stream`, it may pass per-stream metadata:

- `x-rt-window-sec`
- `x-rt-overlap-sec`
- `x-rt-emit-every-sec`
- `x-rt-partial-enable: true|false`

`x-rt-partial-enable=false` disables PARTIAL emission for that stream (FINALs still emit).

## Security notes

- These knobs affect compute and storage cost.
- The BFF should clamp `x-outputs` to allowed combinations and enforce tenant quotas.
- The BFF should validate/limit any compute amplifiers (emit cadence, refinement windows, etc.).
