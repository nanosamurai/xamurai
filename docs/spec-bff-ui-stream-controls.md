# Spec: BFF + UI stream controls (outputs + realtime knobs)

This document specifies the **samuraibff + UI** changes needed to support
per-stream configuration of:

- which transcript layers are produced (realtime/refined/final)
- whether to store the recording artifact
- existing realtime stream knobs (window, overlap, partial cadence, etc.)

Xamurai services (this repo) already implement the corresponding controls.

## Goals

1) Let end users choose to receive **only**:
   - realtime (rtservice gRPC)
   - refined (Kafka transcripts.refined)
   - final (Kafka transcripts.final)
   - or any combination

2) Make **recording retention** an independent dimension:
   - produce final transcript, but optionally delete the recording after transcription

3) Keep the **stream as source of truth** (no worker DB lookups), while also
   persisting the chosen settings in the DB for UI “session detail” replay.

4) Assume settings do **not** change mid-stream.

## Non-goals

- No attempt to retrofit DB/session config fetching into xamurai workers.
- No attempt to merge refined layers (multi-window refinement) here.

## Terminology

- **realtime**: gRPC stream `RealtimeASR.Stream(...) -> AsrEvent` from `rtservice`
- **refined**: Kafka topic `transcripts.refined` (`RefinedEvent`) from `whisperx_worker`
- **final**: Kafka topic `transcripts.final` (`SessionTranscript`) from `finalizer_worker`
- **recording**: WAV on disk (`file://...`) or object storage (`s3://...`) created by `recorder_worker`

## Control model

### Output selection

Represent as a set of 3 booleans or an enum-set:

- `want_realtime`
- `want_refined`
- `want_final`

Notes:
- `want_realtime` affects **only** BFF <-> rtservice behavior and what the UI subscribes to.
- `want_refined` / `want_final` affect Kafka pipeline cost.

### Recording retention

Independent boolean:

- `store_recording`

Semantics:
- If `want_final=false` then recording isn’t required (and recorder is gated off).
- If `want_final=true` then recorder must record (at least until finalizer consumes it).
- If `store_recording=false` then after final transcript is published, the recording is deleted (best-effort).

## Transport: BFF -> xamurai

### 1) Kafka headers on `audio.raw`

When BFF produces `AudioChunk` to Kafka (`audio.raw`), attach headers:

#### `x-outputs`

```
x-outputs: realtime,refined,final
```

Allowed tokens: `realtime`, `refined`, `final`.

Backwards compatibility:
- if header is missing, xamurai assumes **all outputs enabled**.

Important: `realtime` token has **no direct effect** on Kafka pipeline (rtservice does not consume Kafka).
It is included for a unified “stream config snapshot” and future-proofing, but only `refined` and `final`
are acted on by xamurai Kafka workers.

#### `x-store-recording`

```
x-store-recording: true|false
```

If omitted: defaults to `true`.

### 2) gRPC metadata on rtservice stream

When BFF starts the rtservice bidirectional gRPC stream, attach metadata:

- existing:
  - `x-rt-window-sec: <float>`
  - `x-rt-overlap-sec: <float>`
  - `x-rt-emit-every-sec: <float>`
- new:
  - `x-rt-partial-enable: true|false`

Semantics:
- `x-rt-partial-enable=false` disables PARTIAL emission completely (FINALs still emit).

### Why we need `x-rt-partial-enable` (and why emit_every=0 isn’t enough)

In `rtservice.engine.RealtimeConfig.emit_every_samples` we clamp the effective cadence to a minimum:

```python
sec = max(0.05, float(self.emit_every_sec))
```

and in `_effective_cfg()` we ignore non-finite / <=0 values by falling back to defaults.

So setting `x-rt-emit-every-sec=0` does **not** disable partials; it results in a small cadence
or a default fallback. Disabling partials requires a separate boolean knob.

## BFF behavior

### Stream start / handshake

At stream start, BFF should collect from UI/SDK a `StreamControls` payload containing:

- outputs:
  - realtime (bool)
  - refined (bool)
  - final (bool)
- recording:
  - store_recording (bool)
- realtime knobs:
  - rt_window_sec (float)
  - rt_overlap_sec (float)
  - rt_emit_every_sec (float)
  - rt_partial_enable (bool)

Then:

1) Persist this selection in DB associated with the session (for UI history).
2) Use it as source of truth for:
   - what headers BFF puts on `audio.raw`
   - what metadata BFF puts on rtservice gRPC stream
   - what event subscriptions BFF activates (refined/final consumption)

### Kafka production (audio.raw)

When emitting each AudioChunk:

- Always include `bff_origin_uri` and `tenant_id` as today.
- Add headers:
  - `x-outputs`
  - `x-store-recording`

Recommended: compute header bytes once per stream and reuse.

### Subscriptions / fan-out

Depending on outputs:

- `want_realtime=false`:
  - BFF **does not** call rtservice.
  - UI still streams audio to BFF and BFF still publishes to Kafka if refined/final enabled.

- `want_refined=false`:
  - BFF does not need refined callbacks/subscriptions.
  - whisperx_worker will skip work due to `x-outputs` gating.

- `want_final=false`:
  - BFF does not need transcripts.final consumption.
  - recorder_worker will skip recording due to `x-outputs` gating.

### Validation / security policy

BFF must treat UI/SDK inputs as untrusted:

- clamp allowed `x-outputs` combinations per tenant/plan
- clamp realtime knobs:
  - min/max rt_window_sec
  - overlap < window
  - emit_every_sec >= some minimum when partial enabled
- enforce quotas:
  - max concurrent sessions per tenant
  - max GPU-heavy workers per tenant

## UI changes

### Session start settings panel

Add a “Transcription outputs” section:

- Checkboxes:
  - Realtime (live)
  - Refined (near-RT)
  - Final (post-session)

Add “Recording retention”:

- Radio/select:
  - Store recording
  - Delete recording after transcription

And keep/extend existing realtime settings (already exposed in UI):

- Realtime window size (sec)
- Realtime overlap (sec)
- Emit partials (on/off)
- Emit every (sec)

UI constraints:
- If Realtime unchecked, realtime settings may be disabled/hidden.
- If Final unchecked, “Recording retention” can be hidden or forced to “delete” (no recording should exist anyway).

### Persist + display

Store the chosen settings with the session in DB so the UI can show:
- “this session was realtime-only”
- “recording was deleted”
- which realtime knobs were used

## Example configurations

1) **Realtime only** (no Kafka compute)
- want_realtime=true, want_refined=false, want_final=false
- BFF calls rtservice only
- BFF publishes audio.raw with `x-outputs: realtime` (optional but consistent)

2) **Final transcript only, delete recording**
- want_realtime=false, want_refined=false, want_final=true, store_recording=false
- BFF publishes audio.raw headers:
  - `x-outputs: final`
  - `x-store-recording: false`

3) **Realtime + refined + final**
- `x-outputs: realtime,refined,final`
- store_recording true/false depending on retention

## Implementation notes for samuraibff repo

- Kafka producer: set headers on `audio.raw` message publish.
- rtservice gRPC client: pass metadata (new key `x-rt-partial-enable`).
- DB schema: add a JSON blob / columns on session record for:
  - outputs set
  - store_recording
  - realtime knobs
