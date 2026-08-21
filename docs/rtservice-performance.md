# rtservice performance & scalability notes

This document captures practical performance characteristics of **rtservice** (gRPC realtime ASR + diarization) and the main knobs that influence scalability.

## TL;DR recommendations (production-ish defaults)

### PARTIAL cadence (`RT_EMIT_EVERY_SEC`)

PARTIALs are a **compute amplifier**. Recommended starting points:

- **`RT_PARTIAL_MODE=cumulative`**:
  - start with **`RT_EMIT_EVERY_SEC=1.5`** (or even **2.0**) for medium/large models
  - use smaller values (e.g. 0.7) only if you’ve measured you can keep up in realtime (RTF < ~0.2) *and* you have a reason to spend the extra GPU budget

- **`RT_PARTIAL_MODE=tail`**:
  - you can usually afford a tighter cadence (e.g. **0.7–1.0**) because each PARTIAL transcribes only a small lookback chunk
  - tune lookback via `RT_PARTIAL_LOOKBACK_SEC` (typical 1–2s)

### Overload protection

Enable overload protection so long sessions do not drift further behind realtime:

- `RT_PARTIAL_MAX_BEHIND_SEC` (default **2.0s**)
  - if the stream is behind wall clock by more than this threshold, rtservice will **skip PARTIAL emissions** (FINALs still run)

## Why cumulative PARTIALs can create “progressive lag”

In `RT_PARTIAL_MODE=cumulative`, each PARTIAL re-runs ASR on the **entire prefix** of the current window.

Example (illustrative):

- `RT_WINDOW_SEC=10`
- `RT_EMIT_EVERY_SEC=0.7` → ~14 PARTIAL attempts per window

Average prefix duration is ~5s, so total audio transcribed per 10s window is roughly:

```
14 * 5s ≈ 70 seconds of audio transcribed
```

That’s ~**7× compute amplification** for the PARTIALs alone. If the GPU throughput cannot sustain that, the gRPC stream lags (server-side backlog).

## Implementation notes (what rtservice does)

### No temporary WAV files

rtservice runs faster-whisper on an **in-memory waveform** (numpy float32) to reduce latency and jitter.

### PARTIAL decode is cheaper than FINAL decode

rtservice uses two decode settings:

- **FINAL**: higher quality (beam search + word timestamps)
- **PARTIAL**: cheaper (lower beam, no word timestamps)

This is intentional: PARTIALs should be fast and replaceable; FINALs are the converged results.

### Repetition fallback

Both FINAL and PARTIAL decoding use faster-whisper's native compression-ratio
check and temperature fallback. Decoding starts deterministically at temperature
`0.0`. When faster-whisper classifies that decode as excessively repetitive, it
retries at the remaining configured temperatures instead of returning the first
failed decode immediately.

Normal decodes stop after the first successful attempt and therefore have no
additional inference cost. A pathological or otherwise failed decode can require
multiple attempts and temporarily increase latency. rtservice does not rewrite,
trim, or otherwise post-process repeated transcript text.

### FINAL diarization turn stabilization

Before FINAL ASR, rtservice normalizes and orders the diarization turns for the
current window. Consecutive turns with the same raw diarization speaker are
merged when their gap is at most `RT_FINAL_DIAR_MERGE_GAP_SEC`. Duration
filtering, overlapping-window ownership, speaker enrollment mapping, and ASR
then operate on the merged span. The waveform includes the short gaps between
the merged turns so faster-whisper receives useful speech context.

Merged turns shorter than `RT_FINAL_DIAR_MIN_TRANSCRIBE_SEC` are not sent to
ASR or speaker embedding. Different-speaker turns are never merged. If no
eligible diarization turn produces FINAL text, the existing full-window FINAL
fallback still runs. End-of-stream tail finalization is unchanged.

## Related environment variables

Existing Plan C variables (see `docs/plan-rtservice-cumulative-refinement.md`) still apply.

Additional perf/overload knobs:

| Env var | Default | Meaning |
|---|---:|---|
| `RT_ASR_TEMPERATURES` | `0.0,0.2,0.4,0.6,0.8,1.0` | Non-decreasing faster-whisper fallback schedule used by FINAL and PARTIAL decoding. |
| `RT_ASR_COMPRESSION_RATIO_THRESHOLD` | `2.4` | Faster-whisper threshold above which a decode is treated as too repetitive and retried. |
| `RT_FINAL_DIAR_MERGE_GAP_SEC` | `0.75` | Maximum gap between consecutive same-speaker diarization turns merged before FINAL ASR. Set to `0` to disable positive-gap merging. |
| `RT_FINAL_DIAR_MIN_TRANSCRIBE_SEC` | `0.7` | Minimum merged diarization duration sent to FINAL ASR and speaker mapping. Set to `0` to disable this additional guard; the internal FINAL safety floor still applies. |
| `RT_PARTIAL_MAX_BEHIND_SEC` | `2.0` | If wall clock minus audio time exceeds this, skip PARTIALs (FINALs still run). |
| `RT_PARTIAL_IDLE_RESET_SEC` | `3.0` | If no audio arrives for this many seconds, treat it as a pause and reset lag baseline. |

## What this does *not* solve (future work)

- Per-chunk `np.concatenate` buffer growth is still O(n) copying; a ring-buffer would reduce CPU overhead under high chunk rates.
- GPU scheduling/fairness across many concurrent sessions is still “best effort” in a single process.
- True multi-GPU scaling needs multiple rtservice instances and session
  stickiness at the load-balancing layer.

---

## Observability and crash diagnostics

In a containerized GPU load test, rtservice crashed with exit code 139
(SIGSEGV).
This is a **native crash** (likely a C/CUDA stack) and bypasses normal Python exception logs.

rtservice therefore supports the following observability knobs:

### Crash diagnostics (faulthandler)

Env vars:
- `RT_FAULTHANDLER_ENABLE` (default `true`)
  - enables `faulthandler.enable(all_threads=True)`
  - registers SIGUSR1 so an operator can request a stack dump from the process

### Prometheus metrics

rtservice can expose a Prometheus `/metrics` endpoint from the same process.

Env vars:
- `RT_METRICS_ENABLE` (default `true`)
- `RT_METRICS_PORT` (default `8008`)
- `RT_METRICS_ADDR`
  - recommended for local runs: `127.0.0.1` (security-first)
  - broader binds must be protected by the runtime network boundary

### Trace/log correlation

All Python services use `drsynth_common.logging_setup.setup_logging()`.
When OpenTelemetry is enabled, logs include:

- `trace_id` and `span_id` (from the current OTEL span)
- `session_id` and `session_trace_id` (`session_id` without dashes)

This makes it easy to jump from a Loki log line to a Tempo trace:
- copy `session_trace_id`
- in Grafana Tempo, use TraceID lookup
