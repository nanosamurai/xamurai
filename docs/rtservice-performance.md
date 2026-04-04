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

## Related environment variables

Existing Plan C variables (see `docs/plan-rtservice-cumulative-refinement.md`) still apply.

Additional perf/overload knobs:

| Env var | Default | Meaning |
|---|---:|---|
| `RT_PARTIAL_MAX_BEHIND_SEC` | `2.0` | If wall clock minus audio time exceeds this, skip PARTIALs (FINALs still run). |
| `RT_PARTIAL_IDLE_RESET_SEC` | `3.0` | If no audio arrives for this many seconds, treat it as a pause and reset lag baseline. |

## What this does *not* solve (future work)

- Per-chunk `np.concatenate` buffer growth is still O(n) copying; a ring-buffer would reduce CPU overhead under high chunk rates.
- GPU scheduling/fairness across many concurrent sessions is still “best effort” in a single process.
- True multi-GPU scaling needs multiple rtservice pods and session stickiness at the LB layer.

---

## Observability / crash diagnostics (EKS dev)

In EKS dev we observed rtservice crashing under load with exit code 139 (SIGSEGV).
This is a **native crash** (likely a C/CUDA stack) and bypasses normal Python exception logs.

rtservice therefore supports the following observability knobs:

### Crash diagnostics (faulthandler)

Env vars:
- `RT_FAULTHANDLER_ENABLE` (default `true`)
  - enables `faulthandler.enable(all_threads=True)`
  - registers SIGUSR1 so you can dump stacks on demand:
    - `kubectl exec -it <pod> -- kill -USR1 1`

### Prometheus metrics

rtservice can expose a Prometheus `/metrics` endpoint from the same process.

Env vars:
- `RT_METRICS_ENABLE` (default `true`)
- `RT_METRICS_PORT` (default `8008`)
- `RT_METRICS_ADDR` (default `127.0.0.1` for security)

In Kubernetes, prefer accessing this via `kubectl port-forward`.

### Trace/log correlation

All Python services use `drsynth_common.logging_setup.setup_logging()`.
When OpenTelemetry is enabled, logs include:

- `trace_id` and `span_id` (from the current OTEL span)
- `session_id` and `session_trace_id` (`session_id` without dashes)

This makes it easy to jump from a Loki log line to a Tempo trace:
- copy `session_trace_id`
- in Grafana Tempo, use TraceID lookup
