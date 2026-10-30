# Voxtral realtime

`voxtral_rtservice` is an independent `RealtimeASR` gRPC peer for
`mistralai/Voxtral-Mini-4B-Realtime-2602` (the published release is **2602**).
It runs BF16 inference through one shared **vLLM 0.32.0 AsyncLLM** engine per
replica, using native causal audio processing and per-request state. It does
not depend on Qwen's older vLLM environment, CTranslate2, Triton, or a hosted
Mistral API. The image is built with `docker buildx bake --load voxtral-rtservice`.

This implementation supplies production lifecycle and resource controls; a
successful image build or CPU test does not qualify a GPU concurrency setting.
Complete the capacity and quality checks below before production rollout.

## Runtime decision and boundaries

The model vendor recommends vLLM. Its async scheduler shares weights and batches
compatible inference steps across concurrent requests. A single Python event
loop services all admitted gRPC streams; there is no one-session lock, fixed
four-thread RPC bottleneck, or model copy per session. GPU worker processes are
owned by the engine. `RT_SERVING_MAX_SESSIONS` is required explicitly and sets
both admission capacity and vLLM `max_num_seqs`.

We integrate the engine directly because vLLM 0.32's WebSocket frontend and
native audio feeder use unbounded queues. The adapter instantiates a native
`VoxtralRealtimeBuffer` per epoch, replaces only its initially empty audio queue
with a bounded queue, and drives the documented streaming prompt/token-feedback
sequence. Audio production waits when that queue fills. There is no HTTP or
WebSocket listener through which callers can bypass gRPC admission.

The native buffer and renderer are **version-specific internal APIs**. Runtime
startup rejects a different vLLM version, incompatible sample geometry, or an
oversized token prefix. Upgrade the image, adapter, and GPU qualification
together. This small integration avoids maintaining a model implementation or
forking the upstream inference scheduler.

## Stream behavior

The existing public protobuf is unchanged. A caller supplies `x-session-id` and
waits for `SESSION_ACCEPTED` before sending audio. A full replica returns
`RESOURCE_EXHAUSTED / REPLICA_FULL` without reading audio. Retry that unadmitted
RPC on another replica. Accepted streams remain on their original replica.

Input is PCM16 little-endian mono at 16 kHz. Each chunk must contain 1–16000
samples; empty or odd-length payloads, unsupported language hints, changes to
session/tenant/language, and noncontiguous sequence numbers fail the stream.
The first sequence number can be any uint64. Session and tenant IDs are bounded
to 256 characters and each protobuf message to 64 KiB. Model IDs, runtime flags,
and artifact paths are never accepted from clients; session settings are `{}`.

Each public stream is split at exact integer-sample boundaries into independent
60-second native transcription epochs. Within an epoch audio is submitted once
and decoder/encoder state is reused. `PARTIAL` replaces the current epoch's text;
normal epoch completion emits one immutable `FINAL`, then advances the public
replacement interval. Normal request EOF flushes right padding and emits the
last final. Empty streams close without inference. Cancellation aborts the
engine request immediately and does not manufacture a final. No audio or text
from earlier epochs is retained for inference.

**Epoch resets can split words and lose linguistic context.** This first version
does not use VAD endpointing, overlap, seam deduplication, forced alignment,
diarization, or speaker enrollment. Raising epoch length reduces reset frequency
but increases context and text-memory requirements. Seam accuracy must be
measured before selecting a production epoch duration. Native state is bounded;
the public stream has no provider-imposed total recording duration limit
(`maximum_audio_seconds=0`). An inactive input sender still times out.

Event start/end values describe the epoch's submitted-audio interval, not
recognized word timing. A partial's end can lead decoded speech by the bounded
input queue. Capabilities therefore advertise `segment_timestamps=false`,
`word_timestamps=false`, `speaker_labels=false`, and `language_detection=false`.
Voxtral can transcribe multilingual speech but does not return a detected
language through this adapter; events leave `lang` empty instead of inventing a
result from the input hint. Supported hints are en, fr, es, de, ru, zh, ja, it,
pt, nl, ar, hi, ko. **Czech is not in this release's supported language list.**
`batch=true` describes shared engine scheduling, not an offline batch RPC.

Invalid input, idle timeout, inference failure, context truncation, and text-limit
overflow end the affected stream explicitly. They never turn a partial epoch
into an apparently successful final. Every admitted RPC releases its slot once,
after backend teardown. Failed or timed-out engine aborts retire the whole
replica, fail readiness and new admission, and exit for an orchestrator restart;
unknown engine capacity must not be reused.

## Configuration

All settings are operator-owned and validated before loading model weights.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RT_SERVING_MAX_SESSIONS` | **required** | Admitted streams and vLLM `max_num_seqs`; 1–1024 is a configuration guard, not a qualified GPU capacity |
| `RT_SERVING_INSTANCE_ID` | hostname | Admission response identity; use a unique identity per replica |
| `VOXTRAL_RTSERVICE_BIND_ADDR` | `127.0.0.1` | Listener; explicitly choose `0.0.0.0` inside a private container/pod network |
| `VOXTRAL_RTSERVICE_PORT` | `50052` | gRPC port |
| `VOXTRAL_METRICS_BIND_ADDR` | `127.0.0.1` | Prometheus listener; explicitly allow a private pod network for scraping |
| `VOXTRAL_METRICS_PORT` | `8001` | Prometheus HTTP port |
| `VOXTRAL_EPOCH_SECONDS` | `60` | 10–600; resets native state after this much submitted audio |
| `VOXTRAL_MAX_MODEL_LEN` | `2048` | Must cover `ceil(epoch_seconds / 0.08) + 512`; maximum 131072 |
| `VOXTRAL_KV_CACHE_MIB` | `2048` | Explicit engine KV allocation per GPU; independent of session count and model-weight memory |
| `VOXTRAL_MAX_BATCHED_TOKENS` | `2048` | Scheduler throughput/latency budget; must be at least the session count |
| `VOXTRAL_TENSOR_PARALLEL_SIZE` | `1` | GPUs per shared engine; values above one need separate hardware qualification |
| `VOXTRAL_AUDIO_QUEUE_FRAMES` | `32` | 1–256 native input frames; 160 ms per frame |
| `VOXTRAL_IDLE_TIMEOUT_SECONDS` | `30` | Maximum wait for the next audio chunk |
| `VOXTRAL_PROGRESS_TIMEOUT_SECONDS` | `60` | Maximum wait for an engine result, including blank-token progress |
| `VOXTRAL_OUTPUT_TIMEOUT_SECONDS` | `30` | Maximum wait for a gRPC write blocked by a slow consumer |
| `VOXTRAL_CLEANUP_TIMEOUT_SECONDS` | `10` | Engine abort/health-check deadline; failure retires the replica |
| `VOXTRAL_MAX_EPOCH_TEXT_BYTES` | `65536` | UTF-8 text limit per epoch; overflow fails the stream |
| `VOXTRAL_MODEL_MANIFEST` | bundled profile | Optional operator-mounted JSON manifest described below |
| `HF_HOME` | `/data/huggingface` in image | Model cache, writable by UID 10004 |

The queue alone retains at most `frames * 2560 * 4` float32 bytes: 320 KiB
and 5.12 seconds with defaults. There is also one producer frame, one incoming
PCM message (up to 32000 bytes), native look-back/look-ahead and padding, bounded
epoch text, gRPC flow-control buffers, and engine-owned state. This is **not** a
bound on total process RSS or VRAM. The 1 GiB multimodal processor cache, CUDA
graphs, encoder caches, activations, weights, and allocator overhead must also
be budgeted. No whole recording or epoch PCM array is retained by the adapter.

## Model provenance and switching

The bundled profile is `voxtral-mini-4b-realtime-2602-vllm-r1`, HF revision
`2769294da9567371363522aac9bbcfdd19447add`. Startup downloads only `params.json`,
`tekken.json`, and `consolidated.safetensors` and verifies all three SHA-256
digests before loading inference. Digests live in
[`models.py`](../voxtral_rtservice/src/voxtral_rtservice/models.py).
The image pins the vLLM 0.32.0 manifest to
`sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`.
Remote model Python code is disabled. The model is Apache-2.0 and ungated;
no token is required for the default profile.

Voxtral pins its own gRPC/protobuf and aligned OpenTelemetry dependencies in
`voxtral_rtservice/requirements.txt`; Qwen and other service images are unchanged.
The build checks dependencies and permits exactly one vendor-documented
exception: torch 2.13.0+cu130 declares NCCL 2.29.7, while the upstream vLLM image
deliberately installs NCCL 2.30.7 for DeepEPv2. The checker verifies all three
runtime versions and rejects every other dependency conflict. This exception
must be reviewed when upgrading the base image; it is not a blanket `pip check`
bypass. See the [pinned upstream Dockerfile](https://github.com/vllm-project/vllm/blob/v0.32.0/docker/Dockerfile).

An operator manifest has exactly `profile_id`, `model_id`, `revision`,
`languages` (list of public language codes), and `files` (the three filenames
mapped to SHA-256 hex digests). Use a new profile ID for changed artifacts.
The manifest permits compatible future Voxtral realtime checkpoints without
changing public APIs. It does not imply that offline Voxtral models or arbitrary
architectures work. Verify architecture, tokenizer geometry, language claims,
context headroom, and quality before installing another manifest. There is
currently one selected realtime size, approximately 4.4B total parameters.

The pinned tokenizer supplies the vendor's 480 ms transcription delay. This is
a model delay setting, **not an end-to-end latency guarantee**. This version
does not mutate cached tokenizers or expose a per-session delay override.

## Deployment and scaling

Example for a **two-session qualification run**, not a validated capacity claim:

```bash
docker buildx bake --load voxtral-rtservice
docker run --rm --gpus all --shm-size=2g \
  --cap-drop=ALL --security-opt=no-new-privileges \
  -p 127.0.0.1:50056:50052 \
  -e VOXTRAL_RTSERVICE_BIND_ADDR=0.0.0.0 \
  -e RT_SERVING_MAX_SESSIONS=2 \
  -v voxtral-model-cache:/data/huggingface \
  xamurai-voxtral-rtservice:local
```

The host mapping is loopback-only. Container gRPC has no embedded authentication
or TLS: use the authenticated BFF, network policies restricting callers to BFF,
and transport protection appropriate to the cluster. Do not expose this port
publicly. Runtime is non-root; cache access is limited to the service identity.
No audio/transcript files are written and adapter logs contain outcomes, counts,
and exception types, not transcript/audio contents. Upstream request logging,
prefix caching, and usage telemetry are disabled. Audit upstream debug/error
logging before enabling verbose production logs.

For horizontal scale, deploy multiple independent replicas with GPU allocations,
a headless internal service, and the existing BFF DNS/round-robin admission retry
policy. Register the `voxtral` track with the BFF using the deployment repository's
operator configuration. There is no shared admission database. Do not run several
adapters against one shared engine: this implementation owns its engine and
capacity. Add replicas for new sessions; live state is not migrated. Drain or
allow active sessions to finish before planned scale-down; the process uses a
five-second shutdown grace, then cancels remaining streams.

For vertical scale, raise `RT_SERVING_MAX_SESSIONS` together with a measured KV,
encoder-cache, CPU/RSS, and latency budget. A larger GPU or tensor parallelism
can increase capacity, but simply raising `max_num_seqs` cannot create VRAM or
guarantee realtime deadlines. One-session operation is supported as an operator
choice, never a coded concurrency ceiling. Avoid GPU oversubscription by other
pods/processes. BF16 weights alone are roughly 8.9 GB; the vendor's 16 GB fit
claim is not a multi-session sizing result. Start qualification on a dedicated
24 GB GPU with explicit headroom.

Prometheus exposes `xamurai_voxtral_active_sessions`, `session_capacity`,
`admissions_total{outcome}`, `closed_sessions_total{outcome}`, and histograms
`first_partial_seconds` and `flush_seconds` (all with the `xamurai_voxtral_`
prefix), alongside vLLM's own registered metrics. There are no session, tenant,
or transcript labels. Scale on admitted/capacity ratio, rejection rates,
latency, and GPU/RSS headroom; CPU alone is not a suitable admission signal.
Restrict the metrics listener to the monitoring network as well as gRPC to BFF.

## Validation and rollout gates

CPU tests run real loopback gRPC with synthetic PCM and an injected backend;
backend tests exercise the production async driver with a fake engine. They
cover admission, interleaved streams, tenant/sequence isolation, epoch splitting,
backpressure, EOF, cancellation, backend/input failure, timeout, and failed-abort
retirement. Run them without downloading model weights:

```bash
python -m pip install -r requirements.ci.txt
pytest -q tests/test_voxtral_realtime.py tests/test_voxtral_backend.py
```

With a running GPU service, use a synthetic or consented English fixture:

```bash
python utilities/voxtral_realtime_probe.py /path/to/fixture-16k-mono.wav \
  --target 127.0.0.1:50056 --sessions 2 --seconds 70
```

The probe checks concurrent admission, overload rejection when filling the
advertised capacity, epoch coverage, final text presence, and slot reuse. It
prints timing/count metrics, not transcript text. `--unpaced` additionally tests
input backpressure. It does not measure WER, prove tenant isolation on real
weights, or replace a multi-hour soak. Use separate recordings and reference
transcripts for quality and cross-session contamination checks.

Before rollout, record image/model hashes, GPU/driver, configuration, corpus
provenance, and results. Qualification must include concurrent speech and silence,
cancel/reconnect churn, overload rejection and recovery, slow output consumers,
and recordings crossing many epoch boundaries. Test 1, 2, 4, then larger session
counts as hardware permits; record p50/p95/p99 latency, tail flush time, real-time
factor, peak VRAM/RSS, and sustained memory growth. Include mixed languages,
silence hallucinations, and seam WER at several epoch durations. Require memory
and latency headroom at the chosen capacity; stop admitting additional sessions
before OOM or deadline failure.

The selected vLLM release includes upstream concurrent encoder-cache sizing and
disconnect-abort fixes. Upstream reports still describe engine host-memory
retention in some multimodal paths. Bounded adapter queues do not establish that
all engine allocations are bounded; multi-hour RSS/VRAM soak remains mandatory.
GPU qualification, seam quality, and tensor-parallel performance are separate
acceptance gates. Deployment Helm/Compose wiring belongs in nanodeploy or the
public nanosamurai deployment repository and is not changed by this service PR.

## Primary references (checked 2026-10-30)

- [Mistral model card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [vLLM 0.32 realtime protocol](https://docs.vllm.ai/en/v0.32.0/serving/online_serving/speech_to_text/)
- [Native buffer implementation](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/voxtral_realtime.py)
- [Upstream realtime connection and cleanup](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/speech_to_text/realtime/connection.py)
- [Concurrent streaming fixes, PR 45022](https://github.com/vllm-project/vllm/pull/45022)
- [Host-memory retention report, issue 50860](https://github.com/vllm-project/vllm/issues/50860)
