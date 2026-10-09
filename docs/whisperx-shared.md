# Shared WhisperX refinement and finalization

`python -m whisperx_worker.combined` runs the optional `whisperx-shared` track.
The existing `whisperx_worker.refinement` and `whisperx_worker.finalizer`
entrypoints retain their standalone behavior and default `whisperx` track.
The same worker image supports all three commands. WhisperX is pinned to 3.8.6
because the shared adapter uses its chunk preparation and batch decoder APIs.

## Pipeline and batching

Two Kafka poll loops each have one outstanding processing job. Both reuse
`pipeline.py`; inference hooks execute directly in standalone mode and submit
to a single model-owning thread in combined mode. Common models initialize once.
Each transcription prepares its own VAD chunks and language/tokenizer state.
Ready chunks with the same language share an ASR batch, up to 16 by default.
All requests use the same configured model, transcription task and base decoding
options. Prompts, waveforms, offsets and results remain request scoped.

The scheduler rotates requests by chunk, fills available capacity, and dispatches
partial batches after a maximum 20 ms collection interval. A full batch runs
immediately. The interval starts when prepared chunks become ready; it excludes
time waiting for an already running model operation. Audio accumulation for
refinement uses the existing separate window/idle settings.

Only the current batch's features are materialized. Long recordings retain
their waveform and VAD metadata, so memory still scales with recording duration.
Refinement has one active job and a bounded ready queue. Finalization has a
configurable bounded number of outstanding jobs (two by default), including
completed results waiting for an earlier Kafka offset to finish.

VAD, diarization, enrollment-model operations and ASR use the same owner thread.
Non-ASR operations alternate with ASR batches when both are ready. Refined
speaker-turn retranscription also uses the shared ASR scheduler. Diarization
keeps its whole-job behavior and can still delay refinement.

Alignment has its own bounded executor and keeps one resident model per configured
language. Different languages can align concurrently, with one CUDA stream per
language and at most one call into each model at a time. A queued call for a busy
language does not prevent another language from using a free executor slot.
Alignment retains WhisperX's segment-by-segment algorithm; the ASR batch size
does not batch alignment. ASR can progress while alignment is running. These
operations still compete for the same GPU, so concurrency is not a latency SLA
or a guarantee that GPU kernels overlap. Inputs, results and trace contexts stay
request scoped. All GPU alignment models finish loading before threads start.

## Availability and scaling

Groups default to `refinement.whisperx-shared` and `finalizer.whisperx-shared`.
Both stages retain their existing topics, headers and protobuf outputs. The new
track is opt-in: selection must name `whisperx-shared` for the desired stage.
No existing `whisperx` or `kserve-whisperx` offsets or sessions are migrated.

Finalization keeps polling while inference and output acknowledgment happen on
a processing thread. Unselected records are skipped on the poll thread without
pausing intake or occupying the processing slot. The poll thread commits selected
records only after acknowledgment and all earlier records on that partition
complete, then performs optional recording deletion. Unselected records also
wait behind unfinished earlier offsets. Other partitions may commit independently.
Revocation/loss invalidates completion ownership even if the same partition is
assigned again. Replay can duplicate
publication; existing persistence remains idempotent. Refinement retains its
existing conservative active-session checkpoint.

The supervisor requires both loops, the model owner and alignment executor to
remain healthy. A failed loop or executor restarts the whole process. A hung
model or alignment operation is detected by its deadline. Shutdown stops intake,
allows bounded draining and leaves unfinished offsets uncommitted. The health
probe opens no network port.

One replica has a shared failure boundary and is not highly available. Additional
replicas each load one model bundle and receive work through the existing two
consumer groups. Topic partitions and GPUs limit useful replica counts; requests
on different replicas cannot share batches. Separate entrypoints remain available
for independent scaling and failure isolation.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `REFINEMENT_TRACK_ID`, `FINALIZER_TRACK_ID` | `whisperx-shared` | Output and selection identity in combined mode |
| `KAFKA_GROUP_ID` | `refinement.<track>` | Refinement consumer group |
| `KAFKA_GROUP_ID_FINALIZER` | `finalizer.<track>` | Finalization consumer group |
| `WHISPERX_SHARED_BATCH_SIZE` | `16` | Maximum ASR chunks per call, 1–16 |
| `WHISPERX_SHARED_BATCH_WAIT_MS` | `20` | Partial-batch collection interval, 0–1000 ms |
| `WHISPERX_SHARED_ALIGNMENT_LANGUAGES` | `en` | Comma-separated set of resident alignment models, e.g. `en,de,cs` |
| `WHISPERX_SHARED_ALIGNMENT_CONCURRENCY` | `2` | Concurrent alignments, 1–8; also limited to one per language |
| `WHISPERX_SHARED_FINALIZATION_CONCURRENCY` | `2` | Outstanding finalizations per process, 1–8 |
| `WHISPERX_SHARED_OPERATION_TIMEOUT_S` | `1800` | Maximum running model-operation time |
| `WHISPERX_SHARED_DRAIN_SECONDS` | `30` | Shutdown drain limit |
| `WHISPERX_SHARED_HEALTH_FILE` | `/tmp/whisperx-shared.json` | Local supervisor heartbeat and batch counts |

Existing model, language alignment, enrollment, Kafka/TLS and recording settings
are reused. Combined mode requires configured diarization to initialize before
starting consumers; `WHISPERX_ENABLE_DIARIZATION=false` explicitly disables it.
Batch logs contain counts and stages, not audio or transcript content.

Combined mode loads every configured alignment model and its sentence tokenizer
before consumers or the health heartbeat start; failure leaves the worker unready.
Use `cs` for Czech. Codes are normalized and duplicates removed; unsupported model
codes fail startup. The older singular `WHISPERX_SHARED_ALIGNMENT_LANGUAGE` is
accepted when the plural setting is absent. Requests outside the preloaded set
use the existing unaligned transcript fallback (no word timestamps), without
downloading or replacing models. A serving executor failure instead restarts
the worker and leaves unfinished inputs for replay.

Standalone entrypoints keep their existing single-model, lazy behavior. Shared
mode still loads Whisper ASR only once. Each additional alignment model increases
resident VRAM, and concurrent alignments need additional inference buffers. Tune
both concurrency limits against the GPU and recording lengths. Set `TORCH_HOME`
and `NLTK_DATA` to persistent cache directories to retain checkpoints/tokenizers;
the shared Nanodeploy overlay configures both.

## Validation

Run `pytest -q -m 'not integration'` for lightweight regressions, including mixed
and partial batches, language isolation, long-finalization fairness, nested model
operations, owner failure, polling during inference and commit/deletion fencing.
Startup tests verify that alignment loading blocks consumers/readiness and that
a failed download prevents serving. Scheduler tests cover language isolation,
ASR progress during alignment, per-model/global concurrency bounds, failure and
shutdown. Finalizer tests cover out-of-order completion, unselected records behind
unfinished work, independent partitions, failed publication and revocation fencing.
Nanodeploy owns the optional Compose overlay and the real shared-track smoke.
Use its `docs/whisperx-shared.md` for GPU execution and deployment choices.

For native/shared ASR parity, run
`WHISPERX_SHARED_GPU_TEST=1 pytest -q tests/test_whisperx_shared_integration.py`
inside the pinned GPU worker image with this checkout and model cache mounted.
The test compares text/chunk timestamps against native WhisperX using the same
model, proves a mixed batch, and checks that results stay with their request.
Stop the idle combined worker during this test to avoid loading an extra bundle.

Run `WHISPERX_SHARED_GPU_TEST=1 pytest -q tests/test_whisperx_alignment_integration.py`
in the same GPU image to preload `en,de,cs`, compare concurrent English/Czech
alignment against sequential results (including word timestamps), and assert
that two alignment calls overlap while the same models remain resident.
