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
There is one active job per stage and a bounded refinement ready queue. Additional
capacity comes from replicas, rather than accepting unlimited jobs in one process.

VAD, alignment, diarization, enrollment-model operations and ASR all use the same
owner thread. Non-ASR operations alternate with ASR batches when both are ready.
Alignment and diarization keep existing whole-job behavior and can still delay
refinement; batching does not guarantee a maximum end-to-end latency. Refined
speaker-turn retranscription also uses the shared ASR scheduler.

## Availability and scaling

Groups default to `refinement.whisperx-shared` and `finalizer.whisperx-shared`.
Both stages retain their existing topics, headers and protobuf outputs. The new
track is opt-in: selection must name `whisperx-shared` for the desired stage.
No existing `whisperx` or `kserve-whisperx` offsets or sessions are migrated.

Finalization keeps polling while inference and output acknowledgment happen on
a processing thread. Unselected records are skipped on the poll thread without
pausing intake or occupying the processing slot. The poll thread commits selected
records only after acknowledgment, then
performs optional recording deletion. Revocation/loss invalidates completion
ownership even if the same partition is assigned again. Replay can duplicate
publication; existing persistence remains idempotent. Refinement retains its
existing conservative active-session checkpoint.

The supervisor requires both loops and the model owner to remain alive. A failed
loop or owner restarts the whole process. A hung model operation is detected by
its deadline. Shutdown stops intake, allows bounded draining and leaves
unfinished offsets uncommitted. The health probe opens no network port.

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
| `WHISPERX_SHARED_OPERATION_TIMEOUT_S` | `1800` | Maximum running model-operation time |
| `WHISPERX_SHARED_DRAIN_SECONDS` | `30` | Shutdown drain limit |
| `WHISPERX_SHARED_HEALTH_FILE` | `/tmp/whisperx-shared.json` | Local supervisor heartbeat and batch counts |

Existing model, language alignment, enrollment, Kafka/TLS and recording settings
are reused. Combined mode requires configured diarization to initialize before
starting consumers; `WHISPERX_ENABLE_DIARIZATION=false` explicitly disables it.
Batch logs contain counts and stages, not audio or transcript content.

## Validation

Run `pytest -q -m 'not integration'` for lightweight regressions, including mixed
and partial batches, language isolation, long-finalization fairness, nested model
operations, owner failure, polling during inference and commit/deletion fencing.
Nanodeploy owns the optional Compose overlay and the real shared-track smoke.
Use its `docs/whisperx-shared.md` for GPU execution and deployment choices.

For native/shared ASR parity, run
`WHISPERX_SHARED_GPU_TEST=1 pytest -q tests/test_whisperx_shared_integration.py`
inside the pinned GPU worker image with this checkout and model cache mounted.
The test compares text/chunk timestamps against native WhisperX using the same
model, proves a mixed batch, and checks that results stay with their request.
Stop the idle combined worker during this test to avoid loading an extra bundle.
