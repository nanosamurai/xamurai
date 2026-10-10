# WhisperX shared track

`whisperx-shared` is an optional refinement and finalization track. It runs both
stages in one Python process and loads Whisper ASR once. It can combine ready
audio chunks from different requests into one ASR batch. This reduces the model
memory needed when both stages share a GPU.

The existing `whisperx` track still runs refinement and finalization as separate
workers. It remains useful when the stages need independent capacity or failure
isolation. Both tracks use the same WhisperX pipeline and output contracts.

## How it compares with the other tracks

Track IDs are selected separately for realtime, refinement and finalization.
The table lists the default IDs provided by Xamurai. A deployment must start the
matching workers and register its available tracks with SamuraiBFF.

| Track | Stages | Main difference |
| --- | --- | --- |
| `faster-whisper` | Realtime | Whisper medium produces replaceable partial text and speaker-labelled committed results over gRPC. It is separate from the offline WhisperX workers. |
| `nemotron` | Realtime | Nemotron uses a native streaming model, with optional Sortformer diarization. It does not provide the refinement or full-recording finalization stages. |
| `qwen` | Realtime, refinement, finalization | Separate Qwen services use Qwen ASR, forced alignment and pyannote. Offline workers batch speaker-turn crops from one job; they do not share model memory between processes. |
| `parakeet` | Refinement, finalization | Separate workers use Parakeet TDT with native word timing and embedded Sortformer for up to four anonymous speakers. No separate word-alignment model is needed. |
| `whisperx` | Refinement, finalization | Separate workers use WhisperX and pyannote. Each process loads its own models and can scale independently. |
| `whisperx-shared` | Refinement, finalization | One process shares Whisper ASR and diarization between both stages. Compatible requests can share ASR batches. Selected language aligners stay loaded and can run concurrently. |

`whisperx-shared` does not provide realtime transcription. It can be used with
any configured realtime track. For the other worker pipelines, see
[Qwen workers](qwen-workers.md), [Parakeet refinement](parakeet-refinement.md)
and the [realtime provider guide](modular-asr-providers.md).

## Run and select the track

The image built from `whisperx_worker/Dockerfile.refinement` supports the combined
worker. Use this command in the container:

```sh
python -m whisperx_worker.combined
```

Provide the existing Kafka, recording-storage, model-access and enrollment
settings. Register `whisperx-shared` in SamuraiBFF's refinement and final track
lists. Clients can then select `refinement_tracks=whisperx-shared` and
`final_tracks=whisperx-shared` independently. These settings do not rename or
replace the existing `whisperx` track or its consumer groups.

The standalone commands remain `python -m whisperx_worker.refinement` and
`python -m whisperx_worker.finalizer`. The combined worker uses the same pipeline
functions. WhisperX is pinned to 3.8.6 because the batching adapter calls its
chunk-preparation and decoder APIs.

## Batching and alignment

Two Kafka consumers keep polling while inference runs. Refinement processes one
window at a time. Finalization allows two jobs at a time by default, including
completed jobs waiting for an earlier Kafka offset to finish.

Each ASR request runs VAD and prepares its own speech chunks and language state.
Ready chunks with the same language can share a batch, up to 16 by default.
The scheduler takes turns selecting chunks from ready requests so a long
finalization does not always run ahead of refinement. A full batch runs
immediately. A partial batch waits up to 20 ms from the time its first chunks
become ready. Time spent waiting for a running model operation is additional.
Different languages use separate ASR batches.

VAD, ASR, diarization and enrollment inference use one model thread. When both
are queued, other model operations alternate with ASR batches. Diarization runs
for the whole input and can delay other requests. Only the current ASR batch's
features are built, but each job still holds its waveform and VAD metadata.

Refinement keeps its existing window and idle settings. It does not run word
alignment. With speaker-turn split mode enabled, it can transcribe a window
again by speaker turn; those calls also use the shared scheduler. The final
partial window waits for the idle flush, which defaults to 30 seconds. A client
finish request does not directly flush this refinement buffer.

Finalization runs ASR, word alignment and then diarization in sequence. Alignment
has a separate executor, so ASR for another job can progress during alignment.
One alignment model is kept in memory for each configured language. Different
languages can align at the same time, with one CUDA stream per language and at
most one call per model. Waiting for a busy language does not block a free slot
from serving another language. Alignment still processes segments using
WhisperX's algorithm; the ASR batch size does not set an alignment batch size.

Concurrent stages still share GPU capacity. More concurrency does not guarantee
lower latency or simultaneous GPU kernel execution. Audio, results, language
state and trace context remain separate for each request.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `WHISPERX_MODEL` | `medium` | Shared Whisper model; set `large-v3` when GPU capacity allows |
| `WHISPERX_COMPUTE_TYPE` | `float16` on CUDA | Existing Whisper precision setting |
| `REFINEMENT_TRACK_ID`, `FINALIZER_TRACK_ID` | `whisperx-shared` | Track IDs in combined mode |
| `KAFKA_GROUP_ID` | `refinement.<track>` | Refinement consumer group |
| `KAFKA_GROUP_ID_FINALIZER` | `finalizer.<track>` | Finalization consumer group |
| `WHISPERX_SHARED_BATCH_SIZE` | `16` | Maximum ASR chunks per batch, 1–16 |
| `WHISPERX_SHARED_BATCH_WAIT_MS` | `20` | Partial-batch collection wait, 0–1000 ms |
| `WHISPERX_SHARED_ALIGNMENT_LANGUAGE` | `en` | Comma-separated languages to preload, for example `en,de,cs` |
| `WHISPERX_SHARED_ALIGNMENT_CONCURRENCY` | `2` | Concurrent alignment calls, 1–8 and at most one per language |
| `WHISPERX_SHARED_FINALIZATION_CONCURRENCY` | `2` | Admitted finalization jobs per process, 1–8 |
| `WHISPERX_SHARED_OPERATION_TIMEOUT_S` | `1800` | Time limit used to detect a stuck model or alignment operation |
| `WHISPERX_SHARED_DRAIN_SECONDS` | `30` | Shutdown drain limit |
| `WHISPERX_SHARED_HEALTH_FILE` | `/tmp/whisperx-shared.json` | Local health heartbeat and batch/alignment counters |

Existing Kafka/TLS, model, recording, enrollment and refinement settings still
apply. Set `REFINEMENT_READY_QUEUE_MAX` to bound waiting refinement windows for
the available host memory. Health checks run `python -m whisperx_worker.health`
and do not open a network port.

Alignment language codes are normalized and duplicates removed. Use `cs` for
Czech. An unsupported model language fails startup. All configured aligners and
sentence tokenizers load before consumers start and the worker becomes ready.
A failed preload prevents serving.

Requests in a language outside this set keep the existing fallback: text and
segments without aligned word timestamps. They cannot trigger a new model
download. Set `HF_HOME`, `TORCH_HOME` and `NLTK_DATA` to persistent cache locations
to reuse downloads after a restart. Each additional model and concurrent call
needs more GPU memory. Standalone workers keep their existing lazy alignment
loading behavior.

## Availability and scaling

Both stages keep the existing topics, headers and protobuf outputs. Kafka input
is committed only after output publication is acknowledged and all earlier
records on that partition have completed. Other partitions may commit
independently. A worker that loses a partition cannot commit its old work or
delete the recording, even if it later gets the same partition back. Retries can
publish duplicate results, so persistence must keep its existing deduplication.
Refinement keeps its existing active-session recovery behavior.

The supervisor checks both consumer loops, the model thread and the alignment
executor. A failure or stuck operation makes the process exit so the container
manager can restart it. Shutdown stops intake, gives work a bounded time to
finish and leaves unfinished offsets available for replay. Diarization must
load before serving unless `WHISPERX_ENABLE_DIARIZATION=false` disables it.

One shared replica is one failure point for both stages. Each additional replica
loads its own models and joins the two consumer groups. Useful replica counts
depend on Kafka partitions, available GPUs and recording lengths. Batches cannot
span replicas. Use the separate `whisperx` workers when independent scaling or
failure isolation matters more than sharing model memory. Running both tracks
at once loads models for both deployments.

Queues and job counts are bounded; recording memory still grows with duration.
Measure GPU and host memory for the target workload. Batch logs contain counts
and stages rather than transcript content. Existing model trust and storage
access settings still apply.

## Validation

The lightweight CI job runs the shared-worker tests alongside the existing
pipeline and stream-control tests. They cover mixed/partial batches, language
and request isolation, startup readiness, concurrent alignment, bounded queues,
ASR progress during alignment, failed publication, shutdown and safe commits
after a consumer rebalance.

For native/shared ASR parity, run
`WHISPERX_SHARED_GPU_TEST=1 pytest -q tests/test_whisperx_shared_integration.py`
inside the pinned GPU image with the checkout and model cache mounted. It
compares text and chunk timestamps with native WhisperX and checks mixed-batch
result routing.

Run `WHISPERX_SHARED_GPU_TEST=1 pytest -q tests/test_whisperx_alignment_integration.py`
in the same image to compare concurrent English/Czech alignment with sequential
text and word timestamps using the same resident models. Stop any idle worker
that would otherwise occupy the test GPU with another model bundle.

For an application smoke test, select this track through the BFF, submit
refinement while recordings finalize, and verify final/refined track IDs,
speaker labels, aligned words, persisted history and replay. Inspect batch
counts and alignment start/end times to confirm actual overlap; submitting
concurrent jobs alone does not prove it happened.
