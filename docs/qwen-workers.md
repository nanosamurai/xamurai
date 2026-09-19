# Qwen finalization and refinement

`qwen_worker` supplies one offline pipeline and two small entrypoints using the
existing finalization and refinement Kafka runtimes. Model loading, immutable
artifact verification and pyannote diarization live in `xamurai_serving.qwen`
and are shared with Qwen realtime. The realtime profile and streaming API stay
unchanged. No protobuf, transcript data shape, topic or database changes are
needed.

Build both workers, including their Qwen runtime dependency:

```sh
docker buildx bake --load qwen-finalizer qwen-refinement
```

The images run `qwen_worker.finalizer` and `qwen_worker.refinement`. Their
default track is `qwen`, with separate `finalizer.qwen` and `refinement.qwen`
consumer groups. Existing selection, retention, acknowledgement, replay and
refinement window/idle settings apply. The model label is `Qwen/Qwen3-ASR-0.6B`.

## Offline inference

Pyannote diarizes the whole input first, preserving anonymous speaker identity
throughout a recording. Qwen transcribes those speaker turns in vLLM batches,
splitting long turns into at most 30-second crops. Overlapping turns assign each
sample to the first turn so the same audio is not transcribed twice; simultaneous
voices are not separated. Silence without detected turns yields an empty result.
Each recording/window has fresh speaker state; labels across refinement windows
do not establish identity. Enrolled names are not supported.

The existing segment times describe the audio crops, not aligned word boundaries.
No word timing is fabricated. Unlike realtime enrichment, this pipeline does not
require a supported forced-alignment language: Czech and the other ASR languages
can still have speaker-labelled segments. A supported ISO language hint is passed
to Qwen; absent or unsupported hints use automatic detection.

| Setting | Default | Bounds / purpose |
| --- | --- | --- |
| `QWEN_BATCH_SIZE` | 4 | 1–32 crops per inference call and vLLM maximum sequences |
| `QWEN_BATCH_CHUNK_SECONDS` | 30 | 1–30 seconds per crop |
| `QWEN_KV_CACHE_MIB` | 1024 | 512–16384 MiB, explicit vLLM cache allocation |
| `HF_TOKEN` | required | Read access to the pinned gated pyannote models |

Batches contain crops from one recording or refinement window; there is no new
cross-session scheduler. ASR uses the existing pinned vLLM 0.14.0 / qwen-asr 0.0.6
runtime with 1024 output tokens and a 4096-token context. Full audio and diarization
workspace must fit in memory; short-fixture qualification is not a capacity claim.

## Validation and security

`tests/test_qwen_worker_integration.py` observes actual vLLM `generate` calls to
prove multi-crop batching with real models, timed speakers, independent requests,
empty/silent input and invalid WAV rejection. Nanosamurai owns the full Compose
smoke through BFF, recorder, Kafka, Postgres, live refinement and HTTP playback.

On 2026-09-19, both real GPU integration tests and 28 realtime/shared-worker
regression checks passed. Rebuilt finalizer/refinement images also passed the
full [Nanosamurai Compose smoke](https://github.com/nanosamurai/nanosamurai/blob/master/docs/qwen-workers.md),
including replay, stage selection, silence and tenant rejection.

Images retain the realtime model revisions/hashes and run as UID 10002. Model
repository code is disabled, telemetry is disabled, transcript content is not
logged, and no host port is published by the workers. Tokens enter at runtime,
never through build arguments. Refinement needs no storage credentials. Existing
temporary-WAV cleanup, tenant checks and replay behavior are reused.
