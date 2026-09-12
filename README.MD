# Xamurai

Xamurai (voX samurai) provides the speech-to-text and speaker-diarization services used by
the nanosamurai platform. It supports realtime transcription over gRPC and
asynchronous refinement, recording, and finalization over Kafka.

The opt-in [final-track spike](docs/final-tracks.md) adds independent completed-recording
outcomes on `transcripts.final-tracks`, with a primary compatibility projection.
[Refinement tracks](docs/refinement-tracks.md) reuse that processor with immutable
live audio windows, independent track groups and a primary refined projection.

For a complete self-hosted Community Edition stack, use the public
[`nanosamurai/nanosamurai`](https://github.com/nanosamurai/nanosamurai)
repository. Deployment orchestration and environment-specific infrastructure
do not live in this service repository.

## Services

- `rtservice` exposes the bidirectional `RealtimeASR` gRPC API and emits
  Faster-Whisper replaceable partials and pyannote-diarized committed finals,
  with optional enrolled-speaker mapping.
- `qwen_rtservice` exposes the same `RealtimeASR` API using Qwen3-ASR's native
  vLLM streaming implementation, then forced-aligns and pyannote-diarizes each
  bounded final epoch when the detected language is supported.
- `nemotron_rtservice` exposes the same API using Nemotron 3.5 ASR's
  cache-aware streaming graph through the stable NeMo-Speech.cpp C ABI, with
  optional native Sortformer diarization and tenant-specific S3 speaker matching.
- `whisperx_worker` consumes `audio.raw` and emits window-based
  `transcripts.refined` events.
- `recorder_worker` consumes `audio.raw`, stores session audio, and emits
  `recordings.finished`.
- `finalizer_worker` consumes `recordings.finished`, produces the canonical
  session transcript, and emits `transcripts.final`.
- `shared` contains the compatibility package `drsynth_common`, including
  Kafka tracing, enrollment, logging, and stream-control helpers. The import
  name is retained to avoid breaking downstream consumers.

Persistence is handled by the separate SamuraiPersistor service. Browser and
API traffic is handled by SamuraiBFF.

## Realtime provider architecture

SamuraiBFF can fan the same accepted audio stream out to one or more configured
`RealtimeASR` peers. Each service owns its model lifecycle and bounded session
state, and returns independent track-labelled events through the same public
contract. A slow or failed peer does not become an extra network hop for the
others.

```mermaid
flowchart LR
    subgraph bffImage["Docker image: samuraibff"]
        subgraph bffService["Compose service: samuraibff"]
            WS["WebSocket audio session"]
            Fanout["bounded per-track fan-out"]
            KafkaOnce["publish audio.raw once"]
        end
    end

    subgraph fasterImage["Docker image: xamurai-rtservice"]
        subgraph fasterService["Compose service: rtservice"]
            FasterAPI["RealtimeASR\nGetCapabilities + Stream"]
            SpeechProvider["internal Python SpeechProvider"]
            Faster["Systran/faster-whisper-medium"]
            FasterDiar["pyannote/speaker-diarization-3.1\noptional enrolled-speaker embedding"]
            FasterJoin["word timestamps + speaker-turn join"]
        end
    end

    subgraph qwenImage["Docker image: xamurai-qwen-rtservice"]
        subgraph qwenService["Compose service: qwen-rtservice"]
            QwenAPI["RealtimeASR\nGetCapabilities + Stream"]
            QwenBackend["Qwen3-ASR-0.6B\nnative vLLM streaming"]
            QwenAligner["Qwen3-ForcedAligner-0.6B\nper completed epoch"]
            QwenDiar["pyannote/speaker-diarization-3.1\nper completed epoch"]
            QwenJoin["aligned word + speaker-turn join"]
        end
    end

    subgraph nemotronImage["Docker image: xamurai-nemotron-rtservice"]
        subgraph nemotronService["Compose service: nemotron-rtservice"]
            NemotronAPI["RealtimeASR\nGetCapabilities + Stream"]
            NemotronNative["Nemotron 3.5 ASR 0.6B Q8\nNeMo-Speech.cpp cache stream"]
        end
    end

    WS --> Fanout
    WS --> KafkaOnce
    Fanout -->|"track=faster-whisper"| FasterAPI
    Fanout -->|"track=qwen"| QwenAPI
    Fanout -->|"track=nemotron"| NemotronAPI
    FasterAPI --> SpeechProvider --> Faster --> FasterJoin
    FasterAPI -->|"FINAL audio"| FasterDiar --> FasterJoin
    QwenAPI --> QwenBackend --> QwenAligner --> QwenJoin
    QwenAPI -->|"epoch PCM"| QwenDiar --> QwenJoin
    NemotronAPI --> NemotronNative
```

The focused [modular realtime ASR guide](docs/modular-asr-providers.md) defines
the common contract, fixed provider profiles, timing semantics, failure
fallbacks, and resource bounds.

## Model pipelines

| Service | Stage and interface | ASR and alignment models | Speaker and speech models | Output behavior |
| --- | --- | --- | --- | --- |
| `rtservice` | Realtime gRPC | Fixed `Systran/faster-whisper-medium` profile through Faster-Whisper/CTranslate2 | `pyannote/speaker-diarization-3.1`; optional `pyannote/embedding` enrollment mapping and Silero VAD | Replaceable partials; contextual, timestamp-owned, speaker-labelled finals |
| `qwen_rtservice` | Realtime gRPC | Fixed `Qwen/Qwen3-ASR-0.6B` plus `Qwen/Qwen3-ForcedAligner-0.6B` | Fixed `pyannote/speaker-diarization-3.1` pipeline | Native-streaming partials; aligned and diarized epoch finals for supported alignment languages, with a coarse speakerless fallback |
| `nemotron_rtservice` | Realtime gRPC | Fixed `nvidia/nemotron-3.5-asr-streaming-0.6b` Q8 GGUF through NeMo-Speech.cpp | Optional Sortformer v2 Q8; WeSpeaker ONNX for S3 enrollment | Native partials; optionally timed, speaker-labelled finals; no public word-timestamp claim |
| `whisperx_worker` | Kafka refinement | WhisperX `medium` by default (`WHISPERX_MODEL` is operator-configurable) plus its language-specific alignment model | `pyannote/speaker-diarization-3.1` by default; optional `pyannote/embedding` enrollment mapping | Speaker-aware refined windows while the session is in progress |
| `finalizer_worker` | Kafka completed-recording pipeline | Reuses the WhisperX ASR and language-alignment pipeline | Reuses the pyannote diarization and optional enrollment pipeline | Canonical full-session transcript with word timing where alignment succeeds |
| `recorder_worker` | Kafka and object storage | None | None | Model-free session WAV assembly and completion notification |

Realtime clients select configured track IDs, not arbitrary model names,
revisions, URLs, or runtime arguments. Operators add another implementation by
deploying a service that satisfies `RealtimeASR` and registering its internal
endpoint with SamuraiBFF.

## Processing pipeline

1. SamuraiBFF publishes each PCM16 `AudioChunk` once to `audio.raw` and fans it
   out to each configured realtime service.
2. Each realtime service streams track-labelled `AsrEvent` updates back over
   the common gRPC contract.
3. `whisperx_worker` buffers configurable windows and publishes
   `RefinedEvent` messages to `transcripts.refined`.
4. `recorder_worker` finalizes the session recording and publishes
   `RecordingFinished` to `recordings.finished`.
5. `finalizer_worker` transcribes the completed recording and publishes
   `SessionTranscript` to `transcripts.final`.

Kafka messages propagate W3C `traceparent` headers. Services also attach the
session and tenant identifiers needed for correlation and isolation.

## Repository layout

- `proto/` and `proto_gen/`: protobuf contract and generated Python bindings
- `rtservice/`: realtime gRPC service
- `qwen_rtservice/`: isolated Qwen3-ASR/vLLM realtime service
- `nemotron_rtservice/`: isolated Nemotron/NeMo-Speech.cpp realtime service
- `whisperx_worker/`: near-realtime refinement worker
- `recorder_worker/`: session recording worker
- `finalizer_worker/`: full-session finalization worker
- `shared/`: shared Python package
- `tests/`: unit and integration tests
- `utilities/`: service-level development and diagnostic helpers
- `docs/`: service behavior, performance, and interface documentation

## Development

The checked-in Conda specifications use Xamurai environment names:

```powershell
conda env create -f environment.yml
conda env create -f rtservice/environment.yml
conda env create -f whisperx_worker/environment.yml
conda env create -f recorder_worker/environment.yml
conda env create -f finalizer_worker/environment.yml
```

Heavy ML tests require the environment matching the service under test.
`HF_TOKEN` is required for gated pyannote models. Grant it only the minimum
model-read permissions required by the test.

### Generate protobuf bindings

From the repository root:

```bash
python -m grpc_tools.protoc \
  -I proto \
  --python_out=proto_gen \
  --grpc_python_out=proto_gen \
  proto/stream.proto
```

Generated bindings must be committed together with any contract change.

## Tests

Install the lightweight CI dependencies and run the unit suite:

```bash
python -m pip install -r requirements.ci.txt
pytest -q -m "not integration" \
  tests/test_diarization_assign_unit.py \
  tests/test_enrollment_cache_unit.py \
  tests/test_rtservice_decode_config_unit.py \
  tests/test_rtservice_final_segmentation_unit.py \
  tests/test_rtservice_session_isolation_unit.py \
  tests/test_speech_provider_integration.py \
  tests/test_qwen_rtservice_integration.py \
  tests/test_nemotron_rtservice_integration.py \
  tests/test_stream_controls_unit.py
```

The integration requirements are split by ML stack:

```bash
python -m pip install -r requirements.integration.fastwhisper.txt
pytest -q -m integration tests/test_realtime_asr_grpc.py

python -m pip install -r requirements.integration.whisperx.txt
pytest -q -m integration tests/test_whisperx_worker_integration.py
```

Kafka and object-storage integration tests use Testcontainers and therefore
require a working Docker daemon. The canonical audio fixture is
`tests/data/test_cs.wav`; its provenance and publication consent are documented
in `tests/data/README.md`.

End-to-end stack smoke tests live with their owning environments:

- public Community Edition smoke tests: `nanosamurai/nanosamurai`
- private deployment smoke tests: the deployment repository

## Build service images

Each service has a standalone Dockerfile:

```bash
docker build -t xamurai-rtservice:dev -f rtservice/Dockerfile .
docker build -t xamurai-whisperx-worker:dev -f whisperx_worker/Dockerfile .
docker build -t xamurai-recorder-worker:dev -f recorder_worker/Dockerfile .
docker build -t xamurai-finalizer-worker:dev -f finalizer_worker/Dockerfile .
docker build -t xamurai-qwen-rtservice:dev -f qwen_rtservice/Dockerfile .
docker build -t xamurai-nemotron-rtservice:dev -f nemotron_rtservice/Dockerfile .
```

The GitHub image workflow publishes immutable `sha-<git-sha>` tags to:

- `ghcr.io/nanosamurai/xamurai-rtservice`
- `ghcr.io/nanosamurai/xamurai-whisperx-worker`
- `ghcr.io/nanosamurai/xamurai-recorder-worker`
- `ghcr.io/nanosamurai/xamurai-finalizer-worker`
- `ghcr.io/nanosamurai/xamurai-qwen-rtservice`
- `ghcr.io/nanosamurai/xamurai-nemotron-rtservice`

## Configuration

Shared Kafka settings:

- `KAFKA_BOOTSTRAP`
- `KAFKA_TOPIC_AUDIO` (default `audio.raw`)
- `LOG_LEVEL`
- standard OpenTelemetry exporter variables

Realtime service settings include:

- `RT_GRPC_PORT` (default `50052`)
- `RT_SERVING_MAX_SESSIONS` (default `1`; atomic per-process stream admission)
- `HF_TOKEN`
- `RT_WINDOW_SEC` (committed FINAL interval), `RT_OVERLAP_SEC` (left/right
  decoding context), and `RT_EMIT_EVERY_SEC`
- `RT_PARTIAL_ENABLE`, `RT_PARTIAL_MODE`, and overload guardrails
- `RT_ASR_TEMPERATURES` and `RT_ASR_COMPRESSION_RATIO_THRESHOLD`
- `RT_FINAL_DIAR_MERGE_GAP_SEC` and `RT_FINAL_DIAR_MIN_TRANSCRIBE_SEC`
- `ENROLL_BACKEND` plus local or S3 enrollment settings

Qwen realtime service settings include:

- `QWEN_RTSERVICE_PORT` (default `50052`)
- `HF_TOKEN` (required, least-privilege access to the fixed gated pyannote model)
- `QWEN_STREAM_CHUNK_SECONDS` and `QWEN_STREAM_EPOCH_SECONDS`
- `QWEN_EPOCH_CONTEXT_CHARACTERS`, `QWEN_MAX_NEW_TOKENS`, and
  `QWEN_MAX_MODEL_LEN`
- `QWEN_KV_CACHE_MIB` (an explicit vLLM KV-cache allocation rather than a
  fraction of all available VRAM)

Nemotron realtime service settings include:

- `NEMOTRON_RTSERVICE_PORT` (default `50052`)
- `NEMOTRON_GPU` (default `0`)
- `NEMOTRON_RNNT_RIGHT_CONTEXT` (default `1`, about 160 ms)
- `NEMOTRON_DIARIZATION` (default `false`); enable Sortformer finals, optionally
  with `ENROLL_BACKEND=s3_manifest` for enrolled names. See the
  [Sortformer and enrollment guide](docs/nemotron-sortformer.md) for the
  four-speaker limit, S3 scope, thresholds, artifact pins and local validation.
- `RT_SERVING_MAX_SESSIONS` (default `1`; values above one enable native
  cross-session microbatching with the same number of independent cache slots)

Nemotron uses NeMo-Speech.cpp's native token-silence endpointing (800 ms) so
each utterance is committed independently. A 30-second forced endpoint is the
hard backstop for uninterrupted speech; this bounds replaceable partial text
without limiting the total stream duration. The adapter enables the native
automatic-punctuation request gate so Nemotron 3.5's own punctuation and
capitalization are retained; no separate PnC model is loaded.

Worker settings include:

- `WHISPERX_SLICE_SECONDS`, `WHISPERX_IDLE_SECONDS`, and diarization guardrails
- `RECORDING_STORAGE_BACKEND` (`local` or `s3`)
- `RECORDING_DIR` or the corresponding S3 bucket, region, endpoint, and prefix
- topic and consumer-group overrides for each worker

Per-stream output selection and compute controls are documented in
[`docs/stream-output-selection.md`](docs/stream-output-selection.md).
Provider profiles and the native Qwen path are documented in
[`docs/modular-asr-providers.md`](docs/modular-asr-providers.md).

## Security

- Treat Kafka headers and gRPC metadata as untrusted input.
- Enforce tenant quotas and clamp compute-amplifying stream controls at the
  authenticated API boundary.
- Never commit tokens, recordings, transcripts, enrollment samples, or
  customer data.
- Prefer workload identity over static object-storage credentials.
- Pyannote telemetry is force-disabled to prevent unexpected outbound metric
  export.
- Bind diagnostic endpoints to localhost for local development. Any broader
  exposure is an explicit deployment decision and must have access controls.

Report vulnerabilities privately as described in
[`SECURITY.md`](SECURITY.md).

## Additional documentation

- [`docs/stream-output-selection.md`](docs/stream-output-selection.md)
- [`docs/whisperx-worker-diarization.md`](docs/whisperx-worker-diarization.md)
- [`docs/rtservice-performance.md`](docs/rtservice-performance.md)
- [`docs/modular-asr-providers.md`](docs/modular-asr-providers.md)
- [`docs/enrolled-speakers-multi-tenant.md`](docs/enrolled-speakers-multi-tenant.md)
- [`docs/word-level-timing.md`](docs/word-level-timing.md)
- [`docs/ci-cd.md`](docs/ci-cd.md)

## License

Licensed under Apache License 2.0. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
