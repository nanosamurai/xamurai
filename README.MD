# Xamurai

Xamurai (voX samurai) provides the speech-to-text and speaker-diarization services used by
the nanosamurai platform. It supports realtime transcription over gRPC and
asynchronous refinement, recording, and finalization over Kafka.

For a complete self-hosted Community Edition stack, use the public
[`nanosamurai/nanosamurai`](https://github.com/nanosamurai/nanosamurai)
repository. Deployment orchestration and environment-specific infrastructure
do not live in this service repository.

## Services

- `rtservice` exposes the bidirectional `RealtimeASR` gRPC API and emits
  replaceable partial and committed final realtime events.
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

## Processing pipeline

1. SamuraiBFF streams PCM16 audio to `rtservice` and publishes `AudioChunk`
   messages to `audio.raw`.
2. `rtservice` streams low-latency `AsrEvent` updates back over gRPC.
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
  tests/test_rtservice_session_isolation_unit.py \
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
```

The GitHub image workflow publishes immutable `sha-<git-sha>` tags to:

- `ghcr.io/nanosamurai/xamurai-rtservice`
- `ghcr.io/nanosamurai/xamurai-whisperx-worker`
- `ghcr.io/nanosamurai/xamurai-recorder-worker`
- `ghcr.io/nanosamurai/xamurai-finalizer-worker`

## Configuration

Shared Kafka settings:

- `KAFKA_BOOTSTRAP`
- `KAFKA_TOPIC_AUDIO` (default `audio.raw`)
- `LOG_LEVEL`
- standard OpenTelemetry exporter variables

Realtime service settings include:

- `RT_GRPC_PORT` (default `50052`)
- `HF_TOKEN`
- `RT_WINDOW_SEC`, `RT_OVERLAP_SEC`, and `RT_EMIT_EVERY_SEC`
- `RT_PARTIAL_ENABLE`, `RT_PARTIAL_MODE`, and overload guardrails
- `ENROLL_BACKEND` plus local or S3 enrollment settings

Worker settings include:

- `WHISPERX_SLICE_SECONDS`, `WHISPERX_IDLE_SECONDS`, and diarization guardrails
- `RECORDING_STORAGE_BACKEND` (`local` or `s3`)
- `RECORDING_DIR` or the corresponding S3 bucket, region, endpoint, and prefix
- topic and consumer-group overrides for each worker

Per-stream output selection and compute controls are documented in
[`docs/stream-output-selection.md`](docs/stream-output-selection.md).

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
- [`docs/enrolled-speakers-multi-tenant.md`](docs/enrolled-speakers-multi-tenant.md)
- [`docs/word-level-timing.md`](docs/word-level-timing.md)
- [`docs/ci-cd.md`](docs/ci-cd.md)

## License

Licensed under Apache License 2.0. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
