# drsynth

Speach-to-text and speaker diarization service that provides both real-time transcription as well as near-real-time refinement. It also persists the recording session and performs a final transcript when the session finishes.

Streaming audio → real-time transcript updates → session recording → session finalization (full-session transcript + optional alignment).

This repo contains services that respond via gRPC and/or consume/produce on dedicated Kafka topics. Database persistence is handled by a separate service (**samuraipersistor**), which consumes from all relevant Kafka topics (`transcripts.refined`, `transcripts.final`) and stores the data in Postgres.

The architecture is planned to support multiple layers of "refinement" workers (such as `whisperx_worker`) that might refine different windows of the audio (e.g., 60secs, 300secs) using different models (WhisperX, Voxstral, etc.)).

This repo is a Python monorepo with multiple packages (realtime service + workers) and a shared library (`drsynth_common`). The system is Kafka-centric for dataflow between services, while realtime client interaction is done via gRPC (`RealtimeASR`).

---

## High-level architecture

### Kafka topics (current)

* `audio.raw`
  Raw audio stream (`AudioChunk`). Produced by realtime ingress and consumed by:

  * `whisperx_worker` (real-time slicing)
  * `recorder_worker` (records full-session WAV)

* `transcripts.refined`
  Real-time segment events (`RefinedEvent`) from `whisperx_worker` (per slice).

* `recordings.finished`
  Emitted by `recorder_worker` when a session becomes idle/ends and WAV is finalized.

* `transcripts.final`
  Emitted by `finalizer_worker` as the canonical finished transcript event (`SessionTranscript`).

  Persisted to Postgres by **samuraipersistor** (separate service) into
  `sessions`, `recordings`, and `session_transcripts`.

  Note: **samuraipersistor** consumes from both `transcripts.refined` and `transcripts.final` topics to provide DB persistence.

### Processing pipeline

1. **BFF** serves UI/API needs (sessions list/detail, transcript retrieval, future auth/tenant/user logic).

   * Session creation is expected to happen at session start (BFF).
   * UI in the browser streams sound via Web Socket, BFF then stores the chunks onto Kafka for further processing by workers, while also using rtservice for realtime transcription (currently not stored into DB, just sent back to UI)
2. **rtservice** exposes the gRPC service **`RealtimeASR`** and acts as the realtime ingestion layer.

   * It receives streaming audio frames (PCM16) from clients and publishes `AudioChunk` messages to Kafka (`audio.raw`).
   * It can also subscribe to `transcripts.refined` and stream partial results back to the client (depending on our current implementation).

3. **whisperx_worker** consumes `audio.raw`, buffers per session, and every `WHISPERX_SLICE_SECONDS` emits `RefinedEvent` segments to `transcripts.refined`.

4. **recorder_worker** consumes `audio.raw`, writes the full session WAV, and after idle timeout emits `RecordingFinished` to `recordings.finished`.

5. **finalizer_worker** consumes `recordings.finished`, runs full-session WhisperX (+ optional alignment) on the finalized WAV, stores a JSON transcript artifact, and emits a `SessionTranscript` to `transcripts.final`.

---

## Repo structure

* (moved) BFF/API lives in the **samuraibff** repo.

* `rtservice/`
  Realtime (5 sec chunks) ingestion service (gRPC `RealtimeASR` ) - uses fast whisper and separate diarization.

* `whisperx_worker/`
  Almost real-time (60 sec chunks) slice transcription worker (using WhisperX).

* `recorder_worker/`
  Session WAV recorder + emits `recordings.finished`.

* `finalizer_worker/`
  Full-session transcription + JSON artifact + emits `transcripts.final`.

* `shared/`
  Shared package: `drsynth_common` (IDs, DB helpers, Kafka helpers, config, shared utilities).

* (moved) DB migrations live in the **samuraipersistor** repo.

* `tests/`
  Integration tests (Kafka via testcontainers) + test data.

* `utilities/`
  Local scripts / WIP helpers (run under the `drsynth` env).

* `enrolled_speaklers/`
  Speakers' voice samples for diarization - currently is used only by rtservice, to be further standardized in DB (per tenant/user) and used by also the other workers.

---

## Protobufs & gRPC (`stream.proto`)

Workers and services communicate using protobuf messages (`AudioChunk`, `RefinedEvent`, `RecordingFinished`, `SessionTranscript`, …).

In addition, realtime ingress uses the gRPC service **`RealtimeASR`** (defined in the proto). The intended flow is:

* client streams audio frames in
* server streams partial transcript updates out (or emits updates separately)

### Generating Python code

We generally use one of these approaches:

1. Generate into `proto_gen` folder - in `./proto_gen`, run:

* `python -m grpc_tools.protoc -I ../proto --python_out=. --grpc_python_out=. ../proto/stream.proto`

---

## Local development (conda)

### Conda envs (current)

Each subsystem has its own conda env. **To run the full test suite**, run the relevant tests
in the matching env (don't expect a single env to run everything).

Test commands (Windows):

* Recorder/Kafka integration (lightweight):
  * `conda run -n drsynth-bff python -m pytest -q tests/test_recorder_worker_integration.py`

* rtservice gRPC integration (heavy; requires `HF_TOKEN`):
  * `conda run -n drsynth-rtservice python -m pytest -q tests/test_realtime_asr_grpc.py`

* WhisperX worker tests (GPU/heavy):
  * `conda run -n drsynth-whisperx python -m pytest -q tests/test_whisperx_worker_unit.py tests/test_whisperx_worker_integration.py`

* Finalizer worker integration (GPU/heavy):
  * Prefer running it in `drsynth-whisperx` (known-good Torch/WhisperX stack):
    * `conda run -n drsynth-whisperx python -m pytest -q tests/test_finalizer_worker_integration.py`

Notes:
* Kafka integration tests use **Testcontainers Kafka** by default. To run against a real Kafka,
  set `KAFKA_BOOTSTRAP=localhost:9092`.
* `HF_TOKEN` must be visible in the conda env for `drsynth-rtservice` tests.

---

## Docker (k8s-friendly)

This repo is intended to run on Kubernetes. Each service has its own **Dockerfile** (built with **venv + pip**, independent from the conda setups).

### Build images

From repo root:

```bash
docker build -t drsynth-rtservice:dev -f rtservice/Dockerfile .
docker build -t drsynth-whisperx-worker:dev -f whisperx_worker/Dockerfile .
docker build -t drsynth-recorder-worker:dev -f recorder_worker/Dockerfile .
docker build -t drsynth-finalizer-worker:dev -f finalizer_worker/Dockerfile .
```

Note: WhisperX pins `torch~=2.8.0`, so the Dockerfiles align Torch/Torchaudio to `2.8.0+cu128` to avoid resolver downgrades/conflicts.

### Run locally with docker compose

Prereqs:
- Docker installed
- `HF_TOKEN` exported in your environment (needed by `rtservice` / pyannote)

```bash
# Linux/macOS
export HF_TOKEN=... 

# Windows PowerShell
# $env:HF_TOKEN = "..."

docker compose up --build
```

The compose file starts:
- Kafka broker
- Postgres (currently unused by this repo directly; persistence is handled by samuraipersistor)
- rtservice gRPC server on `localhost:50052`
- whisperx_worker, recorder_worker, finalizer_worker

---

## Configuration (environment variables)

### Kafka (shared across services)

* `KAFKA_BOOTSTRAP` (default in code: `localhost:9092`; in compose: `broker:9092`)
* `KAFKA_TOPIC_AUDIO` (default: `audio.raw`)

### rtservice

* `RT_GRPC_PORT` (default: `50052`)
* `HF_TOKEN` (**required**) – used to download pyannote models
* `FW_LANG_DEFAULT` (optional) – default ISO language code
* `ENROLL_DIR` (default: `enrolled_speakers`) – directory with speaker enrollment WAVs

### whisperx_worker

* `KAFKA_BOOTSTRAP`
* `KAFKA_TOPIC_AUDIO` (default: `audio.raw`)
* `KAFKA_TOPIC_REFINED` (default: `transcripts.refined`)
* `KAFKA_GROUP_ID` (default: `whisperx-async`)
* `WHISPERX_SLICE_SECONDS` (default: `60.0`)
* `WHISPERX_IDLE_SECONDS` (default: `30.0`)
* `WHISPERX_COMPUTE_TYPE` (optional)

### recorder_worker

* `KAFKA_BOOTSTRAP`
* `KAFKA_TOPIC_AUDIO` (default: `audio.raw`)
* `KAFKA_TOPIC_RECORDING_FINISHED` (default: `recordings.finished`)
* `KAFKA_GROUP_ID_RECORDER` (default: `recorder-worker`)
* `RECORDER_IDLE_SECONDS` (default: `30.0`)

Storage:

* `RECORDING_STORAGE_BACKEND` (`local` | `s3`, default: `local`)
* `RECORDING_DIR` (default in code: `recordings`; in containers set it to `/data/recordings`)

S3 (used when `RECORDING_STORAGE_BACKEND=s3`):

* `S3_ENDPOINT` (optional; empty means AWS S3)
* `S3_BUCKET` (**required**)
* `S3_REGION` (optional)
* `S3_ACCESS_KEY` / `S3_SECRET_KEY` (optional)
  * If these are **not** set, boto3 falls back to its default credential chain (e.g. IRSA/Pod Identity/instance role)
* `S3_FORCE_PATH_STYLE` (default: `true`) – helps compatibility with S3-like storage (Ceph/MinIO)
* `S3_PREFIX` (optional) – key prefix for uploaded recordings

### finalizer_worker

* `KAFKA_BOOTSTRAP`
* `KAFKA_TOPIC_RECORDING_FINISHED` (default: `recordings.finished`)
* `KAFKA_TOPIC_TRANSCRIPTS_FINAL` (default: `transcripts.final`)
* `KAFKA_GROUP_ID_FINALIZER` (default: `finalizer-worker`)

Important:
- finalizer currently supports only `file://` recording URLs
- therefore it needs access to the **same recordings volume** as `recorder_worker`

---

## Kubernetes notes

### Volumes

If you use `RECORDING_STORAGE_BACKEND=local`, both `recorder_worker` and `finalizer_worker` must mount the same PVC at the same path (recommended: `/data/recordings`).

### GPU scheduling

`rtservice`, `whisperx_worker`, and `finalizer_worker` are designed to use CUDA if available.

In Kubernetes, schedule them on GPU nodes (example):

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

### Health checks

- `rtservice` is gRPC-only; use a `tcpSocket` probe on port `50052` (or your `RT_GRPC_PORT`).
- workers are long-running Kafka consumers; liveness is typically based on the process staying up.

---

## Tests

### Integration tests

Integration tests run Kafka via Testcontainers and execute real worker codepaths.

* `pytest -m integration -o log_cli=true --log-cli-level=DEBUG`

Key test data:

* `tests/data/test_cs.wav` (canonical sample audio)

---

## Known gotchas

* Topic creation timing: we should not rely on Kafka auto-create in tests; topics should be created in fixtures.
* Windows + Torch DLL issues: Torch import can fail if envs diverge. `drsynth-whisperx` is the "known-good" heavy env.
* `PYTHONPATH` vs editable installs: editable installs (`pip install -e`) are the cleanest way to avoid module import issues (including protobuf modules).
* Cline has issues with running tests - it has to use correct conda env: for instance for running bff-related tests it has to run: `conda activate drsynth-bff && "C:\Users\miros\miniconda3\envs\drsynth-bff\python.exe" -m pytest tests/test_bff_session_persistence.py -v`

---

## Suggested next steps

* Build DB-backed session APIs in BFF (list sessions, session detail, transcript detail).
* Add diarization/speaker model later (speakers table already anticipated).
