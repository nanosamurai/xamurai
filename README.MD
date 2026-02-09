# drsynth

Speach-to-text and speaker diarization service that provides both real-time transcription as well as near-real-time refinement. It also persists the recording session and performs a final transcript when the session finishes.

Streaming audio → real-time transcript updates → session recording → session finalization (full-session transcript + optional alignment) → persistence (Postgres) + API (BFF).

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

   Persistence details:
   * Postgres persistence is handled by the separate **samuraipersistor** service.


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
  Speakers' voice samples for diarization - currently is used only by rtservice, to be further standardized in DB (per tenant/user) and used by also the other workers..

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

## Local development

### Conda envs (current)

Each subsystem has its own conda env. **To run the full test suite**, run the relevant tests
in the matching env (don’t expect a single env to run everything).

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

* `drsynth`

  * shared package work (`drsynth_common`)
  * protobuf generation
  * running utilities/WIP scripts in `./utilities`
  * general tooling

* `drsynth-bff`

  * FastAPI / Uvicorn
  * BFF runtime deps

* `drsynth-rtservice`

  * gRPC server + realtime streaming deps
  * Kafka client deps

* `drsynth-whisperx`

  * WhisperX + Torch + heavy GPU/audio stack
  * used for running `whisperx_worker`

* `drsynth-finalizer`

  * currently identical to `drsynth-whisperx` (candidate to consolidate)
  * used for running `finalizer_worker`

* `drsynth-recorder`

  * Kafka + WAV writing + optional S3
  * used for running `recorder_worker`

Consolidation suggestion: `drsynth-whisperx` and `drsynth-finalizer` can usually be merged into one “GPU/heavy” env because both import Torch/WhisperX.

### Editable installs (recommended)

From repo root, installing packages editable avoids most `PYTHONPATH` issues:

* `pip install -e shared`
* `pip install -e bff`
* `pip install -e rtservice`
* `pip install -e whisperx_worker`
* `pip install -e recorder_worker`
* `pip install -e finalizer_worker`

### Windows + `PYTHONPATH` (if we don’t use editable installs)

PowerShell example:

* `$env:PYTHONPATH = ".;shared\src;rtservice\src;bff\src"`

If we want `PYTHONPATH` persistent per conda env:

* `conda env config vars set PYTHONPATH=".;shared\src;rtservice\src;bff\src;recorder_worker\src;finalizer_worker\src;whisperx_worker\src"`
* `conda deactivate`
* `conda activate <env>`

---

## Configuration (environment variables)

### Kafka (shared across services)

* `KAFKA_BOOTSTRAP` (default: `localhost:9092`)
* `KAFKA_TOPIC_AUDIO` (default: `audio.raw`)

### rtservice

Depends on implementation, but typically:

* `KAFKA_BOOTSTRAP`
* `KAFKA_TOPIC_AUDIO`
* `KAFKA_TOPIC_REFINED` (if rtservice streams partial transcript updates back to clients)
* Service bind config (host/port)

### whisperx_worker

* `KAFKA_TOPIC_REFINED` (default: `transcripts.refined`)
* `WHISPERX_SLICE_SECONDS` (default: `60.0`)
* `WHISPERX_COMPUTE_TYPE` (optional override)
* CUDA if available, else CPU

### recorder_worker

* `KAFKA_TOPIC_RECORDING_FINISHED` (default: `recordings.finished`)
* `RECORDING_STORAGE_BACKEND` (`local` | `s3`)
* `RECORDING_DIR` (default: `recordings`)
* `RECORDER_IDLE_SECONDS` (default: `30.0`)
* S3 settings (if backend is `s3`):

  * `RECORDING_S3_BUCKET`
  * `RECORDING_S3_PREFIX`

### finalizer_worker

* Consumes: `KAFKA_TOPIC_RECORDING_FINISHED`
* Produces: `transcripts.final` (`SessionTranscript`) via `KAFKA_TOPIC_TRANSCRIPTS_FINAL`
* `TRANSCRIPTS_DIR` (local JSON artifacts dir)
* `DATABASE_URL` (required if you want final transcript persistence enabled)

Notes:
* DB schema expects `tenant_id` to be a UUID. For resilience in dev/test, the finalizer falls back to the zero UUID when tenant_id is missing/invalid.

### Postgres

* `DATABASE_URL`
  Example:

  * `postgresql+psycopg://drsynth:drsynth@localhost:5432/drsynth`

---

## Dependencies

### Core runtime deps (typical)

* `confluent-kafka`
* `protobuf`
* `numpy`
* `fastapi`, `uvicorn` (BFF)
* gRPC stack for rtservice:

  * `grpcio`, `grpcio-tools`
  * `protobuf` pinned consistently

### WhisperX stack (heavy)

* `torch`
* `whisperx`
* and transitive deps (pyannote, etc.)

### Testing

* `pytest`
* `testcontainers`
* `pytest-timeout` (optional; we use the mark)

---

## Postgres (local via Docker Compose)

Example `docker-compose.postgres.yml`:

services:

* postgres:

  * image: `postgres:16`
  * env:

    * `POSTGRES_DB=drsynth`
    * `POSTGRES_USER=drsynth`
    * `POSTGRES_PASSWORD=drsynth`
  * ports:

    * `5432:5432`
  * volume:

    * `drsynth_pg:/var/lib/postgresql/data`

volumes:

* `drsynth_pg`

Run:

* `docker compose -f docker-compose.postgres.yml up -d`

---

## DB migrations

DB migrations are managed in the **samuraipersistor** repo (Migratus + raw SQL).

---

## Running services

### rtservice (gRPC RealtimeASR)

Entrypoint depends on our `rtservice` implementation, common patterns:

* `python -m rtservice.server`
  or
* `python rtservice/src/rtservice/server.py`

### BFF

PowerShell:

* `$env:PYTHONPATH=".;shared\src;bff\src"`
* `uvicorn bff.app:app --host 0.0.0.0 --port 8000`

### Workers

* `python -m whisperx_worker.whisperx_worker`
* `python -m recorder_worker.recorder_worker`
* `python -m finalizer_worker.finalizer_worker`

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
* Windows + Torch DLL issues: Torch import can fail if envs diverge. `drsynth-whisperx` is the “known-good” heavy env.
* `PYTHONPATH` vs editable installs: editable installs (`pip install -e`) are the cleanest way to avoid module import issues (including protobuf modules).
* Cline has issues with running tests - it has to use correct conda env: for instance for running bff-related tests it has to run: `conda activate drsynth-bff && "C:\Users\miros\miniconda3\envs\drsynth-bff\python.exe" -m pytest tests/test_bff_session_persistence.py -v`

---

## Suggested next steps

* Persist session lifecycle (created at start, updated on end) into Postgres.
* Keep Postgres as source-of-truth, optionally index transcripts into OpenSearch later for search.

(Implemented) `finalizer_worker` now persists the finalized transcript into Postgres on session finalization.
* Build DB-backed session APIs in BFF (list sessions, session detail, transcript detail).
* Add diarization/speaker model later (speakers table already anticipated).
