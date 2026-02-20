 # Enrolled speakers in a multi-tenant environment (drsynth)

Status: **WIP / In progress**

This document describes the target architecture and implementation plan for supporting **tenant-specific enrolled speakers** across the transcription stack:

- `rtservice` (realtime diarization + ASR)
- `whisperx_worker` (near-realtime refinement)
- `finalizer_worker` (final transcript)

It is intended to be updated progressively as phases are implemented.

---

## Problem statement

Today, enrolled speakers are loaded from a local folder (`enrolled_speakers/`) and used **only** by `rtservice`.

In a multi-tenant environment we need:

1. A tenant-specific enrolled speaker set
2. A consistent way for all services to load those enrolled speakers
3. Storage in object storage (S3 or S3-compatible)
4. Correctness under concurrency (multiple sessions/tenants at once)
5. HA-friendly behavior (multiple service replicas)

---

## Key decisions

- **Speaker set scope:** all speakers per tenant (for now)
- **Speaker label in outputs:** human-readable `label` (e.g. "Dr Novak")
- **Consistency requirements:** eventual consistency is OK; cache TTL ~ 5 minutes
- **Storage:** **single shared bucket**, with **per-tenant prefixes**
- **Enrollment manifest:** **one manifest per speaker** (`speaker.json`), optional tenant `index.json`

---

## Current state (baseline)

- `rtservice` loads enrolled speakers at process startup from `ENROLL_DIR` and keeps them in memory.
- `whisperx_worker` and `finalizer_worker` do not diarize and do not label speakers.
- `tenant_id` is already present on protobuf messages:
  - `AudioChunk.tenant_id`
  - `RefinedEvent.tenant_id`
  - `RecordingFinished.tenant_id`
  - `SessionTranscript.tenant_id`

Known issue:
- `rtservice` uses a single rolling buffer shared across all sessions, which breaks correctness for concurrent sessions.

---

## Target architecture overview

### Storage layout (S3)

We store enrollment data under a tenant prefix:

```
 s3://<bucket>/<enrollment_prefix>/<tenant_id>/speakers/
   <speaker_id>/
     speaker.json
     samples/
       <sample_id>.wav
```

Optional (recommended) listing index:

```
 s3://<bucket>/<enrollment_prefix>/<tenant_id>/speakers/index.json
```


### Runtime separation of concerns

1) **Global models (process-level):** loaded once per process/pod
- Whisper / WhisperX
- pyannote diarization pipeline
- pyannote embedding model
- optional VAD

2) **Tenant enrollment cache (tenant-level):** loaded on demand per tenant
- `tenant_id -> {label -> embedding}`
- cached with TTL + max-size (LRU)
- reads S3 manifests + WAV samples

3) **Session state (session-level, realtime only):** kept in-memory per active session
- rolling buffer, offsets, dedupe keys

---

## Implementation phases

### Phase 1 — shared enrollment module (this repo)

Deliverables:
- `shared/src/drsynth_common/enrollment/` module with:
  - per-speaker manifest model (`speaker.json`)
  - provider interface
  - local provider (for dev)
  - S3 provider (manifest + samples)
  - TTL+LRU enrollment cache
- unit tests

Status:
- [x] Started
- [x] Completed

Implemented in:
- `shared/src/drsynth_common/enrollment/`
- `tests/test_enrollment_cache_unit.py`

---

### Phase 2 — rtservice refactor for correctness + tenant enrollment

Deliverables:
- refactor `RealtimeEngine` into:
  - global `ModelBundle`
  - per-session `SessionState`
- pass `tenant_id` into realtime processing
- map diarization speakers to enrolled speakers per tenant
- integration tests: concurrent sessions do not contaminate each other

Status:
- [x] Started
- [x] Completed

Implemented in:
- `rtservice/src/rtservice/engine.py`
- `rtservice/src/rtservice/server.py`
- `tests/test_rtservice_session_isolation_unit.py`

Notes:
- The engine now keys per-session state by `(tenant_id, session_id)` to avoid collisions.
- `AudioChunk.tenant_id` is now passed from gRPC into the engine.

---

### Phase 3 — whisperx_worker diarization + tenant enrollment mapping

Deliverables:
- run diarization per slice
- assign speaker labels to ASR segments
- map diar speaker clusters to enrolled speakers via embeddings
- emit `RefinedEvent.speaker`

Status:
- [x] Started
- [x] Completed

Implemented in:
- `whisperx_worker/src/whisperx_worker/whisperx_worker.py`
- `tests/test_whisperx_worker_unit.py`

Notes:
- Diarization is best-effort and requires `HF_TOKEN` (pyannote models). If missing, the worker falls back to speaker="".
- Enrollment mapping supports both legacy local dir (dev) and manifest-based backends via `ENROLL_BACKEND`.
- Dev/testing:
  - lightweight unit tests run in `drsynth-bff` (no WhisperX installed)
  - full integration/E2E should run in `drsynth-whisperx` (or the Docker image)

---

### Phase 4 — finalizer_worker diarization + tenant enrollment mapping

Deliverables:
- run diarization for full session
- assign speaker labels (word-level optional)
- map diar speaker clusters to enrolled speakers via embeddings
- populate `SessionTranscriptSegment.speaker`

Status:
- [ ] Started
- [ ] Completed

---

### Phase 5 — BFF/Persistor speaker management APIs + schema

Out of scope for this PR series in `drsynth` repo, but required end-to-end.

Deliverables:
- DB schema for multiple samples per speaker
- upload endpoints
- persistor writes
- S3 manifest writer

---

## Notes on HA

- Each Kubernetes pod loads its own model copy into VRAM/RAM.
- Realtime is HA via reconnect; realtime context is not persisted across pods.
- Durable output is provided by Kafka workers (refined + final).
