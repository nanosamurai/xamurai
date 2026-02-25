# Local stack runbook (Docker Compose)

This runbook describes the **fastest local end-to-end workflow**: run the full stack via **Docker Compose**.

If you want the more production-like topology (**Compose infra + Kubernetes apps**), see: **`docs/local-k8s-setup.md`**.

> Repo layout assumption on your machine (defaults used in `.env.example`):
> - `C:/Users/miros/PycharmProjects/xamurai`
> - `C:/Users/miros/IdeaProjects/samuraibff`
> - `C:/Users/miros/IdeaProjects/samuraipersistor`

---

## Prerequisites

- Docker Desktop installed and running
- `HF_TOKEN` available (required by **rtservice** for model downloads)

---

## 1) Create `.env`

From repo root (`xamurai/`):

```bash
copy .env.example .env
```

Edit `.env` and set at least:
- `HF_TOKEN=...`

Optional overrides:
- `SAMURAIBFF_PATH=...`
- `SAMURAIPERSISTOR_PATH=...`

### Security note

By default, `docker-compose.yml` binds published ports to **localhost only** via `COMPOSE_BIND_IP=127.0.0.1`.
If you need services reachable from outside localhost (not recommended), override `COMPOSE_BIND_IP`.

---

## 2) Start the stack (auth disabled by default)

Compose defaults to `SAMURAIBFF_AUTH_REQUIRED=false`.

When auth is disabled, the BFF still needs a **tenant id** so:
- sessions are isolated in memory
- Kafka `AudioChunk.tenant_id` is populated
- enrollment lookup hits the correct S3 prefix

For local dev we use a **guest tenant id** (seeded by `db_seed`):
- `SAMURAIBFF_AUTH_GUEST_TENANT_ID=00000000-0000-0000-0000-000000000000`

This is wired by default in `docker-compose.yml`.

```bash
docker compose up --build
```

> Note: `docker-compose.dev.yml` is deprecated; the full stack is consolidated into `docker-compose.yml`.

What this starts:
- Kafka (listeners for docker network + host)
- **LocalStack (S3)** for enrollment storage (persistent volume)
- Postgres
- DB migrations (`db_migrate`)
- DB seed (`db_seed`, creates the dev tenant row needed by unauth BFF)
- rtservice + workers
- samuraibff
- samuraipersistor

Open:
- BFF/UI: http://localhost:8000
- Persistor health: http://127.0.0.1:8010/health
- LocalStack S3 endpoint: http://localhost:4566

> Windows note: `localhost` may resolve to IPv6 first (`::1`). If something fails, try `127.0.0.1` explicitly.

### Quick E2E verification (speaker labeling)

If you have an enrollment sample in LocalStack, you can validate that
- diarization actually ran, and
- enrollment mapping actually replaced `Speaker_0` with your enrolled label,

using the Tier4 smoke test.

Note: the label comes from the uploaded `speaker.json` (`"label"` field). If your LocalStack data uses
`Miro (cz)` but you want the canonical test label `Miro-cz`, either re-enroll with that label or use
`--expect-speaker-alias` in the smoke test.

Create a small venv (Windows):

```bat
py -m venv .venv-smoke
.venv-smoke\Scripts\pip install -r utilities/k8s_local_smoke_test/requirements.txt -r utilities/k8s_local_smoke_test/requirements.kafka.txt
```

Run Tier4 and require a specific speaker label (with an alias accepted for local data):

```bat
.venv-smoke\Scripts\python utilities/k8s_local_smoke_test/tier4_async_pipeline.py \
  --base-url http://localhost:8000 \
  --wav tests/data/test_cs.wav --lang cs --stream-seconds 6.0 \
  --kafka-bootstrap 127.0.0.1:9092 \
  --timeout 420 --signal final \
  --expect-speaker Miro-cz --expect-speaker-alias "Miro (cz)"
```

If this fails, check logs for `whisperx_worker` and `finalizer_worker`:

```bat
docker compose logs -f whisperx_worker
docker compose logs -f finalizer_worker
```

Important: diarization/enrollment requires **HF_TOKEN** in those workers.

---

## 3) Auth notes (Keycloak)

Compose does **not** start Keycloak by default.

For local testing you can point BFF at any reachable issuer:
- your existing Keycloak in ECS (recommended)
- or a local Keycloak you run separately

To enable auth in BFF, set:

```bat
set SAMURAIBFF_AUTH_REQUIRED=true
set SAMURAIBFF_AUTH_ISSUER=https://<your-issuer>/realms/<realm>
```

When auth is disabled and you are running multi-tenant enrollment, ensure the guest tenant id is set:

```bat
set SAMURAIBFF_AUTH_REQUIRED=false
set SAMURAIBFF_AUTH_GUEST_TENANT_ID=00000000-0000-0000-0000-000000000000
```

> We keep `docker/keycloak/realm-drsynth.json` as a reference realm import if you want to run a local Keycloak.

---

## 4) Enrollment storage (LocalStack S3)

For end-to-end multi-tenant speaker enrollment we run S3 locally using **LocalStack**.

Defaults in `docker-compose.yml`:
- bucket: `xamurai-enrollment`
- prefix: `enrollment/`
- endpoint: `http://localhost:4566`

LocalStack is configured with persistence, so speaker enrollment uploads survive `docker compose down` / up.

To list enrollment objects:
```bash
aws --endpoint-url http://localhost:4566 s3 ls s3://xamurai-enrollment/enrollment --recursive
```

To reset enrollment state completely (destroys persisted LocalStack state):

```bash
docker compose down
docker volume rm drsynth_nanosamurai_localstack
```

---

## 5) Common checks

Kafka topics are created by `kafka_init`.

If a worker logs `Failed to resolve 'broker:29092'`, it usually means the Kafka broker container is not running on the compose network. Start it:

```bash
docker compose up -d broker kafka_init
```

On first run, **rtservice may take several minutes** to download Whisper + Pyannote models.
During that time, BFF `/ready` may return 503 with `grpc.up?=false`.
Check:

```bash
docker compose logs -f rtservice
```

Check service logs:
```bash
docker compose logs -f samuraibff
```

Create a session (unauth mode):
```bash
curl -X POST http://localhost:8000/api/sessions
```

---

## Troubleshooting

### WhisperX worker prints FFmpeg / libavutil warnings

You may see warnings like missing `libavutil.so.58` or torio failing to load FFmpeg extensions.

- The **ffmpeg binary is installed** in the container and WhisperX can still operate.
- These warnings are coming from TorchAudio/TorIO optional FFmpeg extensions looking for different
  `libavutil.so.*` versions than what the base image ships.

If everything else works, treat these as noisy but non-fatal. If we need to eliminate them long-term,
we can either pin to a base image with matching FFmpeg runtime libs or disable the torio extension path.

### Kafka clients in containers fail with `localhost:9092`

Use `docker-compose.yml` (it fixes advertised listeners) and ensure containers use `broker:29092`.

### samuraibff fails inserting sessions due to FK constraint on tenant

Ensure `db_seed` ran; it inserts the dev tenant id `00000000-0000-0000-0000-000000000000`.

### Keycloak redirect_uri errors

Confirm your client has `http://localhost:8000/*` in its redirect URIs.
In this repo, the import file already does.
