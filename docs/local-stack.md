# Local stack runbook (drsynth + samuraibff + samuraipersistor)

This document describes two supported local workflows:

1) **All Docker Compose** (fastest end-to-end)
2) **Docker Compose infra + Minikube apps** (k8s-realistic; supports GPU scheduling)

> Repo layout assumption on your machine (defaults used in `.env.example`):
> - `C:/Users/miros/PycharmProjects/drsynth`
> - `C:/Users/miros/IdeaProjects/samuraibff`
> - `C:/Users/miros/IdeaProjects/samuraipersistor`

---

## Prerequisites

- Docker Desktop installed and running
- `HF_TOKEN` available (needed by **rtservice**)
- Optional for k8s workflow:
  - WSL2 + NVIDIA working (inside WSL: `nvidia-smi`)
  - minikube + kubectl installed

---

## 1) Mode A — Full stack via Docker Compose

### 1.1 Create `.env`

From `drsynth/`:

```bash
copy .env.example .env
```

Edit `.env` and set at least:
- `HF_TOKEN=...`

Optional overrides:
- `SAMURAIBFF_PATH=...`
- `SAMURAIPERSISTOR_PATH=...`

### 1.2 Start the stack (auth disabled)

```bash
docker compose up --build
```

What this starts:
- Kafka (with internal + host + minikube listeners)
- Postgres
- DB migrations (samuraipersistor migratus)
- DB seed (creates the dev tenant row needed by unauth BFF)
- rtservice + workers
- samuraibff
- samuraipersistor

Open:
- BFF/UI: http://localhost:8000
- Persistor health: http://127.0.0.1:8010/health

> Note: on Windows, `localhost` may resolve to IPv6 first (`::1`), and persistor binds IPv4 (`0.0.0.0`).

### 1.3 Auth notes (Keycloak)

Compose does **not** start Keycloak by default.

For local testing you can point BFF at any reachable issuer:
- your existing Keycloak in ECS (recommended)
- or a local Keycloak you run separately

To enable auth in BFF, set:

```bash
set SAMURAIBFF_AUTH_REQUIRED=true
set SAMURAIBFF_AUTH_ISSUER=https://<your-issuer>/realms/<realm>
```

> We keep `docker/keycloak/realm-drsynth.json` as a reference realm import if you want to run a local Keycloak.

### 1.4 Common checks

Kafka topics were created by `kafka_init`.

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

## 2) Mode B — Docker Compose infra + Minikube apps

This runs **Kafka + Postgres** in Docker Compose and deploys app services to minikube using the Helm chart in `charts/drsynth-stack`.

> Keycloak is expected to be reachable externally (your ECS dev profile is fine). Compose does not start it by default.

### 2.1 Start infra only

```bash
docker compose up -d broker kafka_init postgres persistor_migrate db_seed
```

#### Kafka ports and why we expose two of them

Kafka in this stack is reachable from three “worlds”, each needing a different advertised address:

- **Other Docker containers** (compose network): `broker:29092`
- **Host machine tools** (Windows/macOS/Linux): `localhost:9092`
- **Minikube pods**: `host.minikube.internal:39092`

We keep `39092` specifically for minikube because Kafka’s advertised listeners must be different for
host vs. pod networking. Trying to reuse `localhost:9092` inside pods will fail.

If you also want local Keycloak, run it outside this compose (or re-add it as a compose profile later).

Infra endpoints (as seen from minikube pods):
- Kafka: `host.minikube.internal:39092`
- Postgres: `host.minikube.internal:5432`

> We expose Kafka on `39092` specifically so that **minikube pods** can reach it,
> while still keeping `localhost:9092` working for host tools.

### 2.2 Build images and make them available to minikube

You have two common options:

#### Option A (simple): build normally + `minikube image load`

Build images on the host Docker engine:
```bash
# drsynth images
docker build -t drsynth-rtservice:dev -f rtservice/Dockerfile .
docker build -t drsynth-whisperx-worker:dev -f whisperx_worker/Dockerfile .
docker build -t drsynth-recorder-worker:dev -f recorder_worker/Dockerfile .
docker build -t drsynth-finalizer-worker:dev -f finalizer_worker/Dockerfile .

# in the other repos:
# samuraibff: docker build -t samuraibff:local .
# samuraipersistor: docker build -t samuraipersistor:local .
```

Then load into minikube:
```bash
minikube image load drsynth-rtservice:dev
minikube image load drsynth-whisperx-worker:dev
minikube image load drsynth-recorder-worker:dev
minikube image load drsynth-finalizer-worker:dev
minikube image load samuraibff:local
minikube image load samuraipersistor:local
```

#### Option B: build directly into minikube’s Docker daemon

```bash
minikube -p minikube docker-env
# then follow the printed instructions (PowerShell vs bash differs)
```

Then run `docker build ...` and images will already be present in minikube.

### 2.3 Create the recordings host path in minikube

The chart uses a simple hostPath PV by default (`/data/drsynth-recordings`). Create it:

```bash
minikube ssh -- "sudo mkdir -p /data/drsynth-recordings && sudo chmod -R 777 /data/drsynth-recordings"
```

### 2.4 Install the chart

```bash
helm upgrade --install drsynth ./charts/drsynth-stack -f ./charts/drsynth-stack/values.local.yaml \
  --set rtservice.hfToken="$HF_TOKEN"
```

Access BFF:
- NodePort default: http://localhost:30080

### 2.5 GPU notes

For GPU workloads (rtservice/whisperx/finalizer):
- Ensure `nvidia-smi` works inside WSL2.
- Start minikube with GPU support (example):

```bash
minikube start --driver=docker --gpus=all
```

Then enable in values:
- `rtservice.gpu.enabled=true`
- `whisperxWorker.gpu.enabled=true`
- `finalizerWorker.gpu.enabled=true`

> You still need the NVIDIA k8s device plugin in the cluster for `nvidia.com/gpu` to work.

---

## 3) Observability (recommended roadmap)

- Keep **Prometheus** long-term; it remains the standard metrics backend.

Suggested phases:

### Phase 0 (now)
- Docker logs: `docker compose logs -f`
- K8s logs: `kubectl logs -f`.

### Phase 1 (metrics)
- Prometheus + Grafana
- Add `/metrics` endpoints to samuraibff and samuraipersistor (and later Python services).

### Phase 2 (traces + log aggregation)
- OpenTelemetry SDK in each service
- OpenTelemetry Collector
- Tempo (traces) + Loki (logs)
- Grafana as the single UI for metrics/logs/traces

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
