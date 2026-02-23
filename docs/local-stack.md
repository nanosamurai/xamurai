# Local stack runbook (drsynth + samuraibff + samuraipersistor)

This document describes two supported local workflows:

1) **All Docker Compose** (fastest end-to-end)
2) **Docker Compose infra + Kubernetes apps** (k8s-realistic; CPU-first locally)

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

Security note:
- By default, `docker-compose.yml` binds published ports to **localhost only** via `COMPOSE_BIND_IP=127.0.0.1`.
- If you need services reachable from outside localhost (not recommended), override `COMPOSE_BIND_IP`.

Optional overrides:
- `SAMURAIBFF_PATH=...`
- `SAMURAIPERSISTOR_PATH=...`

### 1.2 Start the stack (auth disabled)

```bash
docker compose up --build
```

What this starts:
- Kafka (with internal + host + minikube listeners)
- **LocalStack (S3)** for enrollment storage (persistent volume)
- Postgres
- DB migrations (samuraipersistor migratus)
- DB seed (creates the dev tenant row needed by unauth BFF)
- rtservice + workers
- samuraibff
- samuraipersistor

Open:
- BFF/UI: http://localhost:8000
- Persistor health: http://127.0.0.1:8010/health
- LocalStack S3 endpoint: http://localhost:4566

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

### 1.4 Enrollment storage (LocalStack S3)

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


### 1.5 Common checks

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

## 2) Mode B — Docker Compose infra + K8s apps

This runs **Kafka + Postgres** in Docker Compose and deploys app services to local Kubernetes using the Helm chart in `charts/nanosamurai-stack`.

> Keycloak is expected to be reachable externally (your ECS dev profile is fine). Compose does not start it by default.

### 2.1 Start infra only

Before you start: for **Docker Desktop Kubernetes pods** to reach compose infra via `host.docker.internal`, you may need:
- `COMPOSE_BIND_IP=0.0.0.0` (and then use Windows Firewall to restrict LAN access)
- this is however unsecure and should be avoided

Then:

```bash
docker compose up -d broker kafka_init postgres persistor_migrate db_seed localstack
```

#### Kafka ports and why we expose two of them

Kafka in this stack is reachable from four “worlds”, each needing a different advertised address:

- **Other Docker containers** (compose network): `broker:29092`
- **Host machine tools** (Windows/macOS/Linux): `localhost:9092`
- **Minikube pods**: `host.minikube.internal:39092`
- **Docker Desktop Kubernetes pods**: `host.docker.internal:49092`

We keep `39092` specifically for minikube because Kafka’s advertised listeners must be different for
host vs. pod networking. Trying to reuse `localhost:9092` inside pods will fail.

If you also want local Keycloak, run it outside this compose (or re-add it as a compose profile later).

Infra endpoints (as seen from pods):
- Minikube:
  - Kafka: `host.minikube.internal:39092`
  - Postgres: `host.minikube.internal:5432`
- Docker Desktop Kubernetes:
  - Kafka: `host.docker.internal:49092`
  - Postgres: `host.docker.internal:5432`

> We expose Kafka on `39092` specifically so that **minikube pods** can reach it,
> while still keeping `localhost:9092` working for host tools.

### 2.2 Build images and make them available to minikube

You have two common options:

#### Option A (simple): build normally + `minikube image load`

Build images on the host Docker engine (recommended local tag: `:local`):
```bash
# drsynth images
docker build -t drsynth-rtservice:local -f rtservice/Dockerfile .
docker build -t drsynth-whisperx-worker:local -f whisperx_worker/Dockerfile .
docker build -t drsynth-recorder-worker:local -f recorder_worker/Dockerfile .
docker build -t drsynth-finalizer-worker:local -f finalizer_worker/Dockerfile .

# in the other repos:
# samuraibff: docker build -t samuraibff:local .
# samuraipersistor: docker build -t samuraipersistor:local .
```

Then load into minikube:
```bash
minikube image load drsynth-rtservice:local
minikube image load drsynth-whisperx-worker:local
minikube image load drsynth-recorder-worker:local
minikube image load drsynth-finalizer-worker:local
minikube image load samuraibff:local
minikube image load samuraipersistor:local
```

Note: if you rebuild an image but keep the same tag (e.g. `:local`), Kubernetes will not
restart pods automatically. Force a restart to pick up rebuilt images:
```bash
kubectl rollout restart deploy/nanosamurai-stack-finalizer-worker
kubectl rollout restart deploy/nanosamurai-stack-whisperx-worker
```

#### Option B: build directly into minikube’s Docker daemon

```bash
minikube -p minikube docker-env
# then follow the printed instructions (PowerShell vs bash differs)
```

Then run `docker build ...` and images will already be present in minikube.

### 2.3 Create the recordings host path in minikube

The chart uses a simple hostPath PV by default (`/data/nanosamurai-recordings`). Create it:

```bash
minikube ssh -- "sudo mkdir -p /data/nanosamurai-recordings && sudo chmod -R 777 /data/nanosamurai-recordings"
```

### 2.4 Install the chart

```bash
helm upgrade --install nanosamurai ./charts/nanosamurai-stack -f ./charts/nanosamurai-stack/values.local.minikube.yaml \
  --set rtservice.hfToken="$HF_TOKEN"
```

Access BFF (recommended on Windows Docker Desktop):

Port-forward (preferred):
```bash
kubectl port-forward svc/nanosamurai-stack-bff 8000:8000
```
Then open:
- http://localhost:8000

NodePort (optional / opt-in):
- By default the Helm chart uses **ClusterIP** for security.
- To expose the BFF via NodePort, set:
  ```yaml
  bff:
    service:
      type: NodePort
      nodePort: 30080
  ```
  Then access:
  - http://localhost:30080

> Keycloak note: your Keycloak client redirect URIs must match whichever host+port you use.
> Allow `http://localhost:8000/*` for port-forward and/or `http://localhost:30080/*` for NodePort.

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
