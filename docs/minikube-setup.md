Local Kubernetes setup (nanosamurai)

This doc focuses on running **app services in Kubernetes** while keeping **infra in Docker Compose**.

It supports two local Kubernetes variants:
- **Docker Desktop Kubernetes (recommended)**
- **Minikube (WSL2/Linux)**

Infra (Docker Compose): **Kafka + Postgres** (optionally Keycloak)
Apps (Kubernetes): `samuraibff`, `samuraipersistor`, `rtservice`, `whisperx_worker`, `recorder_worker`, `finalizer_worker`

> GPU note (WSL2): NVIDIA `nvidia.com/gpu` resource advertisement via the k8s device-plugin is currently unreliable on WSL2.
> The chart defaults to CPU-first (no `nvidia.com/gpu` limits). You can still run GPU-enabled containers on some setups,
> but don’t rely on Kubernetes GPU scheduling locally.

This mirrors production topology more closely than “all-compose”, while staying lightweight for dev.

---

## 0) Preconditions (Windows 11 + Docker Desktop + WSL2)

- Docker Desktop running
- `kubectl`, `helm`, `minikube` installed
- For GPU workloads:
  - NVIDIA driver installed on Windows
  - In WSL2: `nvidia-smi` works

> Tip: run minikube from **WSL2** if you’re doing GPU work. If you run it from PowerShell it can still work, but WSL2 tends to be less painful for GPU + Linux tooling.

---

## 1) Start infra in Docker Compose

From repo root:

```bash
copy .env.example .env
# set HF_TOKEN in .env

docker compose up -d broker kafka_init postgres db_migrate db_seed
```

Infra endpoints:
- Postgres (from host): `localhost:5432`
- Postgres (from pods):
  - Docker Desktop k8s: `host.docker.internal:5432`
  - minikube: `host.minikube.internal:5432`

- Kafka (from host tools): `localhost:9092`
- Kafka (from docker containers): `broker:29092`
- Kafka (from pods):
  - Docker Desktop k8s: `host.docker.internal:49092`
  - minikube: `host.minikube.internal:39092`

Why two Kafka ports? See `docs/local-stack.md`.

---

## 2) Choose your local Kubernetes

### Option A (recommended): Docker Desktop Kubernetes

1) Enable Kubernetes in Docker Desktop settings.
2) Switch your kubectl context:

```bash
kubectl config use-context docker-desktop
```

### Option B: Minikube (WSL2/Linux)

```bash
minikube start --driver=docker
```

> We intentionally do **CPU-first** local k8s. See GPU note at the top.

---

## 3) Make images available to Kubernetes

You have two supported approaches.

### Option A: build on host Docker + load into cluster (simple)

Build images:

```bash
# drsynth images
docker build -t drsynth-rtservice:dev -f rtservice/Dockerfile .
docker build -t drsynth-whisperx-worker:dev -f whisperx_worker/Dockerfile .
docker build -t drsynth-recorder-worker:dev -f recorder_worker/Dockerfile .
docker build -t drsynth-finalizer-worker:dev -f finalizer_worker/Dockerfile .

# other repos (run in their directories)
# samuraibff: docker build -t samuraibff:local .
# samuraipersistor: docker build -t samuraipersistor:local .
```

Load into cluster:

```bash
# Minikube only:
minikube image load drsynth-rtservice:dev
minikube image load drsynth-whisperx-worker:dev
minikube image load drsynth-recorder-worker:dev
minikube image load drsynth-finalizer-worker:dev
minikube image load samuraibff:local
minikube image load samuraipersistor:local

# Docker Desktop Kubernetes:
# (Images are already in the Docker Desktop engine; no extra load step needed.)
```

### Option B: build directly into minikube Docker daemon (minikube only)

```bash
minikube -p minikube docker-env
# follow printed instructions for your shell
# then docker build ...
```

---

## 4) Recordings storage

### Minikube
If you use the hostPath mode, create the directory on the minikube VM:

```bash
minikube ssh -- "sudo mkdir -p /data/nanosamurai-recordings && sudo chmod -R 777 /data/nanosamurai-recordings"
```

### Docker Desktop Kubernetes
Prefer the default PVC mode (chart default). No manual directory creation needed.

---

## 5) Install the Helm chart

### Docker Desktop Kubernetes

```bash
# PowerShell note: use $env:HF_TOKEN (not $HF_TOKEN).
helm upgrade --install nanosamurai ./charts/nanosamurai-stack \
  -f ./charts/nanosamurai-stack/values.local.docker-desktop.yaml \
  --set rtservice.hfToken="$env:HF_TOKEN"
```

### Minikube

```bash
helm upgrade --install nanosamurai ./charts/nanosamurai-stack \
  -f ./charts/nanosamurai-stack/values.local.minikube.yaml \
  --set rtservice.hfToken="$HF_TOKEN"
```

Check:

```bash
kubectl get pods
kubectl logs -f deploy/nanosamurai-samuraibff
```

Access BFF (recommended for Docker Desktop on Windows):

Port-forward (preferred):
```bash
# If port 8000 is already used on your machine, you can choose a different local port:
#   kubectl port-forward svc/nanosamurai-stack-bff 8001:8000
kubectl port-forward svc/nanosamurai-stack-bff 8000:8000
```
Then open:
- http://localhost:8000

NodePort (optional):
- http://localhost:30080 (default NodePort)

> Keycloak note: redirects/callbacks are controlled by the client configuration (redirect URIs).
> If you use NodePort, include `http://localhost:30080/*` in allowed redirect URIs.
> If you use port-forward (8000), include `http://localhost:8000/*`.

---

## 6) Stop / cleanup (important on laptops)

If you used **minikube in WSL2**, stop it when you’re done so it doesn’t keep consuming CPU/RAM:

```bash
minikube stop
```

To completely remove the cluster and free disk:

```bash
minikube delete
```

Quick status check:

```bash
minikube status
```

> Docker Desktop Kubernetes does not have a separate VM you need to stop; it’s part of Docker Desktop.

---

## 7) Smoke tests (local k8s)

These are tiered smoke tests. **Tier 1 and 2** are recommended for local dev.
**Tier 3 and 4** are optional (Kafka verification and async pipeline signals).

### Tier 1: BFF connectivity
PASS if we can create a session and receive at least one JSON event on `/ws/events`.

```bash
kubectl port-forward svc/nanosamurai-stack-bff 8000:8000

python -m venv .venv
.venv\\Scripts\\pip install -r utilities/k8s_local_smoke_test/requirements.txt

.venv\\Scripts\\python utilities/k8s_local_smoke_test/tier1_bff_connectivity.py
```

### Tier 2: realtime audio -> ASR event
PASS if streaming PCM16 audio to `/ws/audio` results in at least one `type=asr` event on `/ws/events`.

```bash
.venv\\Scripts\\python utilities/k8s_local_smoke_test/tier2_realtime_asr.py --wav tests/data/test_cs.wav --lang cs
```

### Tier 3 (optional): verify BFF publishes AudioChunk to Kafka
PASS if `audio.raw` contains an `AudioChunk` with the session_id.

```bash
.venv\\Scripts\\pip install -r utilities/k8s_local_smoke_test/requirements.kafka.txt
.venv\\Scripts\\python utilities/k8s_local_smoke_test/tier3_kafka_audio_raw.py --kafka-bootstrap localhost:9092
```

### Tier 4 (optional): verify async pipeline signals
PASS if we observe (at least one) downstream event for the session_id.

```bash
.venv\\Scripts\\pip install -r utilities/k8s_local_smoke_test/requirements.kafka.txt
.venv\\Scripts\\python utilities/k8s_local_smoke_test/tier4_async_pipeline.py --kafka-bootstrap localhost:9092 --timeout 120
```

Notes:
- On CPU-only machines, `rtservice` startup (model downloads) can be long and Tier 2 may fail until warm.
- Recorder emits only after idle timeout (default `RECORDER_IDLE_SECONDS=30`).
- WhisperX refined emits per-slice (default `WHISPERX_SLICE_SECONDS=60`).
- Final transcript can be slow on CPU; treat Tier 4 as opt-in locally.

## 8) Debugging cheatsheet

- Pods:
  ```bash
  kubectl get pods -o wide
  ```

- Logs:
  ```bash
  kubectl logs -f deploy/nanosamurai-whisperx-worker
  ```

- Exec:
  ```bash
  kubectl exec -it deploy/nanosamurai-whisperx-worker -- sh
  ```

- Verify pods can reach Kafka:
  ```bash
  kubectl exec -it deploy/nanosamurai-whisperx-worker -- sh -lc "getent hosts host.minikube.internal"
  ```
