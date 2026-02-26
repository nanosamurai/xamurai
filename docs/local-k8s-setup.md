# Local Kubernetes setup (nanosamurai)

This runbook describes a **k8s-realistic local dev topology**:

- **Infra** runs on your machine via **Docker Compose**: Kafka + Postgres + LocalStack(S3)
- **Apps** run in **Kubernetes** via the Helm chart in `charts/nanosamurai-stack`
  - `samuraibff`, `samuraipersistor`
  - `rtservice`, `whisperx_worker`, `recorder_worker`, `finalizer_worker`

For the **compose-only** (fastest end-to-end) workflow, see: **`docs/local-stack.md`**.

## Supported local Kubernetes variants

- **Docker Desktop Kubernetes (Windows) — preferred and tested**
- **Minikube (Linux/WSL2) — supported, but not the preferred Windows workflow**

> GPU note: `nvidia.com/gpu` scheduling via the k8s device plugin is often unreliable on WSL2.
> The chart defaults to CPU-first (no GPU limits). Treat local GPU scheduling as best-effort.

---

## 0) Preconditions

### Common

- Docker Desktop running
- `kubectl`, `helm`
- `docker compose`
- `HF_TOKEN` (required by `rtservice` for model downloads)

### If you use Minikube

- `minikube` installed
- Recommended: run minikube from **WSL2/Linux shell** (less friction for Linux tooling)

---

## 1) Start infra in Docker Compose

From repo root:

```bash
copy .env.example .env
# set HF_TOKEN in .env
```

### Security note: avoid exposing infra ports to your LAN

By default, `docker-compose.yml` binds published ports (Postgres/Kafka/LocalStack/etc.) to **localhost only**.

- Default: `COMPOSE_BIND_IP=127.0.0.1` (recommended; safest)
- If you run **Docker Desktop Kubernetes** and need pods to reach compose infra via `host.docker.internal`, you may need:
  - `COMPOSE_BIND_IP=0.0.0.0`
  - and rely on **Windows Firewall** to block inbound LAN traffic to these ports.

Sanity check (Windows):
```bat
netstat -ano | findstr ":5432 :4566 :9092 :39092 :49092"
```
- With `COMPOSE_BIND_IP=127.0.0.1`, listeners should show `127.0.0.1:<port>`.
- If you see `0.0.0.0:<port>`, that port is reachable from outside localhost unless blocked by a firewall.

Start infra:

```bash
docker compose up -d broker kafka_init postgres db_migrate db_seed localstack
```

### Infra endpoints and Kafka ports

**Postgres**
- From host tools: `localhost:5432`
- From pods:
  - Docker Desktop k8s: `host.docker.internal:5432`
  - Minikube: `host.minikube.internal:5432`

**LocalStack (S3)**
- From host tools: `http://localhost:4566`
- From pods:
  - Docker Desktop k8s: `http://host.docker.internal:4566`
  - Minikube: `http://host.minikube.internal:4566`

**Kafka**
Kafka is reachable from multiple “worlds”, each requiring a different advertised address:

- **Other Docker containers** (compose network): `broker:29092`
- **Host machine tools**: `localhost:9092`
- **Minikube pods** (pods talking to Docker-hosted Kafka): `host.minikube.internal:39092`
- **Docker Desktop Kubernetes pods** (pods talking to Docker-hosted Kafka): `host.docker.internal:49092`

Why we keep separate `39092` and `49092`:
Kafka returns broker addresses based on the listener the client connects to. If a Docker Desktop k8s pod connects via the minikube listener, it will get back `host.minikube.internal` in metadata (unresolvable in Docker Desktop), and consumers fail.

### LocalStack S3 (speaker enrollment)

If you plan to use **enrolled speakers** end-to-end in k8s mode:
- **samuraibff** needs S3 access (it writes `speaker.json` + samples)
- **rtservice/whisperx/finalizer** need S3 access (they read tenant enrollment and map diarization speakers)

> Important: diarization/enrollment in `whisperx_worker` and `finalizer_worker` also require **HF_TOKEN**.
> When using the chart’s `rtservice.hfTokenSecret`, make sure the Helm chart version you deploy
> propagates that Secret into those worker pods as well.

Create a Kubernetes secret with the LocalStack credentials (default `test`/`test`):

```bash
kubectl create secret generic nanosamurai-localstack-s3 \
  --from-literal=accessKey=test \
  --from-literal=secretKey=test
```

The provided overlays already wire this up:
- `charts/nanosamurai-stack/values.local.docker-desktop.yaml`
- `charts/nanosamurai-stack/values.local.minikube.yaml`

---

## 2) Choose your local Kubernetes

### Option A (preferred on Windows): Docker Desktop Kubernetes

1) Enable Kubernetes in Docker Desktop settings.
2) Switch your kubectl context:

```bash
kubectl config use-context docker-desktop
```

> Docker Desktop may run Kubernetes as **kubeadm-based** or **kind-based** depending on version/settings.
> If your node is named `desktop-control-plane`, you are on the kind-based variant and the image caching notes below apply.

### Option B: Minikube (WSL2/Linux)

```bash
minikube start --driver=docker
```

---

## 3) Build images and make them available to Kubernetes

Build images (recommended local tag: `:local`):

```bash
# xamurai images
docker build -t xamurai-rtservice:local -f rtservice/Dockerfile .
docker build -t xamurai-whisperx-worker:local -f whisperx_worker/Dockerfile .
docker build -t xamurai-recorder-worker:local -f recorder_worker/Dockerfile .
docker build -t xamurai-finalizer-worker:local -f finalizer_worker/Dockerfile .

# other repos (run in their directories)
# samuraibff: docker build -t samuraibff:local .
# samuraipersistor: docker build -t samuraipersistor:local .
```

### Docker Desktop Kubernetes

#### Case 1: kubeadm-based Docker Desktop k8s

Images built into the Docker Desktop engine are typically usable directly by the cluster.
If you rebuild an image but keep the same tag (e.g. `:local`), Kubernetes may still need a restart to pick it up:

```bash
kubectl rollout restart deploy/nanosamurai-stack-rtservice
kubectl rollout restart deploy/nanosamurai-stack-whisperx-worker
kubectl rollout restart deploy/nanosamurai-stack-finalizer-worker
kubectl rollout restart deploy/nanosamurai-stack-recorder-worker
kubectl rollout restart deploy/nanosamurai-stack-bff
kubectl rollout restart deploy/nanosamurai-stack-persistor
```

#### Case 2: kind-based Docker Desktop k8s (node `desktop-control-plane`)

kind nodes use containerd and can keep serving a cached image even if you rebuild `:local` in the host Docker engine.
In that case, **import the image into the node’s containerd**:

```bat
REM --- core services ---
docker save xamurai-rtservice:local | docker exec -i desktop-control-plane sh -lc "ctr -n k8s.io images import -"
docker save xamurai-recorder-worker:local | docker exec -i desktop-control-plane sh -lc "ctr -n k8s.io images import -"
docker save xamurai-finalizer-worker:local | docker exec -i desktop-control-plane sh -lc "ctr -n k8s.io images import -"
docker save xamurai-whisperx-worker:local | docker exec -i desktop-control-plane sh -lc "ctr -n k8s.io images import -"

REM --- optional (only if you also built local tags) ---
docker save samuraibff:local | docker exec -i desktop-control-plane sh -lc "ctr -n k8s.io images import -"
docker save samuraipersistor:local | docker exec -i desktop-control-plane sh -lc "ctr -n k8s.io images import -"
```

Then restart pods to pick up the imported image:

```bash
kubectl rollout restart deploy/nanosamurai-stack-finalizer-worker
kubectl rollout restart deploy/nanosamurai-stack-whisperx-worker
```

> Tip: if your node container name differs, check `kubectl get nodes` and `docker ps`.

### Minikube

Load images into minikube:

```bash
minikube image load xamurai-rtservice:local
minikube image load xamurai-whisperx-worker:local
minikube image load xamurai-recorder-worker:local
minikube image load xamurai-finalizer-worker:local
minikube image load samuraibff:local
minikube image load samuraipersistor:local
```

Alternative (minikube only): build directly into minikube’s Docker daemon:

```bash
minikube -p minikube docker-env
# follow printed instructions for your shell
# then docker build ...
```

---

## 4) Recordings storage

### Docker Desktop Kubernetes

Prefer the default PVC mode (chart default). No manual directory creation needed.

### Minikube

If you use hostPath mode, create the directory on the minikube VM:

```bash
minikube ssh -- "sudo mkdir -p /data/nanosamurai-recordings && sudo chmod -R 777 /data/nanosamurai-recordings"
```

---

## 5) Install the Helm chart

> Note: the chart currently renders resource names with the `nanosamurai-stack-...` prefix (see `charts/nanosamurai-stack/templates/_helpers.tpl`).

### Docker Desktop Kubernetes

```bash
# PowerShell note: use $env:HF_TOKEN (not $HF_TOKEN).
helm upgrade --install nanosamurai-stack ./charts/nanosamurai-stack \
  -f ./charts/nanosamurai-stack/values.local.docker-desktop.yaml \
  --set rtservice.hfToken="$env:HF_TOKEN"
```

### Minikube

```bash
helm upgrade --install nanosamurai-stack ./charts/nanosamurai-stack \
  -f ./charts/nanosamurai-stack/values.local.minikube.yaml \
  --set rtservice.hfToken="$HF_TOKEN"
```

Check:

```bash
kubectl get pods
kubectl get deploy
kubectl logs -f deploy/nanosamurai-stack-bff
```

### Access BFF (recommended for Docker Desktop on Windows)

Port-forward (preferred):

```bash
# If port 8000 is already used on your machine, you can choose a different local port:
#   kubectl port-forward svc/nanosamurai-stack-bff 8001:8000
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

> Keycloak note: redirects/callbacks are controlled by the client configuration (redirect URIs).
> If you use NodePort, include `http://localhost:30080/*` in allowed redirect URIs.
> If you use port-forward (8000), include `http://localhost:8000/*`.

---

## 6) Stop / cleanup (important on laptops)

### Docker Desktop

```bash
helm uninstall nanosamurai-stack
```

### Minikube

If you used **minikube in WSL2**, stop it when you’re done so it doesn’t keep consuming CPU/RAM:

```bash
minikube stop
```

To completely remove the cluster and free disk:

```bash
minikube delete
```

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

PASS if we observe a selected downstream event for the session_id.

```bash
.venv\\Scripts\\pip install -r utilities/k8s_local_smoke_test/requirements.kafka.txt

# Default (quick): verify recorder emits `recordings.finished`
.venv\\Scripts\\python utilities/k8s_local_smoke_test/tier4_async_pipeline.py \
  --kafka-bootstrap localhost:9092 --timeout 120 --signal recording-finished

# Strict: verify finalizer successfully emits `transcripts.final`
# (this will FAIL if finalizer is down/crashlooping)
.venv\\Scripts\\python utilities/k8s_local_smoke_test/tier4_async_pipeline.py \
  --kafka-bootstrap localhost:9092 --timeout 300 --signal final
```

Notes:
- On CPU-only machines, `rtservice` startup (model downloads) can be long and Tier 2 may fail until warm.
- Recorder emits only after idle timeout (default `RECORDER_IDLE_SECONDS=30`).
- WhisperX refined emits per-slice (default `WHISPERX_SLICE_SECONDS=60`).
- Final transcript can be slow on CPU; treat Tier 4 strict mode (`--signal final`) as opt-in locally.

Run Tier4 and require a specific speaker label (with an alias accepted for local data):

```bat
.venv-smoke\Scripts\python utilities/k8s_local_smoke_test/tier4_async_pipeline.py \
  --base-url http://localhost:8000 \
  --wav tests/data/test_cs.wav --lang cs --stream-seconds 6.0 \
  --kafka-bootstrap 127.0.0.1:9092 \
  --timeout 420 --signal final \
  --expect-speaker Miro-cz --expect-speaker-alias "Miro (cz)"
```

---

## 8) Debugging cheatsheet

- Pods:
  ```bash
  kubectl get pods -o wide
  ```

- Logs:
  ```bash
  kubectl logs -f deploy/nanosamurai-stack-whisperx-worker
  ```

- Exec:
  ```bash
  kubectl exec -it deploy/nanosamurai-stack-whisperx-worker -- sh
  ```

- Verify pods can resolve host (pick the one for your cluster):
  ```bash
  # Docker Desktop k8s
  kubectl exec -it deploy/nanosamurai-stack-whisperx-worker -- sh -lc "getent hosts host.docker.internal"

  # Minikube
  kubectl exec -it deploy/nanosamurai-stack-whisperx-worker -- sh -lc "getent hosts host.minikube.internal"
  ```

## Note: an easy way to pause local setup without losing HF cache and data:
#### Option A (safest): scale only this stack via label selector

This avoids accidentally scaling other unrelated deployments:

```bat
kubectl scale deploy -n default -l app.kubernetes.io/name=nanosamurai-stack --replicas=0
```

Resume (you’ll need to set replicas back; if everything was 1):

```bat
kubectl scale deploy -n default -l app.kubernetes.io/name=nanosamurai-stack --replicas=1
```

#### Option B: scale explicit deployments (most explicit)

```bat
kubectl scale deploy/nanosamurai-stack-bff -n default --replicas=0
kubectl scale deploy/nanosamurai-stack-persistor -n default --replicas=0
kubectl scale deploy/nanosamurai-stack-rtservice -n default --replicas=0
kubectl scale deploy/nanosamurai-stack-whisperx-worker -n default --replicas=0
kubectl scale deploy/nanosamurai-stack-recorder-worker -n default --replicas=0
kubectl scale deploy/nanosamurai-stack-finalizer-worker -n default --replicas=0
```

…and then scale them back to 1 when you want to continue.

### Extra safety: keep the PVC even if you *do* uninstall later

If you ever want the convenience of `helm uninstall` but still keep cache, you can annotate the PVC(s) with Helm’s keep policy:

```bat
kubectl annotate pvc nanosamurai-stack-hf-cache-pvc helm.sh/resource-policy=keep
```
