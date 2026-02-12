 Local Minikube setup (drsynth stack)

This doc focuses on running **app services in Kubernetes** while keeping **infra in Docker Compose**.

- Infra (Docker Compose): **Kafka + Postgres** (optionally Keycloak)
- Apps (Minikube): `samuraibff`, `samuraipersistor`, `rtservice`, `whisperx_worker`, `recorder_worker`, `finalizer_worker`

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
- Postgres (from minikube pods): `host.minikube.internal:5432`

- Kafka (from host tools): `localhost:9092`
- Kafka (from docker containers): `broker:29092`
- Kafka (from minikube pods): `host.minikube.internal:39092`

Why two Kafka ports? See `docs/local-stack.md`.

---

## 2) Start Minikube

### 2.1 CPU-only

```bash
minikube start --driver=docker
```

### 2.2 GPU-enabled (recommended for rtservice/whisperx/finalizer)

```bash
minikube start --driver=docker --gpus=all
```

> GPU scheduling also requires the NVIDIA k8s device plugin (next section).

---

## 3) Install NVIDIA device plugin (GPU only)

Inside the cluster, install NVIDIA device plugin so pods can request `nvidia.com/gpu`:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/nvidia-device-plugin.yml
```

Verify:

```bash
kubectl -n kube-system get ds | grep nvidia
kubectl describe node | findstr /i nvidia
```

---

## 4) Make images available to Minikube

You have two supported approaches.

### Option A: build on host Docker + load into Minikube (simple)

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

Load into minikube:

```bash
minikube image load drsynth-rtservice:dev
minikube image load drsynth-whisperx-worker:dev
minikube image load drsynth-recorder-worker:dev
minikube image load drsynth-finalizer-worker:dev
minikube image load samuraibff:local
minikube image load samuraipersistor:local
```

### Option B: build directly into Minikube Docker daemon

```bash
minikube -p minikube docker-env
# follow printed instructions for your shell
# then docker build ...
```

---

## 5) Persistent recordings directory (hostPath PV)

The Helm chart uses a hostPath PV by default.

Create the directory on the minikube VM:

```bash
minikube ssh -- "sudo mkdir -p /data/drsynth-recordings && sudo chmod -R 777 /data/drsynth-recordings"
```

---

## 6) Install the Helm chart

```bash
helm upgrade --install drsynth ./charts/drsynth-stack \
  -f ./charts/drsynth-stack/values.local.yaml \
  --set rtservice.hfToken="$HF_TOKEN"
```

Check:

```bash
kubectl get pods
kubectl logs -f deploy/drsynth-samuraibff
```

Access BFF (NodePort default):
- http://localhost:30080

---

## 7) Debugging cheatsheet

- Pods:
  ```bash
  kubectl get pods -o wide
  ```

- Logs:
  ```bash
  kubectl logs -f deploy/drsynth-whisperx-worker
  ```

- Exec:
  ```bash
  kubectl exec -it deploy/drsynth-whisperx-worker -- sh
  ```

- Verify pods can reach Kafka:
  ```bash
  kubectl exec -it deploy/drsynth-whisperx-worker -- sh -lc "getent hosts host.minikube.internal"
  ```
