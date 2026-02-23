# Kubernetes deployment roadmap (nanosamurai)

This is a living design doc for the long-term Kubernetes deployment of the nanosamurai stack.
It’s intended to guide local Kubernetes development, staging, and production.

Local note:
- On **Windows**, the preferred and tested local cluster is **Docker Desktop Kubernetes**.
- **Minikube** is still useful on Linux/WSL2, but it’s not the primary Windows workflow.

**Services**
- App/API/UI: `samuraibff`
- Persistence: `samuraipersistor`
- Realtime gRPC: `rtservice`
- Kafka consumers/producers: `whisperx_worker`, `recorder_worker`, `finalizer_worker`

**Infra dependencies**
- Kafka
- Postgres
- (Auth) Keycloak
- (Optional) object storage (S3/MinIO) for recordings/artifacts

---

## 1) Goals

### Short-term (dev reliability)
- One-command local bring-up (compose-only)
- One-command “k8s-like” bring-up (compose infra + local k8s apps)
- Deterministic topic + schema initialization

### Medium-term (staging/prod correctness)
- Helm chart(s) with sane defaults and environment overlays
- Secrets management (no tokens in git)
- GPU scheduling on dedicated node pools
- Baseline observability (metrics + logs) with a clear upgrade path to tracing

### Long-term (platform)
- GitOps-style continuous delivery
- Multi-tenant correctness (BFF + persistor + auth)
- Recording storage abstraction (S3 first-class)
- Strong request correlation (trace IDs across WS/gRPC/Kafka)

---

## 2) Packaging strategy

### Recommended
Split packaging into two deployable units:

1) **Infra chart** (later)
   - Kafka (or a managed Kafka)
   - Postgres (or managed RDS)
   - Keycloak (optional)

2) **Apps chart** (in this repo as `charts/nanosamurai-stack`)
   - Deployments for bff/persistor/rtservice/workers
   - ConfigMaps and Secrets
   - Services + Ingress

Why: infra lifecycle differs (stateful, backups, upgrades). Apps iterate faster.

---

## 3) Environments and overlays

Use Helm values overlays per environment:
- `values.yaml` (defaults)
- `values.local.docker-desktop.yaml` (Docker Desktop Kubernetes)
- `values.local.minikube.yaml` (minikube)
- `values.dev.yaml` (shared dev cluster)
- `values.staging.yaml`
- `values.prod.yaml`

Keep “what changes between envs” explicit:
- image tags
- replica counts
- resource requests/limits
- GPU scheduling
- external endpoints (Kafka, Postgres, Keycloak)

---

## 4) Kafka and initialization

### Topic creation
Current compose uses `kafka_init` (idempotent).

K8s equivalent options:
- A Kubernetes Job (or Helm hook) that runs `kafka-topics.sh --create --if-not-exists`
- Prefer managed Kafka topic creation via Terraform/Pulumi in the long term

### Config
Standardize bootstrap values:
- in-cluster apps: `kafka:9092` (service)
- local minikube apps: `host.minikube.internal:39092` (compose)
- local Docker Desktop k8s apps: `host.docker.internal:49092` (compose)

---

## 5) Postgres schema/migrations

Today we run SQL migrations in compose via `psql` (idempotent).

K8s options:
1) **Migration Job** per deploy (Helm hook)
2) **Migrations run by persistor** (app-managed)
3) **External migration pipeline** (CI/CD step)

Recommendation:
- For now: Helm hook Job running `psql` migrations.
- Later: move to a proper migration tool or Flyway/Liquibase (esp. if multiple services evolve schema).

---

## 6) GPU scheduling

GPU services:
- `rtservice` (pyannote + faster-whisper)
- `whisperx_worker`
- `finalizer_worker`

Approach:
- Separate GPU node pool.
- Add:
  - `resources.limits.nvidia.com/gpu: 1`
  - node selectors / taints+tolerations

Also:
- install NVIDIA device plugin in the cluster
- decide on CUDA base images / driver compatibility

---

## 7) Storage strategy

### Recordings
Current local dev uses a shared volume (`drsynth_recordings`) and file:// URLs.

Production recommendation:
- Move to S3-backed recordings and artifacts.
- `recorder_worker` uploads to S3.
- `finalizer_worker` downloads from S3.

K8s benefit: avoids needing shared RWX volumes for large audio artifacts.

---

## 8) Networking and ingress

- `samuraibff` HTTP ingress (web + API)
- `rtservice` gRPC ingress (optional; some deployments may keep it internal)

Decide:
- Ingress controller (nginx/traefik)
- gRPC routing rules

---

## 9) Observability plan

### Phase 0 (now)
- docker/kubectl logs only

### Phase 1 (recommended baseline)
- Prometheus + Grafana
- service-level metrics:
  - JVM metrics for bff/persistor
  - Python metrics for workers (prometheus-client)

### Phase 2 (distributed tracing)
- OpenTelemetry SDK in all services
- OpenTelemetry Collector
- Tempo (traces)
- Loki (logs)

### Required for useful tracing
- Correlation IDs propagated across:
  - Browser ↔ BFF (HTTP/WS)
  - BFF ↔ rtservice (gRPC)
  - Kafka headers (traceparent)
  - persistor writes

---

## 10) CI/CD and promotion

Recommended pipeline:
- Build/push images for each service with immutable tags (git sha)
- Helm chart deploy with pinned image tags
- Environment promotion (dev -> staging -> prod)

GitOps option:
- Argo CD or Flux

---

## 11) Open decisions / next work items

1) Do we want Kafka/Postgres managed in prod, or self-hosted?
2) Should Keycloak be local-only, or always external?
3) Recording storage: file vs S3 (strong recommendation: S3)
4) Observability baseline: Prometheus+Grafana now, tracing later.

---

## Appendix: Local parity rules

To keep local and k8s close:
- keep environment variables consistent
- keep topic names constant
- avoid implicit Kafka topic creation
- avoid implicit DB schema creation
- prefer explicit jobs/hooks for init
