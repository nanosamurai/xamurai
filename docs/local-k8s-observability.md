# Local Kubernetes observability (Grafana OSS + Prometheus + Loki + Tempo)

Status: **WIP** (branch: `add-observability`)

This runbook adds an **observability stack** to the local k8s topology documented in:
- `docs/local-k8s-setup.md` (Compose infra + k8s apps)

## Goals (what “done” means)

We want to see:

1. **Metrics** in Grafana (via Prometheus)
2. **Logs** in Grafana (via Loki)
3. **Traces** in Grafana (via Tempo)
4. Correlation by `session_id` and trace context propagation across:
   - browser ↔ `samuraibff` (HTTP + WS)
   - `samuraibff` ↔ `rtservice` (gRPC)
   - Kafka hops (`audio.raw` → workers → `transcripts.*` / `recordings.finished`)
   - `samuraipersistor`

Key decision (v1): **trace context propagates via Kafka headers** (`traceparent`) rather than adding fields into protobuf messages.

---

## Topology

### Where things run (local dev)

- **Infra**: Docker Compose on host
  - Kafka, Postgres, LocalStack (S3)
- **Apps**: Kubernetes via Helm chart `charts/nanosamurai-stack`
  - `samuraibff`, `samuraipersistor`
  - `rtservice`, `whisperx_worker`, `recorder_worker`, `finalizer_worker`
- **Observability**: Kubernetes (this runbook)
  - `kube-prometheus-stack` (Prometheus + Grafana + kube-state-metrics)
  - Loki
  - Tempo
  - Grafana Alloy (log collection + optional OTLP gateway)
  - OpenTelemetry Collector (OTLP receiver + exporters)

---

## 0) Preconditions

- Docker Desktop + Kubernetes enabled (preferred on Windows), or minikube
- `kubectl`, `helm`
- Infra running from `docs/local-k8s-setup.md`:
  ```bash
  docker compose up -d broker kafka_init postgres db_migrate db_seed localstack
  ```

---

## 1) Install observability stack (Helm)

We deploy into namespace `observability`.

### 1.1 Create namespace

```bash
kubectl create namespace observability
```

### 1.2 Add Helm repos

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo update
```

### 1.3 Install kube-prometheus-stack

Values file: `observability/kube-prometheus-stack.values.local.yaml`

```bash
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  -n observability \
  -f observability/kube-prometheus-stack.values.local.yaml
```

### 1.4 Install Loki

Values file: `observability/loki.values.local.yaml`

```bash
helm upgrade --install loki grafana/loki \
  -n observability \
  -f observability/loki.values.local.yaml
```

### 1.5 Install Tempo

Values file: `observability/tempo.values.local.yaml`

```bash
helm upgrade --install tempo grafana/tempo \
  -n observability \
  -f observability/tempo.values.local.yaml
```

### 1.6 Install Grafana Alloy (logs)

Values file: `observability/alloy.values.local.yaml`

```bash
helm upgrade --install alloy grafana/alloy \
  -n observability \
  -f observability/alloy.values.local.yaml
```

### 1.7 Install OpenTelemetry Collector

Values file: `observability/otel-collector.values.local.yaml`

```bash
helm upgrade --install otel-collector open-telemetry/opentelemetry-collector \
  -n observability \
  -f observability/otel-collector.values.local.yaml
```

---

## 2) Access Grafana

```bash
kubectl -n observability port-forward svc/kube-prometheus-stack-grafana 3000:80
```

Grafana: http://localhost:3000

Default credentials are set in `observability/kube-prometheus-stack.values.local.yaml`.

---

## 3) Validate signal ingestion

### 3.1 Metrics (Prometheus)

In Grafana Explore → Prometheus:
- query: `up`

### 3.2 Logs (Loki)

In Grafana Explore → Loki:
- query for a pod label (exact label keys depend on Alloy relabeling):
  - start with `{namespace="default"}`

### 3.3 Traces (Tempo)

In Grafana Explore → Tempo:
- search by `service.name`

---

## 4) Generate real traffic (existing smoke tests)

Use the existing local-k8s smoke tests:

- Tier2 (realtime)
- Tier4 (async pipeline)

See `docs/local-k8s-setup.md`.

---

## Decisions & change log

- 2026-02-26: Use **Grafana Alloy** (not Promtail) for log shipping.
- 2026-02-26: Use Kafka header propagation (`traceparent`) for tracing v1.
- 2026-02-26: For Clojure/JVM services (`samuraibff`, `samuraipersistor`), prefer **OpenTelemetry Java Agent** over application code changes.
