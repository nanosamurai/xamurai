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

## 1.8 Enable OTEL in nanosamurai-stack Helm chart

The `charts/nanosamurai-stack` chart now supports optional OTEL wiring.

Recommended local settings:

- enable OTEL env vars for all pods
- enable Java agent for JVM services (`samuraibff`, `samuraipersistor`)

Example (Docker Desktop k8s):

```bash
helm upgrade --install nanosamurai-stack ./charts/nanosamurai-stack \
  -f ./charts/nanosamurai-stack/values.local.docker-desktop.yaml \
  --set observability.enabled=true \
  --set observability.otlpEndpoint=http://otel-collector.observability.svc.cluster.local:4317 \
  --set observability.javaAgent.enabled=true
```

Notes:
- Java agent is downloaded by an initContainer (requires cluster egress to GitHub).
- For local dev, this is acceptable; for staging/prod, bake it into the image or use an internal artifact.

---

## 2) Access Grafana

```bash
# If 3000 is already taken on your machine, change it (example uses 3001).
kubectl -n observability port-forward svc/kube-prometheus-stack-grafana 3001:80
```

Grafana: http://localhost:3001

Default credentials are set in `observability/kube-prometheus-stack.values.local.yaml`.

Windows note:
- In PowerShell, `curl` is an alias for `Invoke-WebRequest` (different flags).
- Use `curl.exe` for real curl.
- On Windows, the `python` shim might not exist (Store alias). Use `py ...` instead.

---

## 3) Validate signal ingestion

### 3.1 Metrics (Prometheus)

In Grafana Explore → Prometheus:
- query: `up`
- query: `kube_pod_info{namespace="default"}`

### 3.2 Logs (Loki)

In Grafana Explore → Loki:
- start with: `{job="loki.source.kubernetes.pods"}`
- narrow down to a specific service/pod via the `instance` label, e.g.
  - `{instance=~"default/nanosamurai-stack-bff.*"}`

Tip: you can also verify logs directly via Loki API:

```bash
kubectl -n observability port-forward svc/loki 3100:3100
curl.exe -s "http://127.0.0.1:3100/loki/api/v1/query_range?query=%7Bservice_name%3D%22loki.source.kubernetes.pods%22%7D&limit=1"
```

### 3.3 Traces (Tempo)

Traces arrive via OTLP:
- `samuraibff` + `samuraipersistor` via the OTEL Java agent
- (optionally) other services via OTEL SDKs

Quick verification without touching application code (generate sample traces):

```bash
docker run --rm ghcr.io/open-telemetry/opentelemetry-collector-contrib/telemetrygen:latest \
  traces --otlp-endpoint host.docker.internal:4317 --otlp-insecure \
  --rate 5 --duration 3s --service telemetrygen-local
```

Then in Grafana Explore → Tempo, search by `service.name = telemetrygen-local`.

If you want to hit a real trace from the system, run Tier2 smoke test and search by:
- `service.name = samuraibff`
- or use the Tempo HTTP API through Grafana proxy:

```bash
# Tempo datasource id is typically 4 in this setup (confirm via /api/datasources)
curl.exe -s -u admin:admin "http://127.0.0.1:3001/api/datasources/proxy/4/api/search?service.name=samuraibff&limit=5"
```

---

## 4) Generate real traffic (existing smoke tests)

Use the existing local-k8s smoke tests:

- Tier2 (realtime)
- Tier4 (async pipeline)

See `docs/local-k8s-setup.md`.

---

## 5) Roadmap to **end-to-end** tracing (hybrid approach → B)

You said you ultimately want **B** (single, connected trace across all services), but we’ll phase it in to keep debugging manageable.

### Phase 0 (already done): stack + “first traces”
- Grafana OSS + Prometheus + Loki + Tempo installed in `observability`
- JVM services (`samuraibff`, `samuraipersistor`) traced via OTEL Java agent → OTEL Collector → Tempo
- Logs shipped via Alloy → Loki

### Phase 1: lock in the **propagation contract** (even before Python emits spans)
Goal: make sure the trace context *can* cross boundaries.

1) **Kafka**: use W3C Trace Context header
- Header key: `traceparent`
- Producer MUST inject it into Kafka headers
- Consumer MUST extract it and set it as parent

2) **gRPC**: use W3C trace context in gRPC metadata
- Java agent typically injects/extracts automatically for gRPC clients/servers.
- For Python gRPC server we’ll add an interceptor to extract parent context from metadata.

3) Correlation key
- Add `session_id` to spans as an attribute (recommended key: `nanosamurai.session_id`) and to logs.

### Phase 2: instrument `rtservice` (Python gRPC) with OTEL SDK
Goal: BFF → gRPC → rtservice shows as one connected trace.

Implementation outline:
- Add OTLP exporter (grpc) → `OTEL_EXPORTER_OTLP_ENDPOINT` already provided by Helm
- Add gRPC server instrumentation:
  - easiest: `opentelemetry-instrumentation-grpc`
  - or manual interceptor + `tracer.start_as_current_span(...)`
- Add span attributes:
  - `service.name=rtservice`
  - `nanosamurai.session_id` (from incoming `AudioChunk.session_id`)

### Phase 3: instrument Kafka workers (Python, confluent_kafka)
Target services here:
- `whisperx_worker`
- `recorder_worker`
- `finalizer_worker`

Goal: the trace continues across Kafka hops.

Implementation outline:
- When consuming:
  - read `msg.headers()`
  - extract `traceparent`
  - create a span like `kafka.consume audio.raw` (parent = extracted ctx)
- When producing:
  - inject current context to outgoing headers
  - produce as usual with `headers=[(...)]`

Note: we currently do **not** have any OTEL code in these workers, so this will be new code (small shared helper recommended).

### Phase 4: log ↔ trace correlation
Once Phase 2/3 is in, we can correlate logs and traces cleanly.

- JVM: enable trace/span id injection into logs (Logback/SLF4J MDC via agent settings)
- Python: use `opentelemetry-instrumentation-logging` or a logging Filter to append `trace_id` / `span_id`
- Loki: query logs by `trace_id` and jump to the trace in Tempo

---

## Decisions & change log

- 2026-02-26: Use **Grafana Alloy** (not Promtail) for log shipping.
- 2026-02-26: Use Kafka header propagation (`traceparent`) for tracing v1.
- 2026-02-26: For Clojure/JVM services (`samuraibff`, `samuraipersistor`), prefer **OpenTelemetry Java Agent** over application code changes.
