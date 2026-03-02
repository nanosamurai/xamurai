# PR: Local k8s observability stack (Grafana OSS + Prometheus + Loki + Tempo) + OTEL wiring

Branch: `add-observability`

## Summary

This PR introduces a local Kubernetes **observability stack** and the first-stage wiring needed to start collecting metrics/logs/traces from the `nanosamurai-stack` services.

It also adds a clear runbook: **`docs/local-k8s-observability.md`**.

## What’s included

### Observability stack (namespace `observability`)

Installed via Helm using values files under `observability/`:

- **Grafana OSS** + **Prometheus** (`kube-prometheus-stack`)
- **Loki**
- **Tempo**
- **Grafana Alloy** (log shipping from k8s pods → Loki)
- **OpenTelemetry Collector** (central OTLP receiver → Tempo)

### App chart wiring (`charts/nanosamurai-stack`)

- Optional OTEL env injection controlled by:
  - `observability.enabled=true`
  - `observability.otlpEndpoint=...`
- Optional OTEL Java agent injection for JVM pods:
  - `observability.javaAgent.enabled=true`

### Python tracing foundation (Kafka propagation)

- Shared helper modules:
  - `shared/src/drsynth_common/otel_setup.py`
  - `shared/src/drsynth_common/otel_kafka.py`

- Worker changes to propagate W3C trace context across Kafka via header `traceparent`:
  - `whisperx_worker`
  - `recorder_worker`
  - `finalizer_worker`

### Local stability fix

- `rtservice` could CrashLoop if `HF_TOKEN` is not provided.
- Local Docker Desktop values now default to using existing secret `nanosamurai-hf`:
  - `charts/nanosamurai-stack/values.local.docker-desktop.yaml`

## What’s verified (local)

- Grafana reachable via port-forward.
- Tempo contains traces for `service.name=samuraibff`.
- `rtservice` starts once `HF_TOKEN` secret wiring is present.

See `docs/local-k8s-observability.md` for the exact verification commands and the consolidated port-forward list.

## What is *not* fully verified yet (still roadmap)

- Python services exporting spans to Tempo (SDK setup exists; end-to-end discoverability/indexing is next)
- gRPC end-to-end tracing across `samuraibff` → `rtservice`
- log ↔ trace correlation (trace_id/span_id in logs + Tempo “trace to logs” UX)

## Next steps (recommended follow-up PRs)

1) **Finish Python span exporting verification**
   - Add a tiny, explicit smoke span emission on worker startup (or a small `utilities/` script) and verify it shows in Tempo.
   - If needed, adjust OTEL exporter configuration / ensure proper shutdown/flush.

2) **Instrument `rtservice` (Python gRPC)**
   - Add `opentelemetry-instrumentation-grpc` or manual interceptors.
   - Goal: one trace from BFF to rtservice.

3) **Add semantic spans around the Kafka pipeline**
   - `kafka.consume` / `kafka.produce` spans per topic
   - span attributes: `nanosamurai.session_id`, `nanosamurai.tenant_id`, chunk sizes, model latency

4) **Logs ↔ traces correlation**
   - JVM: configure trace/span id injection to logs.
   - Python: add logging filter/instrumentation.
   - Verify Grafana “trace to logs” workflow.

## How to test locally (quick)

1) Bring up infra:

```bash
docker compose up -d broker kafka_init postgres db_migrate db_seed localstack
```

2) Install observability stack (namespace `observability`) per `docs/local-k8s-observability.md`.

3) Ensure HF token secret exists (recommended):

```bash
kubectl -n default create secret generic nanosamurai-hf --from-literal=HF_TOKEN="$HF_TOKEN"
```

4) Deploy apps + enable OTEL:

```bash
helm upgrade --install nanosamurai-stack ./charts/nanosamurai-stack \
  -f ./charts/nanosamurai-stack/values.local.docker-desktop.yaml \
  --set observability.enabled=true \
  --set observability.otlpEndpoint=http://otel-collector-opentelemetry-collector.observability.svc.cluster.local:4317 \
  --set observability.javaAgent.enabled=true
```

5) Port-forward Grafana + BFF and generate traffic using existing smoke tests.

---

## Notes for reviewers

- This is intentionally phased to keep the stack debuggable.
- The runbook explicitly calls out what’s implemented and what’s just roadmap.
