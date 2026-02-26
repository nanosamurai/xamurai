"""Minimal OpenTelemetry SDK setup for Python services.

We intentionally keep this small and explicit (no auto-instrumentation CLI) so
it works the same in k8s, local tests, and in workers.

- Reads standard OTEL env vars (OTEL_SERVICE_NAME, OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
- Configures OTLP exporter (gRPC) to the in-cluster collector
- Installs W3C TraceContext propagator by default

This module is safe to import even if opentelemetry deps are missing; setup()
will become a no-op in that case.
"""

from __future__ import annotations

import os
from typing import Optional


def setup_otel(*, service_name: Optional[str] = None) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    except Exception:
        # Degrade gracefully if deps are not installed.
        return

    # If already configured, don't reconfigure (idempotent)
    prov = trace.get_tracer_provider()
    if isinstance(prov, TracerProvider):
        return

    sn = service_name or os.getenv("OTEL_SERVICE_NAME") or "unknown-service"

    # Resource attributes: keep it simple, but include service.name
    resource = Resource.create({"service.name": sn})

    provider = TracerProvider(resource=resource)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        # OTEL not configured.
        return

    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        insecure=True,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Ensure W3C propagation
    set_global_textmap(TraceContextTextMapPropagator())
