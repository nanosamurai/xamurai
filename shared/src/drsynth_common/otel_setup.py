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

    # The Python gRPC exporter expects a gRPC target (host:port) or a URL.
    # In our Helm chart we set OTEL_EXPORTER_OTLP_ENDPOINT to an HTTP URL
    # (e.g. http://otel-collector...:4317) for compatibility with other SDKs.
    # Normalize it here to avoid gRPC channel target parsing issues.
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        endpoint = endpoint.split("://", 1)[1]
        endpoint = endpoint.split("/", 1)[0]

    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        insecure=True,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Ensure W3C propagation
    set_global_textmap(TraceContextTextMapPropagator())


def extract_trace_context_from_headers(headers: dict[str, str | bytes | None]):
    """Extract OTEL context from W3C TraceContext headers.

    Intended for gRPC metadata and other header-like carriers.
    Returns an OTEL context object (or None if OTEL isn't available).

    Notes:
    - Input is a simple dict-like mapping; keys are treated case-insensitively.
    - Currently we care mainly about `traceparent`.
    """

    try:
        from opentelemetry.propagate import get_global_textmap
    except Exception:
        return None

    propagator = get_global_textmap()
    if propagator is None:
        return None

    # Normalize carrier to string->string.
    carrier: dict[str, str] = {}
    for k, v in (headers or {}).items():
        if k is None:
            continue
        kk = str(k).lower()
        if v is None:
            continue
        if isinstance(v, bytes):
            try:
                carrier[kk] = v.decode("utf-8", errors="ignore")
            except Exception:
                continue
        else:
            carrier[kk] = str(v)

    try:
        return propagator.extract(carrier)
    except Exception:
        return None
