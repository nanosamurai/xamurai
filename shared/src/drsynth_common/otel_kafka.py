"""OpenTelemetry + Kafka context propagation helpers.

Goal
----
Provide a tiny, dependency-light helper that lets our Python services propagate
W3C Trace Context through Kafka headers (confluent_kafka).

Design constraints
------------------
- We use `confluent_kafka` in workers.
- We want W3C propagation via the standard `traceparent` header.
- Keep this module importable even if OTEL packages are not installed, so
  services can choose to enable it at runtime.

Usage
-----
Producer:

    headers = with_current_trace_context(existing_headers)
    producer.produce(..., headers=headers)

Consumer:

    with extracted_context_from_headers(msg.headers()):
        with tracer.start_as_current_span("kafka.consume ..."):
            ...

If OTEL isn't installed, functions degrade gracefully.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

KafkaHeader = Tuple[str, Optional[bytes]]
KafkaHeaders = List[KafkaHeader]


def _normalize_headers(headers: Optional[Sequence[KafkaHeader]]) -> KafkaHeaders:
    if not headers:
        return []
    # Make a shallow copy so callers can pass msg.headers() directly.
    return [(str(k), v) for (k, v) in headers]


def _try_get_otel_propagator():
    try:
        from opentelemetry.propagate import get_global_textmap

        return get_global_textmap()
    except Exception:
        return None


def _try_get_otel_context_api():
    try:
        from opentelemetry import context

        return context
    except Exception:
        return None


class _KafkaHeaderGetter:
    def get(self, carrier: Sequence[KafkaHeader], key: str) -> List[str]:
        key_l = key.lower()
        out: List[str] = []
        for k, v in carrier:
            if k.lower() != key_l:
                continue
            if v is None:
                continue
            try:
                out.append(v.decode("utf-8"))
            except Exception:
                # ignore undecodable headers
                continue
        return out


class _KafkaHeaderSetter:
    def set(self, carrier: KafkaHeaders, key: str, value: str) -> None:
        # Remove any existing headers of same key (case-insensitive) to prevent
        # duplicates when re-injecting.
        key_l = key.lower()
        carrier[:] = [(k, v) for (k, v) in carrier if k.lower() != key_l]
        carrier.append((key, value.encode("utf-8")))


def with_current_trace_context(headers: Optional[Sequence[KafkaHeader]] = None) -> KafkaHeaders:
    """Return headers with the current OTEL context injected.

    If OpenTelemetry isn't available, returns the original headers (copied).
    """

    base = _normalize_headers(headers)
    propagator = _try_get_otel_propagator()
    if propagator is None:
        return base

    carrier: KafkaHeaders = list(base)
    try:
        propagator.inject(carrier, setter=_KafkaHeaderSetter())
    except Exception:
        # Best-effort: if injection fails, just return base.
        return base

    return carrier


@contextmanager
def extracted_context_from_headers(headers: Optional[Sequence[KafkaHeader]]):
    """Context manager that sets current context extracted from Kafka headers."""

    hdrs = _normalize_headers(headers)
    propagator = _try_get_otel_propagator()
    ctx_api = _try_get_otel_context_api()

    if propagator is None or ctx_api is None:
        yield
        return

    try:
        ctx = propagator.extract(hdrs, getter=_KafkaHeaderGetter())
    except Exception:
        yield
        return

    token = ctx_api.attach(ctx)
    try:
        yield
    finally:
        ctx_api.detach(token)
