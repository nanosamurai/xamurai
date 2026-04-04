"""Common logging setup for xamurai Python services.

We control verbosity via env var so Kubernetes/Helm can configure it without
rebuilding images.

Env vars:
- LOG_LEVEL: standard Python level name (DEBUG|INFO|WARNING|ERROR|CRITICAL)
  Defaults to INFO.

Observability:
- We inject OpenTelemetry trace_id/span_id into every log record when OTEL is
  configured, so Loki logs can be correlated with Tempo traces.
- We also support injecting a per-thread/per-request `session_id` (contextvar)
  into logs so operators can correlate without parsing arbitrary messages.

Log record fields added:
- trace_id: 32-hex lowercase trace id (or "-" when not available)
- span_id:  16-hex lowercase span id (or "-")
- session_id: session id (or "-")
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar


_session_id_var: ContextVar[str] = ContextVar("xamurai_session_id", default="-")


def set_session_id_for_logging(session_id: str | None) -> None:
    """Set current session_id for log correlation.

    This is intentionally lightweight. Callers should set it at stream/request
    start and reset to "-" on exit.
    """

    sid = (session_id or "-").strip() or "-"
    _session_id_var.set(sid)


def _safe_current_otel_ids() -> tuple[str, str]:
    """Return (trace_id, span_id) for the current OTEL span.

    Works even when opentelemetry is not installed/configured.
    """

    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = getattr(span, "get_span_context", lambda: None)()
        if ctx is None:
            return "-", "-"
        # If span context is invalid (no active span), keep dashes.
        if not getattr(ctx, "is_valid", False):
            return "-", "-"

        # OTEL uses ints; format as fixed-width hex to match Tempo UI.
        tid = getattr(ctx, "trace_id", 0)
        sid = getattr(ctx, "span_id", 0)
        if not tid or not sid:
            return "-", "-"
        return f"{int(tid):032x}", f"{int(sid):016x}"
    except Exception:
        return "-", "-"


def _install_log_record_factory() -> None:
    """Install a LogRecordFactory that enriches records with trace/span/session ids."""

    # Idempotent installation.
    if getattr(logging, "_xamurai_log_factory_installed", False):  # type: ignore[attr-defined]
        return

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        trace_id, span_id = _safe_current_otel_ids()

        # Ensure the attributes always exist so formatters never KeyError.
        setattr(record, "trace_id", trace_id)
        setattr(record, "span_id", span_id)
        setattr(record, "session_id", _session_id_var.get())
        return record

    logging.setLogRecordFactory(record_factory)
    setattr(logging, "_xamurai_log_factory_installed", True)  # type: ignore[attr-defined]


def _parse_log_level(value: str | None) -> int:
    raw = (value or "").strip().upper()
    if not raw:
        return logging.INFO

    # Accept common aliases.
    if raw == "WARN":
        raw = "WARNING"

    level = getattr(logging, raw, None)
    if isinstance(level, int):
        return level

    # Fallback: keep logs readable even if misconfigured.
    return logging.INFO


def setup_logging(*, default_level: str = "INFO") -> int:
    """Configure stdlib logging based on env var.

    Returns the effective logging level (int).
    """

    level = _parse_log_level(os.getenv("LOG_LEVEL", default_level))

    _install_log_record_factory()

    # `basicConfig` is a no-op if handlers are already configured.
    # For our container entrypoints we want deterministic behavior.
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s [%(levelname)s] %(name)s "
            "trace_id=%(trace_id)s span_id=%(span_id)s session=%(session_id)s: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    return level
