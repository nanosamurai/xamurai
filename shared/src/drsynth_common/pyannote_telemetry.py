"""pyannote telemetry hardening.

pyannote.audio (v4+) includes an optional telemetry/metrics exporter that, by
default, sends anonymous usage metrics to an upstream OTLP endpoint.

In xamurai we **force-disable** this feature for security reasons (avoid any
unexpected outbound connections).

Upstream knob:
  PYANNOTE_METRICS_ENABLED=0

We set this env var as early as possible in each process entrypoint.
"""

from __future__ import annotations

import os


def disable_pyannote_telemetry() -> None:
    """Force-disable pyannote.audio telemetry.

    This is intentionally unconditional (no override) to guarantee no egress.
    """

    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
