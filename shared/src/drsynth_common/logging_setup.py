"""Common logging setup for xamurai Python services.

We control verbosity via env var so Kubernetes/Helm can configure it without
rebuilding images.

Env vars:
- LOG_LEVEL: standard Python level name (DEBUG|INFO|WARNING|ERROR|CRITICAL)
  Defaults to INFO.
"""

from __future__ import annotations

import logging
import os


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

    # `basicConfig` is a no-op if handlers are already configured.
    # For our container entrypoints we want deterministic behavior.
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    return level
