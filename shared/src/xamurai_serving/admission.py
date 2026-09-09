"""Process-local admission for stateful realtime sessions."""

from __future__ import annotations

import os
import socket
import threading


def max_sessions_from_env(default: int = 1) -> int:
    """Return the positive ``RT_SERVING_MAX_SESSIONS`` process limit."""
    raw = os.getenv("RT_SERVING_MAX_SESSIONS", str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("RT_SERVING_MAX_SESSIONS must be a positive integer") from exc
    if value < 1:
        raise ValueError("RT_SERVING_MAX_SESSIONS must be a positive integer")
    return value


def serving_instance_id() -> str:
    """Return a stable identity for this serving process."""
    return (
        os.getenv("RT_SERVING_INSTANCE_ID", "").strip()
        or os.getenv("HOSTNAME", "").strip()
        or socket.gethostname()
    )


class SessionSlots:
    """Atomically acquire and release a bounded number of process-local slots."""

    def __init__(self, maximum: int) -> None:
        if maximum < 1:
            raise ValueError("maximum session count must be positive")
        self.maximum = maximum
        self._active = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        """Return the current admitted count."""
        with self._lock:
            return self._active

    def acquire(self) -> bool:
        """Reserve one slot without blocking; return false when the process is full."""
        with self._lock:
            if self._active >= self.maximum:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        """Return one acquired slot."""
        with self._lock:
            if self._active < 1:
                raise RuntimeError("session slot released without a matching acquisition")
            self._active -= 1
