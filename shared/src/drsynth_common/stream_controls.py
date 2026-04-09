"""Stream-level control knobs transported via gRPC metadata / Kafka headers.

This module defines a small, dependency-light parsing layer for configuration
that originates from the BFF / client stream and is forwarded downstream.

Motivation
----------
We support multiple transcript signal layers:

- realtime: rtservice gRPC AsrEvent (PARTIAL + FINAL)
- refined:  Kafka transcripts.refined (RefinedEvent)
- final:    Kafka transcripts.final (SessionTranscript)

End users may want only a subset. Additionally, storing the session recording
(WAV / S3 object) is an independent concern from producing the final transcript.

Transport
---------
For workers consuming Kafka, we rely on Kafka headers.

Headers (proposed Phase 1)
--------------------------
- x-outputs: CSV set, e.g. "realtime,refined,final"
  - recognized tokens: realtime, refined, final
  - if header is missing/empty => treat as "all" (backwards compatible)

- x-store-recording: boolean ("true"/"false"/"1"/"0")
  - when false, the pipeline should delete the recording after final transcript
    was produced successfully.
  - if header is missing => default true (backwards compatible)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

KafkaHeader = Tuple[str, Optional[bytes]]


def _as_text(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    return str(v)


def _header_value_opt(headers: Optional[Sequence[KafkaHeader]], key: str) -> Optional[str]:
    """Return header value (stripped) or None when key not present."""

    if not headers:
        return None
    key_l = key.lower()
    for k, v in headers:
        if str(k).lower() == key_l:
            return _as_text(v).strip()
    return None


def _parse_bool(s: str, *, default: bool) -> bool:
    if s is None:
        return bool(default)
    raw = str(s).strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


_KNOWN_OUTPUTS = {"realtime", "refined", "final"}


@dataclass(frozen=True)
class StreamControls:
    want_realtime: bool = True
    want_refined: bool = True
    want_final: bool = True
    store_recording: bool = True

    @property
    def want_any(self) -> bool:
        return bool(self.want_realtime or self.want_refined or self.want_final)


def parse_stream_controls_from_kafka_headers(headers: Optional[Sequence[KafkaHeader]]) -> StreamControls:
    """Parse StreamControls from Kafka headers.

    Backwards compatibility rules:
    - Missing x-outputs => all outputs enabled.
    - Missing x-store-recording => True.
    """

    outputs_opt = _header_value_opt(headers, "x-outputs")
    store_opt = _header_value_opt(headers, "x-store-recording")

    # Backwards compat: header missing => default ALL.
    # Explicit but empty header => interpret as NONE (lets BFF disable everything if desired).
    if outputs_opt is None:
        want = set(_KNOWN_OUTPUTS)
        want_realtime = True
        want_refined = True
        want_final = True
    else:
        want = {p.strip().lower() for p in (outputs_opt or "").split(",") if p.strip()}
        want = {w for w in want if w in _KNOWN_OUTPUTS}
        want_realtime = "realtime" in want
        want_refined = "refined" in want
        want_final = "final" in want

    return StreamControls(
        want_realtime=want_realtime,
        want_refined=want_refined,
        want_final=want_final,
        store_recording=_parse_bool(store_opt or "", default=True),
    )
