"""Versioned final-track identities and header validation, without ML imports.

The BFF freezes the plan; every subsequent hop verifies its tenant/session and
the redundant routing headers. Topic names are intentionally unversioned.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from proto_gen import stream_pb2 as pb

SCHEMA_VERSION = 1
FINAL_TRACK_TOPIC = "transcripts.final-tracks"
PLAN_HEADER = "x-asr-plan"
PLAN_ID_HEADER = "x-asr-plan-id"
TRACKS_HEADER = "x-final-track-ids"
MAX_PLAN_BYTES = 8192
MAX_EVENT_BYTES = 65536
INLINE_SCHEMA_VERSION = 2
# Leave room for the Kafka key, bounded tracing headers and record/batch framing.
MAX_RESULT_BYTES = 900_000
MAX_RECORD_BYTES = 1_000_000
MAX_AUDIO_SECONDS = 600
NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class ContractError(ValueError):
    """A sanitized, permanent input error; never includes input content."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def canonical_json(value: Any) -> bytes:
    """Serialize bounded contract/artifact values deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def require_uuid(value: str) -> str:
    """Require a canonical UUID, suitable for tenant-scoped object keys."""
    try:
        parsed = str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        raise ContractError("invalid_identity") from None
    if parsed != value:
        raise ContractError("invalid_identity")
    return parsed


def require_name(value: str) -> str:
    """Validate an operator-selected deployment identity."""
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ContractError("invalid_profile_identity")
    return value


def validate_plan(value: dict, tenant_id: str, session_id: str) -> dict:
    """Validate a frozen JSON plan against its authenticated event identity."""
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "plan_id", "tenant_id", "session_id", "final_tracks"
    }:
        raise ContractError("invalid_plan")
    if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ContractError("unsupported_plan_version")
    for name in ("tenant_id", "session_id", "plan_id"):
        require_uuid(value[name])
    if value["tenant_id"] != tenant_id or value["session_id"] != session_id:
        raise ContractError("plan_owner_mismatch")
    selections = value["final_tracks"]
    if not isinstance(selections, list) or not 1 <= len(selections) <= 4:
        raise ContractError("invalid_track_count")
    for selection in selections:
        if not isinstance(selection, dict) or set(selection) != {"track_id", "profile_id", "primary"}:
            raise ContractError("invalid_selection")
        if any(not isinstance(selection[k], str) or not NAME.fullmatch(selection[k])
               for k in ("track_id", "profile_id")):
            raise ContractError("invalid_profile_identity")
        if type(selection["primary"]) is not bool:
            raise ContractError("invalid_primary")
    if len({s["track_id"] for s in selections}) != len(selections):
        raise ContractError("duplicate_track")
    if sum(s["primary"] for s in selections) != 1:
        raise ContractError("invalid_primary")
    if len(canonical_json(value)) > MAX_PLAN_BYTES:
        raise ContractError("plan_too_large")
    return value


def plan_headers(plan: dict) -> list[tuple[str, bytes]]:
    """Build the three routing headers from one validated frozen plan."""
    validate_plan(plan, plan["tenant_id"], plan["session_id"])
    return [(PLAN_HEADER, canonical_json(plan)),
            (PLAN_ID_HEADER, plan["plan_id"].encode()),
            (TRACKS_HEADER, ",".join(s["track_id"] for s in plan["final_tracks"]).encode())]


def read_plan(headers, tenant_id: str, session_id: str) -> dict | None:
    """Read strict UTF-8 headers; absent means legacy, partial means invalid."""
    wanted = {PLAN_HEADER, PLAN_ID_HEADER, TRACKS_HEADER}
    found = {}
    for key, value in headers or ():
        if key in wanted:
            if key in found or not isinstance(value, bytes) or len(value) > MAX_PLAN_BYTES:
                raise ContractError("invalid_plan_headers")
            found[key] = value
    if not found:
        return None
    if set(found) != wanted:
        raise ContractError("incomplete_plan_headers")
    try:
        plan = json.loads(found[PLAN_HEADER].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ContractError("invalid_plan_json") from None
    validate_plan(plan, tenant_id, session_id)
    if dict(plan_headers(plan)) != found:
        # Require canonical JSON too, eliminating ambiguity across languages.
        raise ContractError("plan_header_mismatch")
    return plan


def plan_proto(plan: dict) -> pb.AsyncTrackPlan:
    """Convert the validated JSON snapshot into the additive protobuf plan."""
    validate_plan(plan, plan["tenant_id"], plan["session_id"])
    return pb.AsyncTrackPlan(**plan)


def plan_dict(plan: pb.AsyncTrackPlan) -> dict:
    """Convert protobuf without losing explicit false primary values."""
    return dict(schema_version=plan.schema_version, plan_id=plan.plan_id,
                tenant_id=plan.tenant_id, session_id=plan.session_id,
                final_tracks=[dict(track_id=t.track_id, profile_id=t.profile_id,
                                   primary=t.primary) for t in plan.final_tracks])


def validate_recording(event: pb.RecordingFinished, headers) -> dict:
    """Require matching typed/header plans and a bounded immutable WAV source."""
    plan = read_plan(headers, event.tenant_id, event.session_id)
    if plan is None or not event.HasField("source") or not event.HasField("final_plan"):
        raise ContractError("missing_final_track_metadata")
    if plan != plan_dict(event.final_plan):
        raise ContractError("plan_payload_mismatch")
    source = event.source
    require_uuid(source.artifact_id)
    if not SHA256.fullmatch(source.sha256):
        raise ContractError("invalid_audio_digest")
    if (source.media_type != "audio/wav" or source.sample_rate != 16000
            or not 0 < source.sample_count <= MAX_AUDIO_SECONDS * 16000
            or not 44 <= source.size_bytes <= MAX_AUDIO_SECONDS * 32000 + 4096
            or event.recording_url != source.storage_uri
            or event.sample_rate != source.sample_rate
            or abs(event.duration_s - source.sample_count / source.sample_rate) > 0.001):
        raise ContractError("invalid_audio_metadata")
    return plan


def identities(plan: dict, source: pb.AudioArtifact, selection: dict) -> tuple[str, str]:
    """Derive replay-stable run/result UUIDs, scoped to tenant and source."""
    parts = [plan["tenant_id"], plan["session_id"], plan["plan_id"],
             source.artifact_id, source.sha256, "final",
             selection["track_id"], selection["profile_id"]]
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, canonical_json(parts).decode()))
    return run_id, str(uuid.uuid5(uuid.UUID(run_id), "recording:1"))


def file_digest(path: Path) -> str:
    """SHA-256 a file without loading a recording/model into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class AudioInput:
    """Validated local WAV plus immutable source and half-open sample range."""
    path: Path
    source: pb.AudioArtifact
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class RunContext:
    """Execution identity and language; does not carry infrastructure clients."""
    tenant_id: str
    session_id: str
    track_id: str
    profile_id: str
    run_id: str
    attempt_id: str
    language: str
    stage: str = "final"


@dataclass
class TranscriptResult:
    """Normalized output; absent enrichment must be reported explicitly."""
    text: str
    segments: list[dict]
    language: str
    capabilities: dict[str, bool]
    provenance: dict
    degradations: list[str] = field(default_factory=list)


class BatchTrack(Protocol):
    """Small in-process seam for a complete bounded speech pipeline."""
    def describe(self) -> dict: ...
    def process(self, audio: AudioInput, context: RunContext) -> TranscriptResult: ...
