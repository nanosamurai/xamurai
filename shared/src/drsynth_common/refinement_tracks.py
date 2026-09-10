"""Bounded refinement window identities shared by assembly and track execution."""

import uuid
from drsynth_common.final_tracks import ContractError, canonical_json, validate_recording

WINDOW_TOPIC = "audio.refinement-windows"
RESULT_TOPIC = "transcripts.refined-tracks"


def selected_audio(raw, headers):
    """Legacy refinement must not process audio owned by a frozen track plan."""
    from proto_gen import stream_pb2 as pb
    from drsynth_common.final_tracks import read_plan
    event = pb.AudioChunk.FromString(raw)
    plan = read_plan(headers, event.tenant_id, event.session_id)
    return bool(plan and plan.get("refinement_tracks"))


def unit_id(window):
    """A fixed policy and half-open ownership range identify one replaceable unit."""
    return f"fixed-{window.window_samples}:{window.start_sample}:{window.end_sample}"


def identities(plan, window, selection):
    """Keep one run per generation/track; revision one is stable across replays."""
    parts = [plan["tenant_id"], plan["session_id"], plan["plan_id"], "refined",
             selection["track_id"], selection["profile_id"]]
    run = uuid.uuid5(uuid.NAMESPACE_URL, canonical_json(parts).decode())
    return str(run), str(uuid.uuid5(run, unit_id(window) + ":1"))


def source_id(plan, window):
    """All tracks share the same source for a window in a frozen generation."""
    return str(uuid.uuid5(uuid.UUID(plan["plan_id"]), unit_id(window)))


def validate_window(window, headers):
    """Verify immutable source, generation, range and policy before storage access."""
    event = window.recording
    plan = validate_recording(event, headers)
    size = window.window_samples
    length = window.end_sample - window.start_sample
    if (not plan.get("refinement_tracks") or size != plan["refinement_window_samples"]
            or not 0 <= window.start_sample < window.end_sample <= 9600000
            or window.start_sample % size or not 0 < length <= size
            or length != event.source.sample_count
            or window.flush_reason not in ("slice", "eof", "idle")
            or (window.flush_reason == "slice" and length != size)
            or source_id(plan, window) != event.source.artifact_id
            or len(window.bff_origin_uri.encode()) > 2048):
        raise ContractError("invalid_refinement_window")
    return plan
