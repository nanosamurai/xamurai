"""Model-independent processing of one selected completed-recording track."""

from __future__ import annotations

import logging
import math
import socket
import time
import uuid

from proto_gen import stream_pb2 as pb
from drsynth_common.final_tracks import (
    AudioInput, ContractError, MAX_TRANSCRIPT_BYTES, RunContext, canonical_json,
    identities, validate_recording,
)
from drsynth_common.stream_controls import parse_stream_controls_from_kafka_headers

logger = logging.getLogger(__name__)


def normalize_transcript(result, duration):
    """Check output bounds without inventing unavailable timing or speakers."""
    if not isinstance(result.text, str) or not isinstance(result.segments, list):
        raise ContractError("invalid_provider_output")
    segments = []
    for segment in result.segments:
        if not isinstance(segment.get("text"), str):
            raise ContractError("invalid_provider_output")
        clean = {"text": segment["text"]}
        for start, end in [("start_s", "end_s")]:
            if start in segment or end in segment:
                s, e = segment.get(start), segment.get(end)
                if (not isinstance(s, (int, float)) or not isinstance(e, (int, float))
                        or not math.isfinite(s) or not math.isfinite(e)
                        or not 0 <= s <= e <= duration + 0.01):
                    raise ContractError("invalid_provider_timing")
                clean.update(start_s=float(s), end_s=float(e))
        if segment.get("speaker"):
            clean["speaker"] = str(segment["speaker"])
        words = []
        for word in segment.get("words") or []:
            s, e = word.get("start_s"), word.get("end_s")
            if (not isinstance(word.get("text"), str)
                    or not isinstance(s, (int, float)) or not isinstance(e, (int, float))
                    or not math.isfinite(s) or not math.isfinite(e)
                    or not 0 <= s < e <= duration + 0.01):
                raise ContractError("invalid_provider_timing")
            words.append(dict(text=word["text"], start_s=float(s), end_s=float(e)))
        if words:
            clean["words"] = words
        segments.append(clean)
    value = dict(schema_version=1, full_text=result.text, segments=segments,
                 lang=result.language, capabilities=result.capabilities,
                 provenance=result.provenance, degradations=result.degradations)
    if len(canonical_json(value)) > MAX_TRANSCRIPT_BYTES:
        raise ContractError("transcript_too_large")
    return value


def primary_projection(event, transcript):
    """Produce the legacy primary representation from accepted normalized data."""
    if not event.primary or event.status != "succeeded":
        return None
    legacy = pb.SessionTranscript(
        session_id=event.session_id, tenant_id=event.tenant_id,
        recording_url=event.source.storage_uri, lang=event.lang,
        duration_s=event.source.sample_count / event.source.sample_rate,
        full_text=transcript["full_text"], created_at_ns=event.created_at_ns)
    for segment in transcript["segments"]:
        if "start_s" not in segment or "end_s" not in segment:
            # Full text remains available; absent timing is not projected as 0.
            continue
        out = legacy.segments.add(start_s=segment["start_s"], end_s=segment["end_s"],
                                  text=segment["text"], speaker=segment.get("speaker", ""))
        for word in segment.get("words", []):
            out.words.add(**word)
    data = legacy.SerializeToString(deterministic=True)
    if len(data) > MAX_TRANSCRIPT_BYTES:
        raise ContractError("primary_projection_too_large")
    return data


def process_recording(event, headers, *, track_id, profile_id, provider, store, attempts=3):
    """Return immutable outcome bytes, or None when this track is unselected.

    Bad routing/identity fails visibly before storage. Storage outages escape
    without committing. Bounded provider failures become durable outcomes.
    Existing manifests are replayed without rerunning the provider.
    """
    plan = validate_recording(event, headers)
    controls = parse_stream_controls_from_kafka_headers(headers)
    if not controls.want_final or not controls.store_recording:
        raise ContractError("unsupported_retention_or_outputs")
    selection = next((s for s in plan["final_tracks"]
                      if (s["track_id"], s["profile_id"]) == (track_id, profile_id)), None)
    if selection is None:
        return None
    store.validate_source(event.source, event.tenant_id, event.session_id)
    run_id, result_id = identities(plan, event.source, selection)
    outcome = pb.FinalTrackResult(
        schema_version=1, session_id=event.session_id, tenant_id=event.tenant_id,
        stage="final", track_id=track_id, profile_id=profile_id, run_id=run_id,
        result_id=result_id, unit_id="recording", revision=1, source=event.source,
        primary=selection["primary"], plan_id=plan["plan_id"])
    existing = store.read_outcome(outcome)
    if existing is not None:
        logger.info("Replaying final track outcome track=%s result_id=%s", track_id, result_id)
        return existing
    started = time.monotonic()
    result = None
    error_code = ""
    try:
        with store.audio_file(event.source, event.tenant_id, event.session_id) as path:
            audio = AudioInput(path, event.source, 0, event.source.sample_count)
            for attempt in range(max(1, min(attempts, 3))):
                outcome.attempt_id = str(uuid.uuid4())
                context = RunContext(event.tenant_id, event.session_id, track_id,
                                     profile_id, run_id, outcome.attempt_id, event.lang)
                try:
                    result = provider.process(audio, context)
                    if provider.describe()["profile_id"] != profile_id:
                        raise ContractError("profile_mismatch")
                    result = (result, normalize_transcript(result, event.duration_s))
                    break
                except ContractError:
                    raise
                except Exception as error:
                    result = None
                    logger.warning("Final track inference failed track=%s attempt=%s kind=%s",
                                   track_id, attempt + 1, type(error).__name__)
                    error_code = "inference_failed"
    except ContractError as error:
        error_code = error.code
        result = None
    outcome.created_at_ns = time.time_ns()
    if not outcome.attempt_id:
        outcome.attempt_id = str(uuid.uuid4())
    legacy = None
    execution_metadata = {
        "profile_id": profile_id,
        "worker_instance": socket.gethostname(),
        "processing_seconds": round(time.monotonic() - started, 6),
        "queue_wait_seconds": max(0, (time.time_ns() - event.created_at_ns) / 1e9
                                  - (time.monotonic() - started)),
    }
    if result is None:
        outcome.status = "failed"
        outcome.error_code = error_code or "inference_failed"
        outcome.provenance_json = canonical_json(execution_metadata).decode()
    else:
        model_result, transcript = result
        outcome.status = "succeeded"
        outcome.lang = model_result.language
        outcome.degradations.extend(model_result.degradations)
        for capability in ("segment_timestamps", "word_timestamps", "speaker_labels"):
            setattr(outcome, capability, bool(model_result.capabilities.get(capability, False)))
        provenance = dict(model_result.provenance)
        provenance.update(execution_metadata)
        transcript["provenance"] = provenance
        outcome.provenance_json = canonical_json(provenance).decode()
        legacy = primary_projection(outcome, transcript)
        outcome.result_uri, outcome.result_sha256 = store.put_transcript(outcome, transcript)
    accepted = store.publish_once(outcome, legacy)
    logger.info("Final track outcome ready track=%s result_id=%s status=%s",
                track_id, result_id, accepted[0].status)
    return accepted


def outcome_headers(outcome):
    """Stable projection identity and source metadata for idempotent SQL writes."""
    return [(key, str(value).encode("utf-8")) for key, value in {
        "x-result-id": outcome.result_id, "x-source-artifact-id": outcome.source.artifact_id,
        "x-source-sha256": outcome.source.sha256, "x-source-size-bytes": outcome.source.size_bytes,
        "x-source-sample-count": outcome.source.sample_count,
        "x-source-sample-rate": outcome.source.sample_rate,
        "x-source-version-id": outcome.source.version_id,
        "x-asr-plan-id": outcome.plan_id, "x-final-track-id": outcome.track_id,
        "x-run-id": outcome.run_id,
        "x-final-profile-id": outcome.profile_id,
    }.items()]
