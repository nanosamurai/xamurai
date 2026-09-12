"""Model-independent processing of one selected completed-recording track."""

from __future__ import annotations

import logging
import math
import time
import uuid

from proto_gen import stream_pb2 as pb
from drsynth_common.final_tracks import (
    AudioInput, ContractError, INLINE_SCHEMA_VERSION, MAX_RESULT_BYTES, RunContext,
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
    value = dict(full_text=result.text, segments=segments,
                 lang=result.language, capabilities=result.capabilities,
                 degradations=result.degradations)
    return value


def process_recording(event, headers, *, track_id, profile_id, provider, store, attempts=3):
    """Return an inline terminal result, or None when this track is unselected.

    Bad routing/identity fails visibly before storage. Storage outages escape
    without committing. Bounded provider failures become durable outcomes.
    Replay can repeat inference. Persistor chooses the first accepted outcome.
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
        schema_version=INLINE_SCHEMA_VERSION, session_id=event.session_id, tenant_id=event.tenant_id,
        track_id=track_id, profile_id=profile_id, result_id=result_id,
        source=event.source, plan_id=plan["plan_id"])
    result = None
    error_code = ""
    try:
        with store.audio_file(event.source, event.tenant_id, event.session_id) as path:
            audio = AudioInput(path, event.source, 0, event.source.sample_count)
            for attempt in range(max(1, min(attempts, 3))):
                context = RunContext(event.tenant_id, event.session_id, track_id,
                                     profile_id, run_id, str(uuid.uuid4()), event.lang)
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
    if result is None:
        outcome.status = "failed"
        outcome.error_code = error_code or "inference_failed"
    else:
        model_result, transcript = result
        outcome.status = "succeeded"
        outcome.lang = model_result.language
        outcome.degradations.extend(model_result.degradations)
        segments = transcript["segments"]
        outcome.full_text = transcript["full_text"]
        outcome.segments.extend(pb.SessionTranscriptSegment(**segment) for segment in segments)
        outcome.segment_timestamps = bool(segments) and all("start_s" in s for s in segments)
        outcome.word_timestamps = any(s.get("words") for s in segments)
        outcome.speaker_labels = any(s.get("speaker") for s in segments)
    if outcome.ByteSize() > MAX_RESULT_BYTES:
        outcome.ClearField("full_text")
        outcome.ClearField("segments")
        outcome.ClearField("degradations")
        outcome.ClearField("lang")
        outcome.segment_timestamps = outcome.word_timestamps = outcome.speaker_labels = False
        outcome.status, outcome.error_code = "failed", "result_too_large"
    logger.info("Final track outcome ready track=%s result_id=%s status=%s",
                track_id, result_id, outcome.status)
    return outcome
