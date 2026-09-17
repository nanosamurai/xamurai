"""Refine audio windows with the shared WhisperX pipeline."""
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from confluent_kafka import Consumer, Producer
from proto_gen import stream_pb2
from drsynth_common.otel_setup import setup_otel
from drsynth_common.otel_kafka import extracted_context_from_headers, with_current_trace_context
from drsynth_common.logging_setup import setup_logging
from whisperx_worker import pipeline
from whisperx_worker.decoupled_runtime import run_decoupled

try:
    from opentelemetry import trace
except ImportError:
    trace = None

logger = logging.getLogger(__name__)
SR = 16000
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").strip().upper()
KAFKA_SSL_CA_LOCATION = os.getenv("KAFKA_SSL_CA_LOCATION", "").strip()
TOPIC_AUDIO = os.getenv("KAFKA_TOPIC_AUDIO", "audio.raw")
TOPIC_REFINED = os.getenv("KAFKA_TOPIC_REFINED", "transcripts.refined")
TRACK_ID = os.getenv("REFINEMENT_TRACK_ID", "whisperx").strip()
if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", TRACK_ID):
    raise ValueError("Invalid REFINEMENT_TRACK_ID")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", f"refinement.{TRACK_ID}")

SLICE_SECONDS = float(os.getenv("WHISPERX_SLICE_SECONDS", "60.0"))
SESSION_IDLE_SEC = float(os.getenv("WHISPERX_IDLE_SECONDS", "30.0"))


def _run_inference_and_publish(*, job: Dict[str, Any], producer: Producer) -> None:
    """Inference path for decoupled runtime (Kafka poll decoupled from GPU inference).

    NOTE: This function used to be inlined in the main loop. It exists so the
    `decoupled_runtime` module can call back into the same publish semantics.
    """

    session_id = str(job.get("session_id") or "")
    if not session_id:
        return

    pcm = job.get("pcm16")
    if not isinstance(pcm, np.ndarray) or pcm.size == 0:
        return

    base_start = float(job.get("base_start_s") or 0.0)
    window_sec = float(job.get("window_sec") or 0.0)
    lang_hint = job.get("lang") or None
    tenant = job.get("tenant_id") or None
    bff_uri = job.get("bff_origin_uri") or None
    trace_headers = job.get("trace_headers")
    flush_reason = str(job.get("flush_reason") or "slice")
    slice_index = int(job.get("slice_index") or 0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        x = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        sf.write(wav_path, x, SR, subtype="PCM_16")

        with extracted_context_from_headers(trace_headers):
            slice_span_cm = None
            if trace is not None:
                try:
                    tracer = trace.get_tracer("whisperx_worker")
                    slice_span_cm = tracer.start_as_current_span("whisperx.slice")
                    slice_span_cm.__enter__()
                    span = trace.get_current_span()
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("nanosamurai.session_id", session_id)
                        if tenant:
                            span.set_attribute("nanosamurai.tenant_id", tenant)
                        span.set_attribute("nanosamurai.slice_index", slice_index)
                        span.set_attribute("nanosamurai.slice_start_s", float(base_start))
                        span.set_attribute("nanosamurai.slice_duration_s", float(pcm.size) / float(SR))
                        span.set_attribute("nanosamurai.flush_reason", flush_reason)
                except Exception:
                    slice_span_cm = None

            try:
                text, segments = pipeline.run_whisperx_diarized(
                    wav_path,
                    tenant=tenant,
                    lang=lang_hint,
                    use_alignment=False,
                )

                if trace is not None:
                    try:
                        span = trace.get_current_span()
                        if hasattr(span, "set_attribute"):
                            span.set_attribute("nanosamurai.segments_count", int(len(segments)))
                            span.set_attribute("nanosamurai.text_len", int(len(text or "")))
                    except Exception:
                        pass

                publish_span_cm = None
                if trace is not None:
                    try:
                        tracer = trace.get_tracer("whisperx_worker")
                        publish_span_cm = tracer.start_as_current_span("kafka.produce transcripts.refined")
                        publish_span_cm.__enter__()
                    except Exception:
                        publish_span_cm = None

                try:
                    # Publish ONE refined event per refinement window slice.
                    seg_msgs: List[stream_pb2.SessionTranscriptSegment] = []
                    for (s0, s1, seg_text, speaker) in segments:
                        seg_msgs.append(
                            stream_pb2.SessionTranscriptSegment(
                                start_s=float(base_start + s0),
                                end_s=float(base_start + s1),
                                text=str(seg_text or ""),
                                speaker=str(speaker or ""),
                                words=[],
                            )
                        )

                    # Identity comes from audio samples, never model segment ends.
                    window_end_s = float(base_start) + float(pcm.size) / float(SR)

                    window_start_s = float(base_start)
                    effective_window_sec = float(window_sec) if float(window_sec) > 0 else float(pcm.size) / float(SR)
                    full_text = " ".join([s.text for s in seg_msgs if (s.text or "").strip()]).strip() or text or ""

                    ev = stream_pb2.RefinedEvent(
                        session_id=session_id,
                        # Legacy scalar fields filled for backwards compatibility.
                        start_s=window_start_s,
                        end_s=float(window_end_s),
                        text=full_text,
                        speaker="",
                        supersedes_seq=[],
                        lang=str(lang_hint or ""),
                        bff_origin_uri=str(bff_uri or ""),
                        tenant_id=str(tenant or ""),
                        window_sec=float(effective_window_sec),
                        slice_index=int(slice_index),
                        flush_reason=str(flush_reason),
                        segments=seg_msgs,
                        created_at_ns=int(time.time_ns()),
                        refinement_model=os.getenv("WHISPERX_MODEL", "medium"),
                        track_id=TRACK_ID,
                    )
                    delivery = []
                    producer.produce(
                        topic=TOPIC_REFINED,
                        key=session_id.encode("utf-8"),
                        value=ev.SerializeToString(),
                        headers=with_current_trace_context(),
                        on_delivery=lambda error, _message: delivery.append(error),
                    )
                    if producer.flush(30) or not delivery or delivery[0] is not None:
                        raise RuntimeError("Refinement Kafka publication was not acknowledged")
                finally:
                    if publish_span_cm is not None:
                        try:
                            publish_span_cm.__exit__(None, None, None)
                        except Exception:
                            pass
                producer.poll(0)
            finally:
                if slice_span_cm is not None:
                    try:
                        slice_span_cm.__exit__(None, None, None)
                    except Exception:
                        pass
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def make_consumer() -> Consumer:
    logger.info("Creating Kafka consumer")
    cfg = {
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        "client.id": os.getenv("KAFKA_CLIENT_ID", f"refinement.{TRACK_ID}"),
        "enable.auto.commit": False,
        "enable.auto.offset.store": False,
        "enable.partition.eof": True,
        "auto.offset.reset": "earliest",
        "max.partition.fetch.bytes": 5_000_000,
        "fetch.wait.max.ms": 50,
    }
    if KAFKA_SECURITY_PROTOCOL and KAFKA_SECURITY_PROTOCOL != "PLAINTEXT":
        # librdkafka expects 'SSL' (uppercase)
        cfg["security.protocol"] = KAFKA_SECURITY_PROTOCOL
        if KAFKA_SSL_CA_LOCATION:
            cfg["ssl.ca.location"] = KAFKA_SSL_CA_LOCATION
    return Consumer(cfg)


def make_producer() -> Producer:
    logger.info("Creating Kafka producer")
    cfg = {
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "client.id": f"refinement.{TRACK_ID}",
        "enable.idempotence": True,
        "compression.type": "zstd",
        "linger.ms": 10,
        "batch.size": 131072,
    }
    if KAFKA_SECURITY_PROTOCOL and KAFKA_SECURITY_PROTOCOL != "PLAINTEXT":
        cfg["security.protocol"] = KAFKA_SECURITY_PROTOCOL
        if KAFKA_SSL_CA_LOCATION:
            cfg["ssl.ca.location"] = KAFKA_SSL_CA_LOCATION
    return Producer(cfg)


def main():
    setup_logging(default_level="INFO")
    setup_otel(service_name=os.getenv("OTEL_SERVICE_NAME", "whisperx-worker"))

    logger.info("Starting whisperx_worker")

    if pipeline.torch is None:
        logger.exception("whisperx_worker: torch not installed; cannot start")
        return

    if not pipeline.torch.cuda.is_available():
        logger.warning(
            "whisperx_worker: CUDA not available; running on CPU (torch=%s torch.version.cuda=%s)",
            getattr(pipeline.torch, "__version__", "unknown"),
            getattr(getattr(pipeline.torch, "version", None), "cuda", None),
        )
    else:
        logger.info(
            "whisperx_worker: CUDA available; will use GPU (torch=%s torch.version.cuda=%s)",
            getattr(pipeline.torch, "__version__", "unknown"),
            getattr(getattr(pipeline.torch, "version", None), "cuda", None),
        )

    try:
        pipeline._init_whisperx()
    except Exception as e:
        logger.exception("Failed to initialize WhisperX at startup: %s", e)
        return

    c = make_consumer()
    p = make_producer()
    run_decoupled(
        consumer=c,
        topic_audio=TOPIC_AUDIO,
        slice_seconds=SLICE_SECONDS,
        sample_rate=SR,
        session_idle_sec=SESSION_IDLE_SEC,
        track_id=TRACK_ID,
        parse_audio_chunk=stream_pb2.AudioChunk.FromString,
        run_inference_and_publish=lambda job: _run_inference_and_publish(job=job, producer=p),
    )


if __name__ == "__main__":
    main()
