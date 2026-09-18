"""Model-independent window inference, publication and replay-safe Kafka runtime."""
import logging
import os
import re
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
from confluent_kafka import Consumer, Producer
from proto_gen import stream_pb2
from drsynth_common.otel_setup import setup_otel
from drsynth_common.otel_kafka import extracted_context_from_headers, with_current_trace_context
from drsynth_common.logging_setup import setup_logging
from xamurai_serving.refinement_runtime import run_decoupled

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
SLICE_SECONDS = float(os.getenv("REFINEMENT_SLICE_SECONDS", os.getenv("WHISPERX_SLICE_SECONDS", "60.0")))
SESSION_IDLE_SEC = float(os.getenv("REFINEMENT_IDLE_SECONDS", os.getenv("WHISPERX_IDLE_SECONDS", "30.0")))


def _run_inference_and_publish(*, job, producer, transcribe, model):
    """Publish one sample-counted window; model segment/word times are WAV-relative."""
    pcm = job.get("pcm16")
    if not job.get("session_id") or not isinstance(pcm, np.ndarray) or not pcm.size:
        return
    base_start = float(job.get("base_start_s") or 0.0)
    duration = pcm.size / SR
    tracer = trace.get_tracer(__name__) if trace is not None else None
    with tempfile.TemporaryDirectory(prefix="refinement-") as tmp:
        wav_path = str(Path(tmp) / "window.wav")
        sf.write(wav_path, pcm.astype(np.float32) / 32768.0, SR, subtype="PCM_16")
        with extracted_context_from_headers(job.get("trace_headers")), (
            tracer.start_as_current_span("refinement.slice") if tracer else nullcontext()
        ) as span:
            if span is not None:
                for key, value in {
                    "session_id": job["session_id"], "tenant_id": job.get("tenant_id") or "",
                    "track_id": TRACK_ID, "slice_index": job.get("slice_index", 0),
                    "slice_start_s": base_start, "slice_duration_s": duration,
                    "flush_reason": job.get("flush_reason", "slice"),
                }.items():
                    span.set_attribute(f"nanosamurai.{key}", value)
            text, segments = transcribe(wav_path, tenant=job.get("tenant_id"), lang=job.get("lang"))
            seg_msgs = [stream_pb2.SessionTranscriptSegment(**segment) for segment in segments]
            for segment in seg_msgs:
                for timing in (segment, *segment.words):
                    timing.start_s += base_start
                    timing.end_s += base_start
            if span is not None:
                span.set_attribute("nanosamurai.segments_count", len(seg_msgs))
                span.set_attribute("nanosamurai.text_len", len(text or ""))
            event = stream_pb2.RefinedEvent(
                session_id=job["session_id"], tenant_id=job.get("tenant_id") or "",
                start_s=base_start, end_s=base_start + duration,
                text=" ".join(s.text for s in seg_msgs if s.text.strip()).strip() or text or "",
                lang=job.get("lang") or "", bff_origin_uri=job.get("bff_origin_uri") or "",
                window_sec=float(job.get("window_sec") or duration),
                slice_index=int(job.get("slice_index") or 0),
                flush_reason=job.get("flush_reason") or "slice", segments=seg_msgs,
                created_at_ns=time.time_ns(), refinement_model=model, track_id=TRACK_ID,
            )
            with tracer.start_as_current_span("kafka.produce transcripts.refined") if tracer else nullcontext():
                delivery = []
                producer.produce(
                    topic=TOPIC_REFINED, key=job["session_id"].encode(), value=event.SerializeToString(),
                    headers=with_current_trace_context(),
                    on_delivery=lambda error, _message: delivery.append(error),
                )
                if producer.flush(30) or not delivery or delivery[0] is not None:
                    raise RuntimeError("Refinement Kafka publication was not acknowledged")


def make_consumer():
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
        cfg["security.protocol"] = KAFKA_SECURITY_PROTOCOL
        if KAFKA_SSL_CA_LOCATION:
            cfg["ssl.ca.location"] = KAFKA_SSL_CA_LOCATION
    return Consumer(cfg)


def make_producer():
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


def main(transcribe, *, model):
    """Consume selected audio using a warm pipeline returning text and segment dictionaries."""
    setup_logging(default_level="INFO")
    setup_otel(service_name=os.getenv("OTEL_SERVICE_NAME", f"{TRACK_ID}-refinement"))
    logger.info("Refinement pipeline ready: track=%s model=%s", TRACK_ID, model)
    producer = make_producer()
    run_decoupled(
        consumer=make_consumer(), topic_audio=TOPIC_AUDIO, slice_seconds=SLICE_SECONDS,
        sample_rate=SR, session_idle_sec=SESSION_IDLE_SEC, track_id=TRACK_ID,
        parse_audio_chunk=stream_pb2.AudioChunk.FromString,
        run_inference_and_publish=lambda job: _run_inference_and_publish(
            job=job, producer=producer, transcribe=transcribe, model=model),
    )
