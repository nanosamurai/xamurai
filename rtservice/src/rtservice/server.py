import os
import logging
import math
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from concurrent import futures
from typing import Optional

import grpc

from proto_gen import stream_pb2
from proto_gen import stream_pb2_grpc
from rtservice.engine import RealtimeEngine
from rtservice.providers import SpeechProviderError

from drsynth_common.logging_setup import setup_logging, set_session_id_for_logging
from drsynth_common.otel_setup import setup_otel, extract_trace_context_from_headers
from drsynth_common.pyannote_telemetry import disable_pyannote_telemetry


# Security: pyannote.audio may try to export telemetry to a remote OTLP endpoint.
# Force-disable it as early as possible in the process.
disable_pyannote_telemetry()

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
except Exception:  # pragma: no cover
    otel_context = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
except Exception:  # pragma: no cover
    Counter = Gauge = Histogram = start_http_server = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in ("1", "true", "yes", "y")


def _install_crash_diagnostics() -> None:
    """Enable best-effort crash diagnostics for native crashes (SIGSEGV etc.)."""

    if not _bool_env("RT_FAULTHANDLER_ENABLE", default=True):
        return

    try:
        import faulthandler

        # Dump Python stack traces for all threads on fatal signals.
        faulthandler.enable(all_threads=True)

        # Allow on-demand dump of all thread stacks from inside the container:
        #   kill -USR1 <pid>
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        except Exception:
            pass

        logger.info("rtservice: faulthandler enabled (all_threads=true)")
    except Exception:
        logger.warning("rtservice: failed to enable faulthandler", exc_info=True)


def _install_excepthook() -> None:
    def _hook(tp, value, tb):
        logger.critical("Uncaught exception", exc_info=(tp, value, tb))

    sys.excepthook = _hook


def _maybe_start_metrics_server() -> None:
    if start_http_server is None:
        logger.info("rtservice metrics: prometheus_client not installed; /metrics disabled")
        return

    if not _bool_env("RT_METRICS_ENABLE", default=True):
        logger.info("rtservice metrics: disabled (RT_METRICS_ENABLE=false)")
        return

    port = int(os.getenv("RT_METRICS_PORT", "8008"))
    addr = os.getenv("RT_METRICS_ADDR", "127.0.0.1").strip() or "127.0.0.1"

    # NOTE: binding to localhost by default to avoid accidental LAN exposure.
    start_http_server(port, addr=addr)
    logger.info("rtservice metrics: started /metrics on http://%s:%d/metrics", addr, port)


# Prometheus metrics (best-effort; defined even if not exported)
_M_ACTIVE_STREAMS = Gauge("rtservice_active_streams", "Number of active gRPC realtime streams") if Gauge else None
_M_STREAMS_TOTAL = Counter("rtservice_streams_total", "Total number of rtservice gRPC streams") if Counter else None
_M_STREAM_ERRORS_TOTAL = Counter(
    "rtservice_stream_errors_total",
    "Number of gRPC stream errors",
    labelnames=["code"],
) if Counter else None
_M_CHUNKS_TOTAL = Counter("rtservice_chunks_total", "Audio chunks received") if Counter else None
_M_AUDIO_BYTES_TOTAL = Counter("rtservice_audio_bytes_total", "Total PCM16 bytes received") if Counter else None
_M_FEED_SECONDS = Histogram("rtservice_engine_feed_seconds", "RealtimeEngine.feed latency (seconds)") if Histogram else None
_M_SEND_EVENT_SECONDS = Histogram("rtservice_send_event_seconds", "AsrEvent send/yield latency (seconds)") if Histogram else None


def _parse_finite_float(s: object) -> Optional[float]:
    if s is None:
        return None
    try:
        x = float(str(s).strip())
        if math.isfinite(x):
            return x
        return None
    except Exception:
        return None


def _metadata_to_overrides(context: grpc.ServicerContext) -> dict:
    """Extract per-stream rtservice override knobs from gRPC metadata.

    Expected keys (lowercase on the wire):
    - x-rt-window-sec
    - x-rt-overlap-sec
    - x-rt-emit-every-sec
    - x-rt-partial-enable
    """

    md = {}
    try:
        for k, v in (context.invocation_metadata() or ()):  # type: ignore[attr-defined]
            md[str(k).lower()] = v
    except Exception:
        return {}

    win = _parse_finite_float(md.get("x-rt-window-sec"))
    ov = _parse_finite_float(md.get("x-rt-overlap-sec"))
    emit = _parse_finite_float(md.get("x-rt-emit-every-sec"))
    partial_enable_raw = md.get("x-rt-partial-enable")

    out = {}
    if win is not None:
        out["rt_window_sec"] = win
    if ov is not None:
        out["rt_overlap_sec"] = ov
    if emit is not None:
        out["rt_emit_every_sec"] = emit

    if partial_enable_raw is not None:
        raw = str(partial_enable_raw).strip().lower()
        if raw in ("1", "true", "yes", "y", "on"):
            out["rt_partial_enable"] = True
        elif raw in ("0", "false", "no", "n", "off"):
            out["rt_partial_enable"] = False
    return out


def _metadata_to_dict(context: grpc.ServicerContext) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for k, v in (context.invocation_metadata() or ()):  # type: ignore[attr-defined]
            out[str(k).lower()] = str(v)
    except Exception:
        return {}
    return out


@contextmanager
def _otel_context_from_grpc_metadata(md: dict[str, str]):
    """Attach extracted OTEL context from gRPC metadata (traceparent)."""

    if otel_context is None:
        yield
        return

    ctx = extract_trace_context_from_headers(md)
    if ctx is None:
        yield
        return

    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        try:
            otel_context.detach(token)
        except Exception:
            pass


@dataclass
class _StreamStats:
    chunks: int = 0
    audio_bytes: int = 0
    partial_events: int = 0
    final_events: int = 0
    first_session_id: str = "-"
    first_tenant_id: str = "-"
    first_lang: str = "-"

class RealtimeASRServicer(stream_pb2_grpc.RealtimeASRServicer):
    """
    Synchronous gRPC servicer for the RealtimeASR bidirectional stream.

    - Client sends AudioChunk messages (stream_pb2.AudioChunk)
    - For each chunk we feed PCM16 into RealtimeEngine
    - For every finalized segment, we yield an AsrEvent back.
    """

    def __init__(self, engine: RealtimeEngine) -> None:
        self._engine = engine
        self._log = logging.getLogger(__name__)

    def Stream(self, request_iterator, context):
        """
        NOTE: this MUST be a normal (sync) generator for grpc.server(),
        not async def and not an async generator.
        """
        md = _metadata_to_dict(context)
        overrides = _metadata_to_overrides(context)
        provider_profile_id = md.get("x-rt-provider-profile") or None
        if provider_profile_id is not None:
            overrides["provider_profile_id"] = provider_profile_id

        stats = _StreamStats()
        started_s = time.time()
        sessions: set[tuple[str, Optional[str]]] = set()

        if _M_STREAMS_TOTAL:
            _M_STREAMS_TOTAL.inc()
        if _M_ACTIVE_STREAMS:
            _M_ACTIVE_STREAMS.inc()

        peer = ""
        try:
            peer = context.peer()  # type: ignore[attr-defined]
        except Exception:
            peer = ""

        with _otel_context_from_grpc_metadata(md):
            span_cm = None
            span = None
            if trace is not None:
                try:
                    tracer = trace.get_tracer("rtservice")
                    span_cm = tracer.start_as_current_span("rtservice.grpc.stream")
                    span_cm.__enter__()
                    span = trace.get_current_span()
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("rpc.system", "grpc")
                        span.set_attribute("rpc.service", "RealtimeASR")
                        span.set_attribute("rpc.method", "Stream")
                        if peer:
                            span.set_attribute("net.peer.name", peer)
                        if overrides:
                            span.set_attribute("rtservice.overrides", ",".join(sorted(overrides.keys())))
                except Exception:
                    span_cm = None
                    span = None

            try:
                if overrides:
                    self._log.info("RealtimeASR stream overrides enabled: %s", sorted(overrides.keys()))

                for chunk in request_iterator:
                    session_id = chunk.session_id or "unknown"
                    tenant_id = getattr(chunk, "tenant_id", "") or None
                    lang = getattr(chunk, "lang", "") or None
                    sessions.add((session_id, tenant_id))

                    if stats.first_session_id == "-":
                        stats.first_session_id = session_id
                        stats.first_tenant_id = str(tenant_id or "-")
                        stats.first_lang = str(lang or "-")
                        set_session_id_for_logging(session_id)
                        self._log.info(
                            "rtservice stream started session=%s tenant=%s lang=%s peer=%s",
                            session_id,
                            stats.first_tenant_id,
                            stats.first_lang,
                            peer,
                        )

                        if span is not None and hasattr(span, "set_attribute"):
                            try:
                                span.set_attribute("nanosamurai.session_id", session_id)
                                if tenant_id:
                                    span.set_attribute("nanosamurai.tenant_id", str(tenant_id))
                                if lang:
                                    span.set_attribute("nanosamurai.lang", str(lang))
                            except Exception:
                                pass

                    stats.chunks += 1
                    b = len(chunk.pcm16_le)
                    stats.audio_bytes += b
                    if _M_CHUNKS_TOTAL:
                        _M_CHUNKS_TOTAL.inc()
                    if _M_AUDIO_BYTES_TOTAL:
                        _M_AUDIO_BYTES_TOTAL.inc(b)

                    # Feed into realtime engine
                    t0 = time.time()
                    results = self._engine.feed(
                        session_id,
                        chunk.pcm16_le,
                        lang=lang,
                        tenant_id=tenant_id,
                        sample_rate=getattr(chunk, "sample_rate", 0) or self._engine.cfg.sr,
                        **overrides,
                    )
                    dt = max(0.0, time.time() - t0)
                    if _M_FEED_SECONDS:
                        _M_FEED_SECONDS.observe(dt)

                    # Fan out events
                    for r in results:
                        ev = self._engine.to_asr_events(session_id, r)
                        if ev.type == stream_pb2.FINAL:
                            stats.final_events += 1
                        else:
                            stats.partial_events += 1

                        send0 = time.time()
                        yield ev
                        send_dt = max(0.0, time.time() - send0)
                        if _M_SEND_EVENT_SECONDS:
                            _M_SEND_EVENT_SECONDS.observe(send_dt)

            except SpeechProviderError as e:
                statuses = {
                    "invalid_request": grpc.StatusCode.INVALID_ARGUMENT,
                    "unsupported_capability": grpc.StatusCode.UNIMPLEMENTED,
                    "overloaded": grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "timeout": grpc.StatusCode.DEADLINE_EXCEEDED,
                    "unavailable": grpc.StatusCode.UNAVAILABLE,
                }
                status = statuses.get(e.code, grpc.StatusCode.INTERNAL)
                if _M_STREAM_ERRORS_TOTAL:
                    _M_STREAM_ERRORS_TOTAL.labels(code=status.name).inc()
                self._log.warning(
                    "speech provider ended realtime stream session=%s code=%s retryable=%s",
                    stats.first_session_id,
                    e.code,
                    e.retryable,
                )
                context.abort(status, e.safe_message)
            except Exception as e:
                code = "unknown"
                try:
                    code = str(getattr(getattr(context, "code", None), "name", None) or "exception")
                except Exception:
                    code = "exception"

                if _M_STREAM_ERRORS_TOTAL:
                    _M_STREAM_ERRORS_TOTAL.labels(code=code).inc()

                self._log.exception(
                    "rtservice stream failed session=%s chunks=%d partial=%d final=%d: %s",
                    stats.first_session_id,
                    stats.chunks,
                    stats.partial_events,
                    stats.final_events,
                    e,
                )
                if span is not None and Status is not None and StatusCode is not None:
                    try:
                        span.set_status(Status(StatusCode.ERROR))
                    except Exception:
                        pass
                raise
            finally:
                for session_id, tenant_id in sessions:
                    self._engine.close_session(session_id, tenant_id=tenant_id)
                dur = max(0.0, time.time() - started_s)
                self._log.info(
                    "rtservice stream ended session=%s dur=%.3fs chunks=%d audio_bytes=%d partial=%d final=%d",
                    stats.first_session_id,
                    dur,
                    stats.chunks,
                    stats.audio_bytes,
                    stats.partial_events,
                    stats.final_events,
                )
                set_session_id_for_logging("-")
                if _M_ACTIVE_STREAMS:
                    try:
                        _M_ACTIVE_STREAMS.dec()
                    except Exception:
                        pass
                if span_cm is not None:
                    try:
                        span_cm.__exit__(None, None, None)
                    except Exception:
                        pass


def create_realtime_asr_server(
    port: int = None,
    engine: Optional[RealtimeEngine] = None,
) -> grpc.Server:
    """
    Factory that builds (but does NOT start) a gRPC server.

    Your tests (and production main) can call server.start() and server.wait_for_termination().
    """
    if engine is None:
        engine = RealtimeEngine()

    if port is None:
        port = os.getenv("RT_GRPC_PORT", 50052)

    max_workers = int(os.getenv("RT_GRPC_MAX_WORKERS", "8"))
    logger.info("Creating a RealtimeASR gRPC server on port=%s max_workers=%d", port, max_workers)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    stream_pb2_grpc.add_RealtimeASRServicer_to_server(
        RealtimeASRServicer(engine),
        server,
    )

    bind_addr = os.getenv("RT_GRPC_BIND_ADDR", "[::]").strip() or "[::]"
    bound_port = server.add_insecure_port(f"{bind_addr}:{port}")
    server.bound_port = bound_port  # type: ignore[attr-defined]
    logger.info("RealtimeASR gRPC server created on port: %s", port)
    return server


def main():
    setup_logging(default_level="INFO")
    setup_otel(service_name=os.getenv("OTEL_SERVICE_NAME", "rtservice"))
    _install_crash_diagnostics()
    _install_excepthook()
    _maybe_start_metrics_server()
    server = create_realtime_asr_server()
    server.start()
    logger.info("RealtimeASR server started, waiting for termination…")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Stopping RealtimeASR server")
        server.stop(grace=5.0)


if __name__ == "__main__":
    main()
