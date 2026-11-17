"""Async RealtimeASR peer with replica-local admission and bounded epochs."""

import asyncio
from contextlib import aclosing
import logging
import os
import signal

import grpc

from proto_gen import stream_pb2, stream_pb2_grpc
from xamurai_serving.admission import SessionSlots, serving_instance_id
from .audio import AudioStream, InputIdle, InvalidAudio
from .backend import BackendFailure, VllmBackend
from .config import Config, SAMPLE_RATE
from .models import DEFAULT_PROFILE, load_profile, resolve_model
from .metrics import Metrics

logger = logging.getLogger(__name__)


class VoxtralRealtimeServicer(stream_pb2_grpc.RealtimeASRServicer):
    def __init__(self, backend, config, profile=DEFAULT_PROFILE, metrics=None):
        self.backend = backend
        self.config = config
        self.profile = profile
        self.slots = SessionSlots(config.maximum_sessions)
        self.instance_id = serving_instance_id()
        self.metrics = metrics or Metrics()
        self.metrics.capacity.set(config.maximum_sessions)

    async def GetCapabilities(self, request, context):
        try:
            await self.backend.check_health()
        except Exception:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "ENGINE_UNHEALTHY")
        return stream_pb2.RealtimeCapabilities(
            provider_profile_id=self.profile.profile_id,
            native_streaming=True, windowed_realtime=False, batch=True,
            segment_timestamps=False, word_timestamps=False, speaker_labels=False,
            language_detection=False, supported_languages=self.profile.languages,
            stateful=True, preferred_sample_rate=SAMPLE_RATE, maximum_audio_seconds=0,
            maximum_concurrent_sessions=self.slots.maximum, runtime=self.backend.runtime,
            model_revision=self.profile.revision, model_digest=self.profile.digest,
            implementation_revision="xamurai-voxtral-realtime-v1", session_settings_json="{}",
        )

    async def Stream(self, request_iterator, context):
        metadata = dict(context.invocation_metadata() or ())
        session_id = metadata.get("x-session-id", "")
        if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 256:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "SESSION_ID_REQUIRED")
        if not self.backend.healthy:
            self.metrics.admissions.labels("unhealthy").inc()
            await context.abort(grpc.StatusCode.UNAVAILABLE, "ENGINE_UNHEALTHY")
        if not self.slots.acquire():
            self.metrics.admissions.labels("full").inc()
            logger.info("Voxtral admission rejected active=%d limit=%d", self.slots.active, self.slots.maximum)
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "REPLICA_FULL")
        self.metrics.admissions.labels("accepted").inc()
        self.metrics.active.set(self.slots.active)
        outcome = "cancelled"
        try:
            await self._write(context, stream_pb2.AsrEvent(
                session_id=session_id, type=stream_pb2.SESSION_ACCEPTED,
                provider_profile_id=self.profile.profile_id, serving_instance_id=self.instance_id,
            ))
            stream = AudioStream(request_iterator, session_id, self.profile.languages, self.config)
            while (epoch := await stream.next_epoch()) is not None:
                text = ""
                text_bytes = 0
                epoch_started = asyncio.get_running_loop().time()
                async with aclosing(self.backend.transcribe(epoch.frames())) as updates:
                    async for delta in updates:
                        if not text:
                            self.metrics.first_partial.observe(asyncio.get_running_loop().time() - epoch_started)
                        text_bytes += len(delta.encode("utf-8"))
                        if text_bytes > self.config.max_epoch_text_bytes:
                            raise BackendFailure("EPOCH_TEXT_LIMIT")
                        text += delta
                        await self._write(context, self._event(session_id, epoch, text, final=False))
                if not epoch.exhausted:
                    raise BackendFailure("ENGINE_AUDIO_TRUNCATED")
                self.metrics.flush.observe(asyncio.get_running_loop().time() - epoch.ended_at)
                await self._write(context, self._event(session_id, epoch, text, final=True))
            outcome = "completed"
        except InvalidAudio as exc:
            outcome = "invalid_audio"
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except (InputIdle, TimeoutError):
            outcome = "timeout"
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "VOXTRAL_STREAM_TIMEOUT")
        except Exception as exc:
            outcome = "failed"
            # Exception messages may contain prompts or private runtime paths.
            logger.error("Voxtral stream failed error_type=%s", type(exc).__name__)
            await context.abort(grpc.StatusCode.INTERNAL, "VOXTRAL_INFERENCE_FAILED")
        finally:
            self.slots.release()
            self.metrics.active.set(self.slots.active)
            self.metrics.closed.labels(outcome).inc()
            logger.info("Voxtral stream closed outcome=%s active=%d limit=%d",
                        outcome, self.slots.active, self.slots.maximum)

    def _event(self, session_id, epoch, text, *, final):
        return stream_pb2.AsrEvent(
            session_id=session_id, start_s=epoch.start_samples / SAMPLE_RATE,
            end_s=(epoch.start_samples + epoch.samples) / SAMPLE_RATE,
            text=text.strip(), type=stream_pb2.FINAL if final else stream_pb2.PARTIAL,
            # Input lang is only a validation hint; Voxtral does not report LID.
            lang="", speaker="", provider_profile_id=self.profile.profile_id,
        )

    async def _write(self, context, event):
        async with asyncio.timeout(self.config.output_timeout):
            await context.write(event)


def create_server(backend, config, profile=DEFAULT_PROFILE, metrics=None):
    server = grpc.aio.server(options=(
        ("grpc.max_receive_message_length", 65_536),
        ("grpc.max_send_message_length", config.max_epoch_text_bytes + 4096),
        ("grpc.max_metadata_size", 8192),
    ))
    servicer = VoxtralRealtimeServicer(backend, config, profile, metrics)
    stream_pb2_grpc.add_RealtimeASRServicer_to_server(servicer, server)
    server.bound_port = server.add_insecure_port(f"{config.bind_address}:{config.port}")
    if not server.bound_port:
        raise RuntimeError("could not bind Voxtral gRPC listener")
    return server, servicer


async def serve(config, profile):
    # Download/hash checking runs before readiness; never during a live RPC.
    model_path = await asyncio.to_thread(resolve_model, profile)
    backend = VllmBackend.load(config, model_path)
    server = None
    watchers = []
    metrics_server = None
    try:
        from prometheus_client import REGISTRY, start_http_server
        metrics = Metrics(REGISTRY)
        server, _ = create_server(backend, config, profile, metrics)
        stopped = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stopped.set)
        await backend.check_health()
        metrics_server, _ = start_http_server(config.metrics_port, addr=config.metrics_bind_address)
        await server.start()
        logger.info("Voxtral ready profile=%s capacity=%d epoch_seconds=%d",
                    profile.profile_id, config.maximum_sessions, config.epoch_seconds)
        watchers = [asyncio.create_task(stopped.wait()), asyncio.create_task(backend.fatal.wait())]
        await asyncio.wait(watchers, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in watchers:
            task.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)
        if server is not None:
            await server.stop(grace=5)
        if metrics_server is not None:
            await asyncio.to_thread(metrics_server.shutdown)
            metrics_server.server_close()
        backend.close()
    if backend.fatal.is_set():
        raise RuntimeError("Voxtral engine retired after failed cleanup or health check")


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(serve(Config.from_env(), load_profile()))


if __name__ == "__main__":
    main()
