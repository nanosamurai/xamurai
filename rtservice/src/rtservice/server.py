import os
import logging
from concurrent import futures
from typing import Optional

import grpc

from proto_gen import stream_pb2
from proto_gen import stream_pb2_grpc
from rtservice.engine import RealtimeEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

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
        for chunk in request_iterator:
            session_id = chunk.session_id or "unknown"
            lang = getattr(chunk, "lang", "") or None

            self._log.debug(
                "gRPC Stream: received chunk session=%s seq=%d bytes=%d lang=%s",
                session_id,
                getattr(chunk, "seq", 0),
                len(chunk.pcm16_le),
                lang,
            )

            tenant_id = getattr(chunk, "tenant_id", "") or None

            # Feed into realtime engine
            results = self._engine.feed(
                session_id,
                chunk.pcm16_le,
                lang=lang,
                tenant_id=tenant_id,
            )

            # Fan out final events
            for r in results:
                ev = self._engine.to_asr_events(session_id, r)
                self._log.debug(
                    "gRPC Stream: sending AsrEvent session=%s [%.3f, %.3f] speaker=%s text=%s",
                    ev.session_id,
                    ev.start_s,
                    ev.end_s,
                    ev.speaker,
                    ev.text,
                )
                yield ev


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

    logger.info("Creating a RealtimeASR gRPC server on port: %s", port)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    stream_pb2_grpc.add_RealtimeASRServicer_to_server(
        RealtimeASRServicer(engine),
        server,
    )

    server.add_insecure_port(f"[::]:{port}")
    logger.info("Creating a RealtimeASR gRPC server on port: %d", port)
    return server


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
