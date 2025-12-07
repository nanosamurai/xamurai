import os
import asyncio
import logging

import grpc

from proto import stream_pb2
from proto import stream_pb2_grpc

# Reuse your existing engine implementation
from engine import RealtimeEngine, AsrResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rtservice")


class RealtimeAsrServicer(stream_pb2_grpc.RealtimeASRServicer):
    """
    gRPC wrapper around RealtimeEngine.

    - Accepts a stream of AudioChunk.
    - For each chunk, runs engine.feed(session_id, pcm16, lang).
    - Streams back AsrEvent for any finalized segments.
    """

    def __init__(self) -> None:
        super().__init__()
        self._engine = RealtimeEngine()

    async def Stream(self, request_iterator, context):
        """
        Bi-di streaming RPC:
          client -> stream AudioChunk
          server -> stream AsrEvent
        """
        async for chunk in request_iterator:
            session_id = chunk.session_id or "unknown"
            lang = chunk.lang or None

            # Reuse the engine’s feed() just like your old in-process BFF
            results = self._engine.feed(
                session_id=session_id,
                pcm16=bytes(chunk.pcm16_le),
                lang=lang,
            )

            for r in results:
                # r is AsrResult(start_s, end_s, text, is_final, lang, speaker)
                ev = stream_pb2.AsrEvent(
                    session_id=session_id,
                    start_s=r.start_s,
                    end_s=r.end_s,
                    text=r.text,
                    type=stream_pb2.FINAL if r.is_final else stream_pb2.PARTIAL,
                    lang=(r.lang or ""),
                    speaker=(r.speaker or ""),
                )
                yield ev


async def serve() -> None:
    server = grpc.aio.server()
    servicer = RealtimeAsrServicer()
    stream_pb2_grpc.add_RealtimeASRServicer_to_server(servicer, server)

    bind_addr = os.getenv("RT_GRPC_BIND", "[::]:50051")
    server.add_insecure_port(bind_addr)

    logger.info("Starting RealtimeASR gRPC server on %s", bind_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
