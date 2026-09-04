"""Dependency-light RealtimeASR server for replica-routing verification."""

from concurrent import futures

import grpc

from proto_gen import stream_pb2, stream_pb2_grpc
from xamurai_serving import SessionSlots, max_sessions_from_env, serving_instance_id


class RoutingProofServicer(stream_pb2_grpc.RealtimeASRServicer):
    """Implement only admission and stream lifetime for routing tests."""

    def __init__(self) -> None:
        self._slots = SessionSlots(max_sessions_from_env())
        self._instance_id = serving_instance_id()

    def GetCapabilities(self, request, context):
        return stream_pb2.RealtimeCapabilities(
            provider_profile_id="routing-proof",
            native_streaming=True,
            stateful=True,
            preferred_sample_rate=16000,
            maximum_concurrent_sessions=self._slots.maximum,
            runtime="routing-proof",
        )

    def Stream(self, request_iterator, context):
        metadata = {
            str(key).lower(): str(value)
            for key, value in (context.invocation_metadata() or ())
        }
        session_id = metadata.get("x-session-id", "").strip()
        if not session_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "SESSION_ID_REQUIRED")
        if not self._slots.acquire():
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "REPLICA_FULL")
        try:
            yield stream_pb2.AsrEvent(
                session_id=session_id,
                type=stream_pb2.SESSION_ACCEPTED,
                provider_profile_id="routing-proof",
                serving_instance_id=self._instance_id,
            )
            for chunk in request_iterator:
                if chunk.session_id != session_id:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "SESSION_ID_MISMATCH",
                    )
        finally:
            self._slots.release()


def main() -> None:
    """Serve the routing protocol on the internal test port."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    stream_pb2_grpc.add_RealtimeASRServicer_to_server(RoutingProofServicer(), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
