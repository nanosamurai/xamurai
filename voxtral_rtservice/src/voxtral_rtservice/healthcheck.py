"""Readiness includes the shared engine's health, never transcript data."""

import os
import grpc
from proto_gen import stream_pb2, stream_pb2_grpc


def main():
    port = int(os.getenv("VOXTRAL_RTSERVICE_PORT", "50052"))
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        result = stream_pb2_grpc.RealtimeASRStub(channel).GetCapabilities(
            stream_pb2.RealtimeCapabilitiesRequest(), timeout=2,
        )
    if not result.provider_profile_id:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
