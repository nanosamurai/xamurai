import os

import grpc

from proto_gen import stream_pb2, stream_pb2_grpc


def main() -> None:
    port = int(os.getenv("NEMOTRON_RTSERVICE_PORT", "50052"))
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        response = stream_pb2_grpc.RealtimeASRStub(channel).GetCapabilities(
            stream_pb2.RealtimeCapabilitiesRequest(), timeout=2
        )
    if not response.provider_profile_id:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
