import grpc

from proto_gen import speech_provider_pb2, speech_provider_pb2_grpc


def main() -> None:
    with grpc.insecure_channel("127.0.0.1:50061") as channel:
        response = speech_provider_pb2_grpc.SpeechProviderStub(channel).Health(
            speech_provider_pb2.HealthRequest(), timeout=2
        )
    if not response.ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
