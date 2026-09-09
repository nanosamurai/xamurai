from __future__ import annotations

import argparse
import json
import queue
import time
import wave

import grpc

from proto_gen import stream_pb2, stream_pb2_grpc


class _Requests:
    _END = object()

    def __init__(self) -> None:
        self._queue: queue.Queue[object] = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._END:
            raise StopIteration
        return item

    def send(self, chunk) -> None:
        self._queue.put(chunk)

    def finish(self) -> None:
        self._queue.put(self._END)


def _channel(target: str):
    service_config = json.dumps(
        {"loadBalancingConfig": [{"round_robin": {}}]}, separators=(",", ":")
    )
    return grpc.insecure_channel(
        target,
        options=(("grpc.service_config", service_config),),
    )


def _open(stub, session_id: str, timeout: int):
    requests = _Requests()
    responses = stub.Stream(
        requests,
        timeout=timeout,
        metadata=(("x-session-id", session_id),),
    )
    try:
        accepted = next(responses)
    except Exception:
        requests.finish()
        responses.cancel()
        raise
    if accepted.type != stream_pb2.SESSION_ACCEPTED or not accepted.serving_instance_id:
        requests.finish()
        responses.cancel()
        raise RuntimeError("provider did not complete the admission handshake")
    return requests, responses, accepted.serving_instance_id


def _open_with_admission_retry(stub, session_id: str, timeout: int):
    deadline = time.monotonic() + timeout
    while True:
        remaining = max(1, int(deadline - time.monotonic()))
        try:
            return _open(stub, session_id, remaining)
        except grpc.RpcError as exc:
            if (
                exc.code() != grpc.StatusCode.RESOURCE_EXHAUSTED
                or time.monotonic() >= deadline
            ):
                raise
            # A round-robin DNS channel can make its first calls before every
            # resolved subchannel reaches READY. Retry admission so it can
            # select another replica while existing streams remain held.
            time.sleep(0.1)


def verify_replicas(stub, count: int, timeout: int) -> None:
    held = []
    try:
        for index in range(count):
            held.append(
                _open_with_admission_retry(
                    stub, f"nemotron-probe-{index + 1}", timeout
                )
            )
        instance_ids = {instance_id for _, _, instance_id in held}
        if len(instance_ids) != count:
            raise RuntimeError(
                f"expected {count} distinct replicas but reached {len(instance_ids)}"
            )
        print(f"replica_check=ok distinct_instances={len(instance_ids)}")
    finally:
        for requests, responses, _ in held:
            requests.finish()
            responses.cancel()


def transcribe_wav(stub, path: str, timeout: int) -> None:
    with wave.open(path, "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != 16_000
        ):
            raise ValueError("probe WAV must be mono PCM16 at 16000 Hz")
        audio = source.readframes(source.getnframes())

    requests, responses, _ = _open_with_admission_retry(
        stub, "nemotron-audio-probe", timeout
    )
    chunk_bytes = 3_200 * 2
    for sequence, offset in enumerate(range(0, len(audio), chunk_bytes), start=1):
        requests.send(
            stream_pb2.AudioChunk(
                session_id="nemotron-audio-probe",
                seq=sequence,
                sample_rate=16_000,
                pcm16_le=audio[offset : offset + chunk_bytes],
                lang="cs",
            )
        )
    requests.finish()
    event_count = 0
    final_count = 0
    nonempty_final = False
    for event in responses:
        event_count += 1
        if event.type == stream_pb2.FINAL:
            final_count += 1
            nonempty_final = nonempty_final or bool(event.text.strip())
    if not nonempty_final:
        raise RuntimeError("Nemotron did not return a non-empty final transcript")
    print(f"audio_check=ok events={event_count} finals={final_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Nemotron Compose deployment")
    parser.add_argument("--target", default="dns:///nemotron-rtservice:50052")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--wav")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.replicas <= 16:
        parser.error("--replicas must be between 1 and 16")

    with _channel(args.target) as channel:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        verify_replicas(stub, args.replicas, args.timeout)
        if args.wav:
            transcribe_wav(stub, args.wav, args.timeout)


if __name__ == "__main__":
    main()
