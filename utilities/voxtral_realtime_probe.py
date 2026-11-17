"""Opt-in real-GPU concurrency probe; prints metrics, never transcript contents.

Use a consented or synthetic mono PCM16/16k WAV. The clip repeats to --seconds
to exercise epoch boundaries. This is a smoke probe, not a WER benchmark.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from uuid import uuid4
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grpc
from proto_gen import stream_pb2 as pb, stream_pb2_grpc as rpc


def load_pcm(path):
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("fixture must be PCM16 mono at 16000 Hz")
        if not 0 < source.getnframes() <= 16_000 * 600:
            raise ValueError("fixture must contain 0–600 seconds of audio")
        return source.readframes(source.getnframes())


async def run(args):
    fixtures = [load_pcm(path) for path in args.wav]
    if args.expect and len(args.expect) != len(fixtures):
        raise ValueError("provide one --expect word per fixture, or none")
    async with grpc.aio.insecure_channel(args.target) as channel:
        stub = rpc.RealtimeASRStub(channel)
        caps = await stub.GetCapabilities(pb.RealtimeCapabilitiesRequest(), timeout=5)
        if args.sessions > caps.maximum_concurrent_sessions:
            raise ValueError("requested sessions exceed advertised capacity")
        calls = []
        try:
            for _ in range(args.sessions):
                sid = f"voxtral-probe-{uuid4().hex}"
                call = stub.Stream(metadata=(("x-session-id", sid),), timeout=args.seconds * 3 + 120)
                first = await call.read()
                assert first.type == pb.SESSION_ACCEPTED
                calls.append((sid, call))
            if args.sessions == caps.maximum_concurrent_sessions:
                overflow = stub.Stream(metadata=(("x-session-id", "probe-overflow"),), timeout=5)
                try:
                    await overflow.read()
                    raise AssertionError("full replica admitted another stream")
                except grpc.aio.AioRpcError as exc:
                    assert exc.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
                    assert exc.details() == "REPLICA_FULL"

            async def session(sid, call, index):
                pcm = fixtures[index % len(fixtures)]
                expected = args.expect[index % len(fixtures)].casefold() if args.expect else None
                saw_expected = False
                started = time.monotonic()
                last_sent = started
                first_partial = None
                partials = finals = characters = 0
                final_end = 0.0

                async def send():
                    nonlocal last_sent
                    total_bytes = int(args.seconds * 16000) * 2
                    offset = 0
                    seq = 0
                    while offset < total_bytes:
                        count = min(5120, total_bytes - offset, len(pcm) - offset % len(pcm))
                        start = offset % len(pcm)
                        await call.write(pb.AudioChunk(
                            session_id=sid, tenant_id="synthetic-probe", seq=seq,
                            sample_rate=16000, pcm16_le=pcm[start:start + count], lang="en",
                        ))
                        offset += count
                        seq += 1
                        if not args.unpaced:
                            await asyncio.sleep(max(0, started + offset / 32000 - time.monotonic()))
                    last_sent = time.monotonic()
                    await call.done_writing()

                sender = asyncio.create_task(send())
                try:
                    while (event := await call.read()) is not grpc.aio.EOF:
                        assert event.session_id == sid
                        if event.type == pb.PARTIAL:
                            partials += 1
                            if first_partial is None:
                                first_partial = time.monotonic() - started
                        elif event.type == pb.FINAL:
                            assert abs(event.start_s - final_end) < 1e-6
                            final_end = event.end_s
                            finals += 1
                            characters += len(event.text)
                            if expected:
                                saw_expected |= expected in event.text.casefold()
                                for other in args.expect:
                                    if other.casefold() != expected:
                                        assert other.casefold() not in event.text.casefold(), "cross-session fixture word"
                    await sender
                    assert abs(final_end - args.seconds) < 1 / 16000
                    assert finals and characters, "speech fixture produced no final text"
                    assert expected is None or saw_expected, "expected fixture word not recognized"
                    return dict(first_partial_seconds=first_partial, partials=partials, finals=finals,
                                final_characters=characters, audio_seconds=final_end,
                                elapsed_seconds=time.monotonic() - started,
                                flush_seconds=time.monotonic() - last_sent)
                finally:
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)

            results = await asyncio.gather(*(session(sid, call, i) for i, (sid, call) in enumerate(calls)))
            # Cancel only after text proves that an actual engine request ran.
            for _ in range(args.cancel_cycles):
                sid = f"churn-{uuid4().hex}"
                call = stub.Stream(metadata=(("x-session-id", sid),), timeout=15)
                assert (await call.read()).type == pb.SESSION_ACCEPTED
                try:
                    lead = (fixtures[0] * (96000 // len(fixtures[0]) + 1))[:96000]
                    for seq, start in enumerate(range(0, len(lead), 32000)):
                        await call.write(pb.AudioChunk(session_id=sid, seq=seq, sample_rate=16000,
                                                       pcm16_le=lead[start:start + 32000], lang="en"))
                    assert (await call.read()).type == pb.PARTIAL, "cancellation fixture needs speech in its first three seconds"
                finally:
                    call.cancel()
                await asyncio.sleep(0.2)
            # Refill all the tested slots simultaneously after cancellation.
            reused = []
            try:
                for _ in range(args.sessions):
                    call = stub.Stream(metadata=(("x-session-id", f"reuse-{uuid4().hex}"),), timeout=5)
                    reused.append(call)
                    assert (await call.read()).type == pb.SESSION_ACCEPTED
                for call in reused:
                    await call.done_writing()
                    assert await call.read() is grpc.aio.EOF
            finally:
                for call in reused:
                    call.cancel()
            print(json.dumps(dict(model_revision=caps.model_revision, runtime=caps.runtime,
                                  capacity=caps.maximum_concurrent_sessions,
                                  active_cancellation_cycles=args.cancel_cycles, sessions=results), indent=2))
        finally:
            for _, call in calls:
                call.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, nargs="+")
    parser.add_argument("--expect", action="append", default=[], help="distinct expected word per fixture; repeat in fixture order")
    parser.add_argument("--target", default="127.0.0.1:50056")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=70)
    parser.add_argument("--unpaced", action="store_true")
    parser.add_argument("--cancel-cycles", type=int, default=3,
                        help="cancel after first text, then reuse capacity; first fixture needs speech at its start")
    args = parser.parse_args()
    if not 1 <= args.sessions <= 1024 or not 1 <= args.seconds <= 86400:
        parser.error("sessions must be 1–1024 and seconds must be 1–86400")
    if not 0 <= args.cancel_cycles <= 1000:
        parser.error("cancel-cycles must be 0–1000")
    asyncio.run(run(args))
