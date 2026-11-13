"""Real localhost gRPC transport, synthetic PCM, and an injectable GPU boundary."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import grpc
import pytest

from proto_gen import stream_pb2 as pb, stream_pb2_grpc as rpc
from voxtral_rtservice.config import Config
from voxtral_rtservice.models import DEFAULT_PROFILE, verify_file
from voxtral_rtservice.server import create_server, VoxtralRealtimeServicer


class FakeBackend:
    runtime = "test"
    healthy = True

    def __init__(self):
        self.epochs = []
        self.active = 0
        self.maximum = 0
        self.closed = 0

    async def check_health(self):
        if not self.healthy:
            raise RuntimeError("unhealthy")

    async def transcribe(self, frames):
        audio = bytearray()
        self.epochs.append(audio)
        self.active += 1
        self.maximum = max(self.active, self.maximum)
        try:
            async for frame in frames:
                audio.extend(frame)
                yield str(frame[0])
        finally:
            self.active -= 1
            self.closed += 1


def chunk(sid="s", seq=1, pcm=b"\x01\x00" * 2560, **kwargs):
    return pb.AudioChunk(session_id=sid, seq=seq, sample_rate=16000, pcm16_le=pcm, **kwargs)


@asynccontextmanager
async def running(backend=None, **settings):
    backend = backend or FakeBackend()
    server, servicer = create_server(backend, Config(2, port=0, **settings))
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{server.bound_port}") as channel:
            yield backend, rpc.RealtimeASRStub(channel), servicer
    finally:
        await server.stop(None)


async def accepted(stub, sid="s"):
    call = stub.Stream(metadata=(("x-session-id", sid),), timeout=5)
    first = await call.read()
    assert first.type == pb.SESSION_ACCEPTED
    assert first.session_id == sid and first.serving_instance_id
    return call


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.01)


def test_concurrent_sessions_isolated_capacity_rejection_cancel_and_reuse():
    async def run():
        async with running() as (backend, stub, servicer):
            caps = await stub.GetCapabilities(pb.RealtimeCapabilitiesRequest())
            assert caps.maximum_concurrent_sessions == 2 and caps.batch and caps.native_streaming
            assert not caps.word_timestamps and not caps.segment_timestamps and not caps.speaker_labels
            assert not caps.language_detection and "cs" not in caps.supported_languages
            first, second = await accepted(stub, "same"), await accepted(stub, "same")
            await first.write(chunk("same", tenant_id="one"))
            await second.write(chunk("same", pcm=b"\x02\x00" * 2560, tenant_id="two"))
            assert (await first.read()).text == "1"
            assert (await second.read()).text == "2"
            assert backend.maximum == 2
            full = stub.Stream(metadata=(("x-session-id", "full"),), timeout=2)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await full.read()
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            assert error.value.details() == "REPLICA_FULL"
            first.cancel()
            await eventually(lambda: servicer.slots.active == 1)
            third = await accepted(stub, "third")
            await third.done_writing()  # empty admitted streams are valid
            assert await third.read() is grpc.aio.EOF
            await second.done_writing()
            final = await second.read()
            assert final.type == pb.FINAL and final.text == "2" and final.lang == ""
            assert await second.read() is grpc.aio.EOF
            await eventually(lambda: servicer.slots.active == 0)
            assert backend.closed == 2 and backend.active == 0
    asyncio.run(run())


def test_rejected_admission_never_reads_audio():
    class Context:
        def invocation_metadata(self):
            return (("x-session-id", "s"),)

        async def abort(self, code, detail):
            raise RuntimeError(detail)

    async def run():
        service = VoxtralRealtimeServicer(FakeBackend(), Config(1))
        assert service.slots.acquire()

        async def forbidden():
            pytest.fail("rejected RPC consumed audio")
            yield

        with pytest.raises(RuntimeError, match="REPLICA_FULL"):
            await service.Stream(forbidden(), Context())
        assert service.slots.active == 1
    asyncio.run(run())


def test_epoch_split_flushes_exact_audio_once_and_uses_absolute_intervals():
    async def run():
        async with running(epoch_seconds=10) as (backend, stub, service):
            call = await accepted(stub)
            # 10.24 s, chunk boundaries do not coincide with the epoch boundary.
            data = b"\x03\x00" * 163840

            async def send():
                for i, offset in enumerate(range(0, len(data), 30000)):
                    await call.write(chunk(seq=i, pcm=data[offset:offset + 30000]))
                await call.done_writing()

            sender = asyncio.create_task(send())
            events = []
            while (event := await call.read()) is not grpc.aio.EOF:
                events.append(event)
            await sender
            finals = [e for e in events if e.type == pb.FINAL]
            assert [(e.start_s, e.end_s) for e in finals] == [(0, 10), (10, 10.24)]
            assert b"".join(backend.epochs) == data
            assert len(backend.epochs[0]) == 320000
            assert service.slots.active == 0
    asyncio.run(run())


@pytest.mark.parametrize("bad,code", [
    (pb.AudioChunk(session_id="s", sample_rate=8000, pcm16_le=b"00"), "PCM16_MONO"),
    (chunk(pcm=b"x"), "INVALID_PCM16"), (chunk(pcm=b""), "INVALID_PCM16"),
    (chunk(pcm=b"00" * 16001), "AUDIO_CHUNK_TOO_LARGE"),
    (chunk("different"), "SESSION_ID_MISMATCH"), (chunk(lang="cs"), "UNSUPPORTED_LANGUAGE"),
])
def test_invalid_audio_releases_capacity(bad, code):
    async def run():
        async with running() as (_, stub, service):
            call = await accepted(stub)
            await call.write(bad)
            await call.done_writing()
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call.read()
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert code in error.value.details()
            await eventually(lambda: service.slots.active == 0)
    asyncio.run(run())


@pytest.mark.parametrize("second", [chunk(seq=3), chunk(seq=2, tenant_id="changed"), chunk(seq=2, lang="en")])
def test_sequence_tenant_and_language_cannot_change(second):
    async def run():
        async with running() as (_, stub, service):
            call = await accepted(stub)
            await call.write(chunk())
            assert (await call.read()).type == pb.PARTIAL
            await call.write(second)
            await call.done_writing()
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call.read()
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            await eventually(lambda: service.slots.active == 0)
    asyncio.run(run())


def test_idle_and_failed_backend_release_slots_without_a_fake_final():
    class Broken(FakeBackend):
        async def transcribe(self, frames):
            async for _ in frames:
                raise RuntimeError("do not expose private transcript")
                yield

    async def run():
        async with running(idle_timeout=0.05) as (_, stub, service):
            call = await accepted(stub)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call.read()
            assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            assert service.slots.active == 0
        async with running(Broken()) as (_, stub, service):
            call = await accepted(stub)
            await call.write(chunk())
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call.read()
            assert error.value.details() == "VOXTRAL_INFERENCE_FAILED"
            assert service.slots.active == 0
    asyncio.run(run())


def test_premature_engine_completion_fails_instead_of_truncating_audio():
    class Premature(FakeBackend):
        async def transcribe(self, frames):
            yield "incomplete"

    async def run():
        async with running(Premature()) as (_, stub, service):
            call = await accepted(stub)
            await call.write(chunk())
            assert (await call.read()).type == pb.PARTIAL
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call.read()
            assert error.value.code() == grpc.StatusCode.INTERNAL
            assert service.slots.active == 0
    asyncio.run(run())


def test_unhealthy_backend_fails_readiness_and_new_admission():
    async def run():
        backend = FakeBackend()
        backend.healthy = False
        async with running(backend) as (_, stub, service):
            for request in (stub.GetCapabilities(pb.RealtimeCapabilitiesRequest()),
                            stub.Stream(metadata=(("x-session-id", "s"),)).read()):
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await request
                assert error.value.code() == grpc.StatusCode.UNAVAILABLE
            assert service.slots.active == 0
    asyncio.run(run())


def test_configuration_and_verified_profile(monkeypatch, tmp_path):
    monkeypatch.delenv("RT_SERVING_MAX_SESSIONS", raising=False)
    with pytest.raises(ValueError, match="RT_SERVING_MAX_SESSIONS"):
        Config.from_env()
    monkeypatch.setenv("RT_SERVING_MAX_SESSIONS", "8")
    assert Config.from_env().maximum_sessions == 8
    for changes in ({"maximum_sessions": 0}, {"idle_timeout": float("nan")},
                    {"epoch_seconds": 600}, {"audio_queue_frames": 0},
                    {"gpu_memory_utilization": float("inf")}):
        with pytest.raises(ValueError):
            replace(Config(2), **changes)
    for changes in ({"revision": "main"}, {"files": {"../weights": "0" * 64}}):
        with pytest.raises(ValueError):
            replace(DEFAULT_PROFILE, **changes)
    artifact = tmp_path / "weights"
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_file(artifact, "0" * 64)


def test_slow_consumer_deadline_releases_capacity():
    class Context:
        def invocation_metadata(self):
            return (("x-session-id", "s"),)

        async def write(self, event):
            await asyncio.Event().wait()

        async def abort(self, code, detail):
            assert code == grpc.StatusCode.DEADLINE_EXCEEDED
            raise RuntimeError(detail)

    async def run():
        service = VoxtralRealtimeServicer(FakeBackend(), Config(2, output_timeout=0.05))

        async def unread():
            pytest.fail("audio must wait for acceptance")
            yield

        with pytest.raises(RuntimeError, match="VOXTRAL_STREAM_TIMEOUT"):
            await service.Stream(unread(), Context())
        assert service.slots.active == 0
    asyncio.run(run())


def test_text_overflow_aborts_without_final():
    class Verbose(FakeBackend):
        async def transcribe(self, frames):
            yield "x" * 1025

    async def run():
        async with running(Verbose(), max_epoch_text_bytes=1024) as (_, stub, service):
            call = await accepted(stub)
            await call.write(chunk())
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call.read()
            assert error.value.details() == "VOXTRAL_INFERENCE_FAILED"
            assert service.slots.active == 0
    asyncio.run(run())
