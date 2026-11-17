"""Exercise the production async driver without importing CUDA or vLLM."""

import asyncio
from contextlib import aclosing
from types import SimpleNamespace

import numpy as np
import pytest

from voxtral_rtservice.backend import VllmBackend, engine_options
from voxtral_rtservice.config import Config, FRAME_BYTES


class Buffer:
    def __init__(self, size):
        self._audio_queue = asyncio.Queue(maxsize=size)

    async def append_audio(self, value):
        await self._audio_queue.put(value)

    async def append_tokens(self, value):
        pass

    async def get_input_stream(self):
        while (value := await self._audio_queue.get()) is not None:
            yield value


class Engine:
    model_config = None

    def __init__(self):
        self.aborted = []
        self.gate = None
        self.abort_error = False

    async def generate(self, *, prompt, sampling_params, request_id):
        async for frame in prompt:
            if self.gate is not None:
                await self.gate.wait()
            yield SimpleNamespace(outputs=[SimpleNamespace(text="x", token_ids=[1])])

    async def abort(self, request_id):
        self.aborted.append(request_id)
        if self.abort_error:
            raise RuntimeError("abort failed")


async def render(value):
    return value


def backend(engine, **settings):
    return VllmBackend(engine, Config(4, **settings), render=render, sampling=object(),
                       buffer_factory=lambda _, size: (Buffer(size), np.zeros(0), np.zeros(0)))


def test_shared_engine_receives_capacity_and_memory_budget():
    options = engine_options(Config(12, kv_cache_mib=4096), "/verified/model")
    assert options["max_num_seqs"] == 12
    assert options["kv_cache_memory_bytes"] == 4096 * 1024**2
    assert options["enable_prefix_caching"] is False
    assert options["trust_remote_code"] is False


def test_audio_queue_applies_backpressure_and_cancel_aborts_engine():
    async def run():
        engine = Engine()
        engine.gate = asyncio.Event()
        runtime = backend(engine, audio_queue_frames=2)
        produced = 0

        async def source():
            nonlocal produced
            for _ in range(1000):
                produced += 1
                yield b"\x00" * FRAME_BYTES

        async def consume():
            async with aclosing(runtime.transcribe(source())) as result:
                async for _ in result:
                    pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        # Two queued, one being decoded, one producer blocked in put().
        assert produced == 4
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(engine.aborted) == 1 and runtime.healthy
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    asyncio.run(run())


def test_input_error_interrupts_waiting_decoder_and_propagates_original_error():
    async def run():
        engine = Engine()
        runtime = backend(engine)

        async def source():
            raise ValueError("invalid input")
            yield

        with pytest.raises(ValueError, match="invalid input"):
            async for _ in runtime.transcribe(source()):
                pass
        assert len(engine.aborted) == 1
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    asyncio.run(run())


def test_completion_and_progress_timeout_both_abort_and_cleanup_failure_retires():
    async def source():
        yield b"\x00" * FRAME_BYTES

    async def run():
        engine = Engine()
        runtime = backend(engine)
        assert [v async for v in runtime.transcribe(source())] == ["x"]
        assert len(engine.aborted) == 1
        engine.gate = asyncio.Event()
        runtime = backend(engine, progress_timeout=0.05)
        with pytest.raises(TimeoutError):
            async for _ in runtime.transcribe(source()):
                pass
        assert len(engine.aborted) == 2
        engine.abort_error = True
        engine.gate.set()
        with pytest.raises(RuntimeError, match="abort failed"):
            async for _ in runtime.transcribe(source()):
                pass
        assert not runtime.healthy and runtime.fatal.is_set()
    asyncio.run(run())


def test_engine_ending_before_consuming_input_fails_without_waiting_forever():
    class Premature(Engine):
        async def generate(self, **kwargs):
            return
            yield

    async def run():
        engine = Premature()
        runtime = backend(engine, audio_queue_frames=1)

        async def source():
            for _ in range(100):
                yield b"\x00" * FRAME_BYTES

        async with asyncio.timeout(1):
            with pytest.raises(RuntimeError, match="ENGINE_AUDIO_TRUNCATED"):
                async for _ in runtime.transcribe(source()):
                    pass
        assert len(engine.aborted) == 1
    asyncio.run(run())


def test_cleanup_deadline_covers_a_decoder_slow_to_cancel():
    class SlowCancellation(Engine):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def generate(self, **kwargs):
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await self.release.wait()
                raise
            yield

    async def run():
        engine = SlowCancellation()
        runtime = backend(engine, cleanup_timeout=0.05)

        async def source():
            yield b"\x00" * FRAME_BYTES

        task = asyncio.create_task(anext(runtime.transcribe(source())))
        await engine.entered.wait()
        task.cancel()
        with pytest.raises(TimeoutError, match="helper did not stop"):
            await asyncio.wait_for(task, timeout=1)
        assert not runtime.healthy
        # Simulate the supervisor terminating the failed engine, releasing the
        # non-cooperative operation; no background task may survive this test.
        engine.release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    asyncio.run(run())
