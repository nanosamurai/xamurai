"""One shared vLLM engine, independent bounded native buffers per epoch.

This integration intentionally pins vLLM: VoxtralRealtimeBuffer and rendering
are internal APIs. No upstream WebSocket frontend or unbounded audio feeder is
started. Keep the runtime contract tests when upgrading the pinned image.
"""

import asyncio
from contextlib import aclosing
from importlib.metadata import version
import logging
from uuid import uuid4

import numpy as np

from .config import FRAME_SAMPLES, SAMPLE_RATE, VLLM_VERSION

logger = logging.getLogger(__name__)


class BackendFailure(RuntimeError):
    pass


def engine_options(config, model_path):
    return dict(
        model=model_path, tokenizer=model_path, tokenizer_mode="mistral",
        config_format="mistral", load_format="mistral", dtype="bfloat16",
        trust_remote_code=False, max_model_len=config.max_model_len,
        max_num_seqs=config.maximum_sessions,
        max_num_batched_tokens=config.max_batched_tokens,
        kv_cache_memory_bytes=config.kv_cache_mib * 1024**2,
        tensor_parallel_size=config.tensor_parallel_size,
        limit_mm_per_prompt={"audio": 1}, mm_processor_cache_gb=1,
        enable_prefix_caching=False, enable_log_requests=False,
        compilation_config={"cudagraph_mode": "PIECEWISE"},
    )


def _buffer_factory(model_config, queue_frames):
    from vllm.model_executor.models.voxtral_realtime import VoxtralRealtimeBuffer
    from vllm.tokenizers import cached_tokenizer_from_config

    tokenizer = cached_tokenizer_from_config(model_config)
    encoder = tokenizer.instruct.audio_encoder
    audio_config = encoder.audio_config
    if audio_config.sampling_rate != SAMPLE_RATE or audio_config.frame_rate != 12.5:
        raise ValueError("unsupported Voxtral streaming audio geometry")
    prefix = tokenizer.instruct.start() + encoder.encode_streaming_tokens()
    left, right = encoder.get_padding_audio()
    if len(prefix) > 256:
        raise ValueError("streaming prefix exceeds reserved context budget")
    buffer = VoxtralRealtimeBuffer(audio_config, prefix, max_model_len=model_config.max_model_len)
    # This is the only private state replacement. It is request-local, happens
    # before any tasks start, and gives upstream append_audio real backpressure.
    if not buffer._audio_queue.empty():
        raise ValueError("vLLM audio buffer contract changed")
    buffer._audio_queue = asyncio.Queue(maxsize=queue_frames)
    return buffer, left.audio_array, right.audio_array


class VllmBackend:
    runtime = f"vllm=={VLLM_VERSION};native-streaming;bf16"

    def __init__(self, engine, config, *, buffer_factory=_buffer_factory, render=None, sampling=None):
        self.engine = engine
        self.config = config
        self._buffer_factory = buffer_factory
        self._render = render
        self._sampling = sampling
        self.fatal = asyncio.Event()

    @classmethod
    def load(cls, config, model_path):
        if version("vllm") != VLLM_VERSION:
            raise RuntimeError(f"Voxtral requires vllm=={VLLM_VERSION}")
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_options(config, model_path)))
        try:
            # Validate the tokenizer/buffer API before admitting any RPC.
            _buffer_factory(engine.model_config, config.audio_queue_frames)
            return cls(engine, config)
        except BaseException:
            engine.shutdown()
            raise

    @property
    def healthy(self):
        return not self.fatal.is_set()

    async def check_health(self):
        if not self.healthy:
            raise BackendFailure("ENGINE_UNHEALTHY")
        try:
            async with asyncio.timeout(self.config.cleanup_timeout):
                await self.engine.check_health()
        except Exception as exc:
            self.fatal.set()
            raise BackendFailure("ENGINE_UNHEALTHY") from exc

    async def transcribe(self, frames):
        request_id = f"voxtral-{uuid4().hex}"
        buffer, left_pad, right_pad = self._buffer_factory(
            self.engine.model_config, self.config.audio_queue_frames,
        )
        if self._sampling is None:
            from vllm.sampling_params import RequestOutputKind, SamplingParams
            sampling = SamplingParams(temperature=0.0, max_tokens=1,
                                      output_kind=RequestOutputKind.DELTA, skip_clone=True)
        else:
            sampling = self._sampling

        async def put_audio(audio):
            # Padding is split too: queue capacity bounds bytes, not just objects.
            for offset in range(0, len(audio), FRAME_SAMPLES):
                await buffer.append_audio(audio[offset:offset + FRAME_SAMPLES].copy())

        async def feed_audio():
            await put_audio(left_pad)
            async with aclosing(frames):
                async for pcm in frames:
                    await put_audio(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
            await put_audio(right_pad)
            await buffer.append_audio(None)

        async def prompts():
            async with aclosing(buffer.get_input_stream()) as input_stream:
                async for prompt in input_stream:
                    if self._render is not None:
                        yield await self._render(prompt)
                    else:
                        from vllm.engine.protocol import StreamingInput
                        from vllm.renderers.inputs.preprocess import parse_model_prompt
                        parsed = parse_model_prompt(self.engine.model_config, prompt)
                        (engine_input,) = await self.engine.renderer.render_cmpl_async([parsed])
                        yield StreamingInput(prompt=engine_input)
            # A model context cutoff also closes the iterator. Never silently
            # report it as a successfully transcribed epoch while audio remains.
            if not feeder.done() or not buffer._audio_queue.empty():
                raise BackendFailure("ENGINE_AUDIO_TRUNCATED")
            feeder.result()

        feeder = asyncio.create_task(feed_audio())
        prompt_stream = prompts()
        outputs = self.engine.generate(prompt=prompt_stream, sampling_params=sampling, request_id=request_id)
        pending = None
        try:
            while True:
                pending = asyncio.create_task(anext(outputs))
                async with asyncio.timeout(self.config.progress_timeout):
                    # A bad chunk or an idle sender must interrupt a decoder
                    # that is waiting for input, not wait for its next token.
                    if not feeder.done():
                        await asyncio.wait((pending, feeder), return_when=asyncio.FIRST_COMPLETED)
                    if feeder.done():
                        feeder.result()
                    try:
                        output = await pending
                    except StopAsyncIteration:
                        break
                for completion in output.outputs:
                    if completion.token_ids:
                        await buffer.append_tokens(completion.token_ids[-1:])
                    if completion.text:
                        yield completion.text
            await feeder
        finally:
            # Abort the actual engine request, including when paused waiting for
            # audio. Cancelling Python tasks alone leaks vLLM scheduler slots.
            for task in (pending, feeder):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (pending, feeder) if task is not None),
                                 return_exceptions=True)
            try:
                async with asyncio.timeout(self.config.cleanup_timeout):
                    await self.engine.abort(request_id)
                    await outputs.aclose()
                    await prompt_stream.aclose()
            except BaseException:
                self.fatal.set()
                logger.error("Voxtral engine cleanup failed; retiring replica")
                raise

    def close(self):
        self.engine.shutdown(timeout=self.config.cleanup_timeout)
