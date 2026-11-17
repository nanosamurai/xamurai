"""Validate a stream and partition it into exact, non-overlapping PCM epochs."""

import asyncio

from .config import FRAME_BYTES, MAX_CHUNK_BYTES, SAMPLE_RATE


class InvalidAudio(ValueError):
    pass


class InputIdle(TimeoutError):
    pass


class AudioStream:
    def __init__(self, requests, session_id, languages, config):
        self.requests = requests.__aiter__()
        self.session_id = session_id
        self.languages = languages
        self.config = config
        self.identity = None
        self.next_sequence = None
        self.pending = b""
        self.total_samples = 0
        self.eof = False

    async def _read(self):
        try:
            async with asyncio.timeout(self.config.idle_timeout):
                chunk = await anext(self.requests)
        except StopAsyncIteration:
            self.eof = True
            return
        except TimeoutError as exc:
            raise InputIdle("AUDIO_IDLE_TIMEOUT") from exc
        if chunk.session_id != self.session_id:
            raise InvalidAudio("SESSION_ID_MISMATCH")
        if len(chunk.tenant_id) > 256:
            raise InvalidAudio("INVALID_TENANT_ID")
        identity = (chunk.tenant_id, chunk.lang)
        if self.identity is not None and identity != self.identity:
            raise InvalidAudio("STREAM_IDENTITY_CHANGED")
        if chunk.lang and chunk.lang not in self.languages:
            raise InvalidAudio("UNSUPPORTED_LANGUAGE")
        if self.next_sequence is not None and chunk.seq != self.next_sequence:
            raise InvalidAudio("NONCONTIGUOUS_AUDIO")
        if chunk.sample_rate != SAMPLE_RATE:
            raise InvalidAudio("PCM16_MONO_16000_REQUIRED")
        if not chunk.pcm16_le or len(chunk.pcm16_le) % 2:
            raise InvalidAudio("INVALID_PCM16_LENGTH")
        if len(chunk.pcm16_le) > MAX_CHUNK_BYTES:
            raise InvalidAudio("AUDIO_CHUNK_TOO_LARGE")
        self.identity = identity
        self.next_sequence = chunk.seq + 1
        self.pending = chunk.pcm16_le

    async def next_epoch(self):
        if not self.pending and not self.eof:
            await self._read()
        if self.eof:
            return None
        return AudioEpoch(self)


class AudioEpoch:
    def __init__(self, stream):
        self.stream = stream
        self.start_samples = stream.total_samples
        self.samples = 0
        self.exhausted = False
        self.ended_at = None

    async def frames(self):
        remaining = self.stream.config.epoch_seconds * SAMPLE_RATE * 2
        while remaining:
            if not self.stream.pending and not self.stream.eof:
                await self.stream._read()
            if self.stream.eof:
                break
            take = min(len(self.stream.pending), FRAME_BYTES, remaining)
            frame = self.stream.pending[:take]
            self.stream.pending = self.stream.pending[take:]
            self.samples += take // 2
            self.stream.total_samples += take // 2
            remaining -= take
            yield frame
        self.exhausted = True
        self.ended_at = asyncio.get_running_loop().time()
