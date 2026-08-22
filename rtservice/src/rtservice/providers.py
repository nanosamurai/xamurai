"""ASR provider profiles and the internal provider client.

The public RealtimeASR protocol remains owned by rtservice.  This module is the
internal boundary between its session/window orchestration and an ASR runtime.
Profiles are registered at process start; requests carry profile IDs only and
cannot select an arbitrary model or revision.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import numpy as np

FASTER_WHISPER_MEDIUM_PROFILE = "faster-whisper-medium-ctranslate2-r1"


@dataclass(frozen=True)
class ProviderCapabilities:
    windowed_realtime: bool
    native_streaming: bool
    batch: bool
    segment_timestamps: bool
    word_timestamps: bool
    language_detection: bool
    supported_languages: tuple[str, ...]
    stateful: bool
    preferred_sample_rate: int
    maximum_audio_seconds: float
    maximum_concurrent_sessions: int


@dataclass(frozen=True)
class ProviderProvenance:
    profile_id: str
    runtime: str
    model_revision: str
    model_digest: str
    implementation_revision: str


@dataclass(frozen=True)
class WindowRequest:
    pcm16_le: bytes
    sample_rate: int
    language: Optional[str]
    start_sample: int
    end_sample: int
    partial: bool


@dataclass(frozen=True)
class ProviderCandidate:
    text: str
    language: Optional[str]
    start_sample: int
    end_sample: int
    terminal: bool
    provider_sequence: int


class StreamingSession(Protocol):
    def push(
        self,
        pcm16_le: bytes,
        *,
        sample_rate: int,
        language: Optional[str],
        end_of_stream: bool = False,
    ) -> ProviderCandidate: ...

    def cancel(self) -> None: ...


class SpeechProvider(Protocol):
    profile_id: str
    capabilities: ProviderCapabilities
    provenance: ProviderProvenance

    def transcribe_window(self, request: WindowRequest) -> ProviderCandidate: ...

    def open_stream(self) -> StreamingSession: ...

    def close(self) -> None: ...


class SpeechProviderError(RuntimeError):
    def __init__(self, code: str, safe_message: str, *, retryable: bool = False) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable


class ProviderRegistry:
    """Immutable-in-use allowlist of configured provider profiles."""

    def __init__(self, providers: tuple[SpeechProvider, ...], *, default_profile_id: str) -> None:
        self._providers = {provider.profile_id: provider for provider in providers}
        if len(self._providers) != len(providers):
            raise ValueError("provider profile IDs must be unique")
        if default_profile_id not in self._providers:
            raise ValueError(f"default provider profile is not registered: {default_profile_id}")
        self.default_profile_id = default_profile_id

    def get(self, profile_id: Optional[str] = None) -> SpeechProvider:
        selected = profile_id or self.default_profile_id
        try:
            return self._providers[selected]
        except KeyError as exc:
            raise SpeechProviderError(
                "invalid_request",
                f"speech provider profile is not configured: {selected}",
            ) from exc

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()


def _decode_config() -> tuple[tuple[float, ...], float]:
    raw = os.getenv("RT_ASR_TEMPERATURES", "0,0.2,0.4,0.6,0.8,1")
    parts = [item.strip() for item in raw.split(",")]
    if any(not item for item in parts):
        raise ValueError("RT_ASR_TEMPERATURES must be a non-empty comma-separated list")
    try:
        temperatures = tuple(float(item) for item in parts)
    except ValueError as exc:
        raise ValueError("RT_ASR_TEMPERATURES must contain only numbers") from exc
    if not temperatures or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in temperatures):
        raise ValueError("RT_ASR_TEMPERATURES values must be finite and between 0.0 and 1.0")
    if any(current < previous for previous, current in zip(temperatures, temperatures[1:])):
        raise ValueError("RT_ASR_TEMPERATURES values must be non-decreasing")
    try:
        compression = float(os.getenv("RT_ASR_COMPRESSION_RATIO_THRESHOLD", "2.4"))
    except ValueError as exc:
        raise ValueError("RT_ASR_COMPRESSION_RATIO_THRESHOLD must be a number") from exc
    if not math.isfinite(compression) or compression <= 0.0:
        raise ValueError("RT_ASR_COMPRESSION_RATIO_THRESHOLD must be finite and greater than zero")
    return temperatures, compression


class LocalFasterWhisperProvider:
    """Pinned in-process compatibility provider for the existing default path."""

    profile_id = FASTER_WHISPER_MEDIUM_PROFILE
    model_id = "Systran/faster-whisper-medium"
    model_revision = "08e178d48790749d25932bbc082711ddcfdfbc4f"
    model_digest = "sha256:9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae"
    capabilities = ProviderCapabilities(
        windowed_realtime=True,
        native_streaming=False,
        batch=True,
        segment_timestamps=True,
        word_timestamps=True,
        language_detection=True,
        supported_languages=(),
        stateful=False,
        preferred_sample_rate=16000,
        maximum_audio_seconds=30.0,
        maximum_concurrent_sessions=0,
    )
    provenance = ProviderProvenance(
        profile_id=profile_id,
        runtime="faster-whisper==1.2.0;ctranslate2==4.6.0",
        model_revision=model_revision,
        model_digest=model_digest,
        implementation_revision="xamurai-window-provider-v1",
    )

    def __init__(self, *, model=None) -> None:
        self._model = model
        self._model_lock = threading.Lock()
        self._serialize = os.getenv("RT_ASR_SERIALIZE", "false").strip().lower() in {"1", "true", "yes", "y"}
        self._temperatures, self._compression_threshold = _decode_config()
        self._load()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                import torch
                from faster_whisper import WhisperModel

                use_cuda = torch.cuda.is_available()
                self._model = WhisperModel(
                    self.model_id,
                    revision=self.model_revision,
                    device="cuda" if use_cuda else "cpu",
                    compute_type="float16" if use_cuda else "int8_float32",
                )
        return self._model

    def transcribe_window(self, request: WindowRequest) -> ProviderCandidate:
        if request.sample_rate != self.capabilities.preferred_sample_rate:
            raise SpeechProviderError("invalid_request", "unsupported provider sample rate")
        if len(request.pcm16_le) % 2:
            raise SpeechProviderError("invalid_request", "PCM16 payload length must be even")
        if request.end_sample < request.start_sample:
            raise SpeechProviderError("invalid_request", "invalid provider sample range")
        samples = len(request.pcm16_le) // 2
        if samples > int(self.capabilities.maximum_audio_seconds * request.sample_rate):
            raise SpeechProviderError("invalid_request", "provider audio window exceeds configured maximum")
        if samples < int(0.25 * request.sample_rate):
            return ProviderCandidate("", request.language, request.start_sample, request.end_sample, not request.partial, 0)

        wave = np.frombuffer(request.pcm16_le, dtype="<i2").astype(np.float32) / 32768.0
        model = self._load()

        def decode() -> str:
            segments, _ = model.transcribe(
                wave,
                language=request.language or None,
                task="transcribe",
                beam_size=1 if request.partial else 5,
                temperature=self._temperatures,
                compression_ratio_threshold=self._compression_threshold,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=not request.partial,
            )
            words: list[str] = []
            for segment in segments:
                if getattr(segment, "words", None):
                    words.extend(
                        word.word for word in segment.words if getattr(word, "word", "").strip()
                    )
                elif getattr(segment, "text", "").strip():
                    words.append(segment.text.strip())
            return " ".join(words).strip()

        if self._serialize:
            with self._model_lock:
                text = decode()
        else:
            text = decode()
        return ProviderCandidate(
            text=text,
            language=request.language,
            start_sample=request.start_sample,
            end_sample=request.end_sample,
            terminal=not request.partial,
            provider_sequence=0,
        )

    def open_stream(self) -> StreamingSession:
        raise SpeechProviderError("unsupported_capability", "profile does not support native streaming")

    def close(self) -> None:
        return None
def registry_from_env(*, local_provider_factory: Callable[[], SpeechProvider] = LocalFasterWhisperProvider) -> ProviderRegistry:
    profile_id = os.getenv("RT_PROVIDER_PROFILE", FASTER_WHISPER_MEDIUM_PROFILE).strip()
    if profile_id == FASTER_WHISPER_MEDIUM_PROFILE:
        return ProviderRegistry((local_provider_factory(),), default_profile_id=profile_id)
    raise ValueError(f"RT_PROVIDER_PROFILE is not a registered profile: {profile_id}")
