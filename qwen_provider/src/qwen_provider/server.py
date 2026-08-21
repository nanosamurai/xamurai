from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import os
import threading
from concurrent import futures
from pathlib import Path
from typing import Optional, Protocol

import grpc
import numpy as np

from proto_gen import speech_provider_pb2, speech_provider_pb2_grpc


logger = logging.getLogger(__name__)

PROFILE_ID = "qwen3-asr-0.6b-vllm-r1"
MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
MODEL_REVISION = "c4468bdb552ddc559e464f6081e22dd4034f2e68"
MODEL_DIGEST = "sha256:79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
SAMPLE_RATE = 16000


class QwenBackend(Protocol):
    runtime: str
    supported_languages: tuple[str, ...]

    def open(self, language: Optional[str]): ...

    def push(self, pcm16: np.ndarray, state): ...

    def finish(self, state): ...


class QwenVllmBackend:
    """Exact-revision Qwen3-ASR runtime using its native vLLM streaming API."""

    def __init__(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from qwen_asr import Qwen3ASRModel

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen vLLM provider requires an NVIDIA CUDA device")

        model_path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
        weights = Path(model_path) / "model.safetensors"
        if not weights.is_file():
            raise RuntimeError("pinned Qwen model artifact is incomplete")
        digest = hashlib.sha256()
        with weights.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if f"sha256:{digest.hexdigest()}" != MODEL_DIGEST:
            raise RuntimeError("pinned Qwen model artifact digest does not match the provider profile")

        gpu_memory = _bounded_float("QWEN_GPU_MEMORY_UTILIZATION", 0.65, 0.1, 0.95)
        max_new_tokens = _bounded_int("QWEN_MAX_NEW_TOKENS", 256, 16, 1024)
        self._chunk_size_sec = _bounded_float("QWEN_STREAM_CHUNK_SECONDS", 2.0, 0.5, 10.0)
        self._model = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory,
            max_new_tokens=max_new_tokens,
            max_num_seqs=1,
            tensor_parallel_size=1,
            trust_remote_code=False,
            disable_log_stats=True,
        )
        self.supported_languages = tuple(self._model.get_supported_languages())
        self.runtime = (
            f"qwen-asr=={importlib.metadata.version('qwen-asr')};"
            f"vllm=={importlib.metadata.version('vllm')}"
        )

    def open(self, language: Optional[str]):
        return self._model.init_streaming_state(
            language=language or None,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=self._chunk_size_sec,
        )

    def push(self, pcm16: np.ndarray, state):
        return self._model.streaming_transcribe(pcm16, state)

    def finish(self, state):
        return self._model.finish_streaming_transcribe(state)


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _provenance(runtime: str) -> speech_provider_pb2.ProviderProvenance:
    return speech_provider_pb2.ProviderProvenance(
        profile_id=PROFILE_ID,
        runtime=runtime,
        model_revision=MODEL_REVISION,
        model_digest=MODEL_DIGEST,
        implementation_revision="xamurai-qwen-native-stream-v1",
    )


def _error(code: int, message: str, *, retryable: bool = False) -> speech_provider_pb2.ProviderError:
    return speech_provider_pb2.ProviderError(code=code, safe_message=message, retryable=retryable)


class QwenProviderServicer(speech_provider_pb2_grpc.SpeechProviderServicer):
    def __init__(self, backend: QwenBackend, *, maximum_audio_seconds: float = 300.0) -> None:
        self._backend = backend
        self._maximum_audio_seconds = max(1.0, min(float(maximum_audio_seconds), 1800.0))
        self._slot = threading.BoundedSemaphore(1)
        self._provenance = _provenance(backend.runtime)

    def GetCapabilities(self, request, context):
        if request.profile_id != PROFILE_ID:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "speech provider profile is not available")
        return speech_provider_pb2.CapabilitiesResponse(
            capabilities=speech_provider_pb2.ProviderCapabilities(
                windowed_realtime=False,
                native_streaming=True,
                batch=False,
                segment_timestamps=False,
                word_timestamps=False,
                language_detection=True,
                supported_languages=self._backend.supported_languages,
                stateful=True,
                preferred_sample_rate=SAMPLE_RATE,
                maximum_audio_seconds=self._maximum_audio_seconds,
                maximum_concurrent_sessions=1,
            ),
            provenance=self._provenance,
        )

    def Health(self, request, context):
        return speech_provider_pb2.HealthResponse(
            ready=True,
            status="ready",
            provenance=self._provenance,
        )

    def TranscribeWindow(self, request, context):
        return speech_provider_pb2.TranscribeWindowResponse(
            error=_error(
                speech_provider_pb2.PROVIDER_ERROR_UNSUPPORTED_CAPABILITY,
                "Qwen native profile does not expose window transcription",
            )
        )

    def StreamTranscribe(self, request_iterator, context):
        if not self._slot.acquire(blocking=False):
            first = next(request_iterator, None)
            if first is not None:
                yield self._event(
                    first,
                    0,
                    "",
                    "",
                    error=_error(
                        speech_provider_pb2.PROVIDER_ERROR_OVERLOADED,
                        "Qwen provider has reached its session limit",
                        retryable=True,
                    ),
                )
            return

        state = None
        total_samples = 0
        expected_sequence = 1
        request_id = None
        provider_session_id = None
        language = None
        try:
            for frame in request_iterator:
                validation_error = self._validate_frame(
                    frame,
                    request_id=request_id,
                    provider_session_id=provider_session_id,
                    expected_sequence=expected_sequence,
                    language=language,
                    total_samples=total_samples,
                )
                if validation_error is not None:
                    yield self._event(frame, total_samples, "", language or "", error=validation_error)
                    return

                if state is None:
                    request_id = frame.request_id
                    provider_session_id = frame.provider_session_id
                    language = frame.language or None
                    state = self._backend.open(language)

                pcm16 = np.frombuffer(frame.pcm16_le, dtype="<i2")
                total_samples += int(pcm16.size)
                if pcm16.size:
                    state = self._backend.push(pcm16, state)
                if frame.end_of_stream:
                    state = self._backend.finish(state)

                yield self._event(
                    frame,
                    total_samples,
                    str(getattr(state, "text", "") or ""),
                    str(getattr(state, "language", "") or language or ""),
                    terminal=frame.end_of_stream,
                )
                expected_sequence += 1
                if frame.end_of_stream:
                    return
        except Exception as exc:
            logger.error("Qwen provider stream failed error_type=%s", type(exc).__name__)
            if request_id is not None:
                synthetic = speech_provider_pb2.ProviderAudioFrame(
                    request_id=request_id,
                    provider_session_id=provider_session_id,
                    provider_sequence=expected_sequence,
                )
                yield self._event(
                    synthetic,
                    total_samples,
                    "",
                    language or "",
                    error=_error(
                        speech_provider_pb2.PROVIDER_ERROR_INTERNAL,
                        "Qwen provider inference failed",
                    ),
                )
        finally:
            self._slot.release()

    def _validate_frame(
        self,
        frame,
        *,
        request_id: Optional[str],
        provider_session_id: Optional[str],
        expected_sequence: int,
        language: Optional[str],
        total_samples: int,
    ) -> Optional[speech_provider_pb2.ProviderError]:
        invalid = speech_provider_pb2.PROVIDER_ERROR_INVALID_REQUEST
        if frame.profile_id != PROFILE_ID:
            return _error(invalid, "speech provider profile is not available")
        if not frame.request_id or not frame.provider_session_id:
            return _error(invalid, "provider request and session IDs are required")
        if request_id is not None and (frame.request_id != request_id or frame.provider_session_id != provider_session_id):
            return _error(invalid, "provider stream identity changed")
        if frame.provider_sequence != expected_sequence:
            return _error(invalid, "provider frame sequence is not contiguous")
        if frame.sample_rate != SAMPLE_RATE:
            return _error(invalid, "Qwen provider accepts PCM16 mono at 16000 Hz")
        if len(frame.pcm16_le) % 2:
            return _error(invalid, "PCM16 payload length must be even")
        if language is not None and (frame.language or None) != language:
            return _error(invalid, "language changed within provider stream")
        samples = total_samples + len(frame.pcm16_le) // 2
        if samples > int(self._maximum_audio_seconds * SAMPLE_RATE):
            return _error(invalid, "provider stream exceeds configured maximum")
        return None

    def _event(
        self,
        frame,
        total_samples: int,
        text: str,
        language: str,
        *,
        terminal: bool = False,
        error: Optional[speech_provider_pb2.ProviderError] = None,
    ) -> speech_provider_pb2.ProviderTranscriptEvent:
        return speech_provider_pb2.ProviderTranscriptEvent(
            request_id=frame.request_id,
            provider_session_id=frame.provider_session_id,
            provider_sequence=frame.provider_sequence,
            text=text,
            language=language,
            start_sample=0,
            end_sample=total_samples,
            terminal=terminal,
            provenance=self._provenance,
            error=error,
        )


def create_server(backend: Optional[QwenBackend] = None, *, port: Optional[int] = None) -> grpc.Server:
    runtime = backend or QwenVllmBackend()
    maximum_audio_seconds = _bounded_float("QWEN_MAX_AUDIO_SECONDS", 300.0, 1.0, 1800.0)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    speech_provider_pb2_grpc.add_SpeechProviderServicer_to_server(
        QwenProviderServicer(runtime, maximum_audio_seconds=maximum_audio_seconds),
        server,
    )
    bind_addr = os.getenv("QWEN_PROVIDER_BIND_ADDR", "[::]").strip() or "[::]"
    selected_port = port if port is not None else int(os.getenv("QWEN_PROVIDER_PORT", "50061"))
    server.bound_port = server.add_insecure_port(f"{bind_addr}:{selected_port}")  # type: ignore[attr-defined]
    return server


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "Loading Qwen provider profile=%s model_revision=%s model_digest=%s",
        PROFILE_ID,
        MODEL_REVISION,
        MODEL_DIGEST,
    )
    server = create_server()
    server.start()
    logger.info("Qwen provider ready port=%d", server.bound_port)  # type: ignore[attr-defined]
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)


if __name__ == "__main__":
    main()
