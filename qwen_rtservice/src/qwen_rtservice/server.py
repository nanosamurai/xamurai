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

from proto_gen import stream_pb2, stream_pb2_grpc


logger = logging.getLogger(__name__)

PROFILE_ID = "qwen3-asr-0.6b-vllm-r1"
MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
MODEL_REVISION = "c4468bdb552ddc559e464f6081e22dd4034f2e68"
MODEL_DIGEST = "sha256:79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
SAMPLE_RATE = 16000
QWEN_LANGUAGE_CODES = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian": "it",
    "Korean": "ko",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Turkish": "tr",
    "Hindi": "hi",
    "Malay": "ms",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Polish": "pl",
    "Czech": "cs",
    "Filipino": "fil",
    "Persian": "fa",
    "Greek": "el",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Macedonian": "mk",
}
QWEN_LANGUAGE_NAMES = {code: name for name, code in QWEN_LANGUAGE_CODES.items()}


class _InvalidRequest(ValueError):
    """Validated public-stream error safe to return to the caller."""


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
            raise RuntimeError("Qwen realtime service requires an NVIDIA CUDA device")

        model_path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
        weights = Path(model_path) / "model.safetensors"
        if not weights.is_file():
            raise RuntimeError("pinned Qwen model artifact is incomplete")
        digest = hashlib.sha256()
        with weights.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if f"sha256:{digest.hexdigest()}" != MODEL_DIGEST:
            raise RuntimeError("pinned Qwen model artifact digest does not match the service profile")

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


class QwenRealtimeServicer(stream_pb2_grpc.RealtimeASRServicer):
    """Expose one pinned Qwen/vLLM profile through the public realtime API."""

    def __init__(self, backend: QwenBackend, *, maximum_audio_seconds: float = 300.0) -> None:
        self._backend = backend
        self._maximum_audio_seconds = max(1.0, min(float(maximum_audio_seconds), 1800.0))
        self._slot = threading.BoundedSemaphore(1)

    def GetCapabilities(self, request, context):
        """Describe the fixed Qwen profile without accepting model selection."""
        return stream_pb2.RealtimeCapabilities(
            provider_profile_id=PROFILE_ID,
            windowed_realtime=False,
            native_streaming=True,
            batch=False,
            segment_timestamps=False,
            word_timestamps=False,
            language_detection=True,
            supported_languages=tuple(
                QWEN_LANGUAGE_CODES[name]
                for name in self._backend.supported_languages
                if name in QWEN_LANGUAGE_CODES
            ),
            stateful=True,
            preferred_sample_rate=SAMPLE_RATE,
            maximum_audio_seconds=self._maximum_audio_seconds,
            maximum_concurrent_sessions=1,
            runtime=self._backend.runtime,
            model_revision=MODEL_REVISION,
            model_digest=MODEL_DIGEST,
            implementation_revision="xamurai-qwen-realtime-v2",
        )

    def Stream(self, request_iterator, context):
        """Transcribe one public audio stream and flush a final result at EOF."""
        if not self._slot.acquire(blocking=False):
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Qwen realtime service has reached its session limit")

        state = None
        total_samples = 0
        session_id = None
        expected_sequence = None
        language = None
        try:
            for chunk in request_iterator:
                error = self._validate_chunk(
                    chunk,
                    session_id=session_id,
                    expected_sequence=expected_sequence,
                    language=language,
                    total_samples=total_samples,
                )
                if error is not None:
                    raise _InvalidRequest(error)

                if state is None:
                    session_id = chunk.session_id
                    expected_sequence = chunk.seq
                    language = chunk.lang or None
                    state = self._backend.open(QWEN_LANGUAGE_NAMES.get(language, language))

                expected_sequence = chunk.seq + 1
                pcm16 = np.frombuffer(chunk.pcm16_le, dtype="<i2")
                total_samples += int(pcm16.size)
                if pcm16.size:
                    state = self._backend.push(pcm16, state)
                    text = str(getattr(state, "text", "") or "").strip()
                    if text:
                        yield self._event(chunk.session_id, total_samples, state, terminal=False)

            if state is not None and context.is_active():
                state = self._backend.finish(state)
                yield self._event(session_id or "", total_samples, state, terminal=True)
        except _InvalidRequest as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except grpc.RpcError as exc:
            code = exc.code()
            if context.is_active() and code != grpc.StatusCode.CANCELLED:
                logger.warning("Qwen realtime transport ended code=%s", getattr(code, "name", "unknown"))
        except Exception as exc:
            if context.is_active():
                logger.error("Qwen realtime stream failed error_type=%s", type(exc).__name__)
                context.abort(grpc.StatusCode.INTERNAL, "Qwen realtime inference failed")
        finally:
            self._slot.release()

    def _validate_chunk(
        self,
        chunk,
        *,
        session_id: Optional[str],
        expected_sequence: Optional[int],
        language: Optional[str],
        total_samples: int,
    ) -> Optional[str]:
        if not chunk.session_id:
            return "session ID is required"
        if session_id is not None and chunk.session_id != session_id:
            return "session ID changed within realtime stream"
        if expected_sequence is not None and chunk.seq != expected_sequence:
            return "audio chunk sequence is not contiguous"
        if chunk.sample_rate != SAMPLE_RATE:
            return "Qwen realtime service accepts PCM16 mono at 16000 Hz"
        if len(chunk.pcm16_le) % 2:
            return "PCM16 payload length must be even"
        if session_id is None and chunk.lang and chunk.lang not in QWEN_LANGUAGE_NAMES:
            return "Qwen realtime service does not support the requested language code"
        if language is not None and (chunk.lang or None) != language:
            return "language changed within realtime stream"
        samples = total_samples + len(chunk.pcm16_le) // 2
        if samples > int(self._maximum_audio_seconds * SAMPLE_RATE):
            return "realtime stream exceeds configured maximum"
        return None

    def _event(self, session_id: str, total_samples: int, state, *, terminal: bool):
        return stream_pb2.AsrEvent(
            session_id=session_id,
            start_s=0.0,
            end_s=total_samples / SAMPLE_RATE,
            text=str(getattr(state, "text", "") or "").strip(),
            type=stream_pb2.FINAL if terminal else stream_pb2.PARTIAL,
            lang=QWEN_LANGUAGE_CODES.get(str(getattr(state, "language", "") or ""), ""),
            provider_profile_id=PROFILE_ID,
        )


def create_server(backend: Optional[QwenBackend] = None, *, port: Optional[int] = None) -> grpc.Server:
    """Create the Qwen RealtimeASR server without starting its worker threads."""
    runtime = backend or QwenVllmBackend()
    maximum_audio_seconds = _bounded_float("QWEN_MAX_AUDIO_SECONDS", 300.0, 1.0, 1800.0)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    stream_pb2_grpc.add_RealtimeASRServicer_to_server(
        QwenRealtimeServicer(runtime, maximum_audio_seconds=maximum_audio_seconds),
        server,
    )
    bind_addr = os.getenv("QWEN_RTSERVICE_BIND_ADDR", "[::]").strip() or "[::]"
    selected_port = port if port is not None else int(os.getenv("QWEN_RTSERVICE_PORT", "50052"))
    server.bound_port = server.add_insecure_port(f"{bind_addr}:{selected_port}")  # type: ignore[attr-defined]
    return server


def main() -> None:
    """Load the pinned Qwen model and serve RealtimeASR until termination."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "Loading Qwen realtime profile=%s model_revision=%s model_digest=%s",
        PROFILE_ID,
        MODEL_REVISION,
        MODEL_DIGEST,
    )
    server = create_server()
    server.start()
    logger.info("Qwen realtime service ready port=%d", server.bound_port)  # type: ignore[attr-defined]
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)


if __name__ == "__main__":
    main()
