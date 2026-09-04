from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import math
import os
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import grpc
import numpy as np

from proto_gen import stream_pb2, stream_pb2_grpc
from qwen_rtservice.enrichment import (
    EnrichmentFailure,
    FinalEnricher,
    PyannoteDiarizer,
    QwenEpochEnricher,
)
from xamurai_serving import SessionSlots, max_sessions_from_env, serving_instance_id


logger = logging.getLogger(__name__)

PROFILE_ID = "qwen3-asr-0.6b-vllm-aligned-diarized-r3"
MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
MODEL_REVISION = "c4468bdb552ddc559e464f6081e22dd4034f2e68"
MODEL_DIGEST = "sha256:79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
ALIGNMENT_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNMENT_MODEL_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
ALIGNMENT_MODEL_DIGEST = "sha256:47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7"
SAMPLE_RATE = 16000
AUDIO_TOKENS_PER_SECOND = 13
KV_CACHE_BYTES_PER_TOKEN = 114688
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


@dataclass(frozen=True)
class QwenRuntimeConfig:
    """Validated Qwen streaming and fixed-profile vLLM memory settings."""

    stream_chunk_seconds: float
    stream_epoch_seconds: float
    epoch_context_characters: int
    max_new_tokens: int
    max_model_len: int
    kv_cache_mib: int


class QwenBackend(Protocol):
    runtime: str
    supported_languages: tuple[str, ...]
    alignment_languages: tuple[str, ...]

    def open(self, language: Optional[str], context: str): ...

    def push(self, pcm16: np.ndarray, state): ...

    def finish(self, state): ...

    def align(self, pcm16: np.ndarray, text: str, language: str): ...


def _verified_snapshot(repo_id: str, revision: str, expected_digest: str) -> str:
    """Download one immutable model snapshot and verify its safetensors payload."""
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=repo_id, revision=revision)
    weights = Path(model_path) / "model.safetensors"
    if not weights.is_file():
        raise RuntimeError(f"pinned model artifact is incomplete: {repo_id}")
    digest = hashlib.sha256()
    with weights.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if f"sha256:{digest.hexdigest()}" != expected_digest:
        raise RuntimeError(
            f"pinned model artifact digest does not match the service profile: {repo_id}"
        )
    return model_path


class QwenVllmBackend:
    """Exact-revision Qwen3-ASR runtime using its native vLLM streaming API."""

    def __init__(self, config: QwenRuntimeConfig) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen realtime service requires an NVIDIA CUDA device")

        model_path = _verified_snapshot(MODEL_ID, MODEL_REVISION, MODEL_DIGEST)
        aligner_path = _verified_snapshot(
            ALIGNMENT_MODEL_ID,
            ALIGNMENT_MODEL_REVISION,
            ALIGNMENT_MODEL_DIGEST,
        )

        self._chunk_size_sec = config.stream_chunk_seconds
        self._model = Qwen3ASRModel.LLM(
            model=model_path,
            max_new_tokens=config.max_new_tokens,
            max_model_len=config.max_model_len,
            kv_cache_memory_bytes=config.kv_cache_mib * 1024 * 1024,
            max_num_seqs=1,
            limit_mm_per_prompt={"audio": 1},
            tensor_parallel_size=1,
            trust_remote_code=False,
            disable_log_stats=True,
            forced_aligner=aligner_path,
            forced_aligner_kwargs={
                "dtype": torch.bfloat16,
                "device_map": "cuda:0",
                "trust_remote_code": False,
            },
        )
        self.supported_languages = tuple(self._model.get_supported_languages())
        forced_aligner = getattr(self._model, "forced_aligner", None)
        if forced_aligner is None:
            raise RuntimeError("pinned Qwen forced aligner did not initialize")
        self._forced_aligner = forced_aligner
        supported_alignment_languages = forced_aligner.get_supported_languages() or ()
        language_names = {name.casefold(): name for name in QWEN_LANGUAGE_CODES}
        self.alignment_languages = tuple(
            language_names[name.casefold()]
            for name in supported_alignment_languages
            if name.casefold() in language_names
        )
        self.runtime = (
            f"qwen-asr=={importlib.metadata.version('qwen-asr')};"
            f"vllm=={importlib.metadata.version('vllm')};"
            f"pyannote-audio=={importlib.metadata.version('pyannote-audio')}"
        )

    def open(self, language: Optional[str], context: str):
        return self._model.init_streaming_state(
            context=context,
            language=language or None,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=self._chunk_size_sec,
        )

    def push(self, pcm16: np.ndarray, state):
        return self._model.streaming_transcribe(pcm16, state)

    def finish(self, state):
        return self._model.finish_streaming_transcribe(state)

    def align(self, pcm16: np.ndarray, text: str, language: str):
        wave = pcm16.astype(np.float32) / 32768.0
        results = self._forced_aligner.align(
            audio=(wave, SAMPLE_RATE),
            text=text,
            language=language,
        )
        if len(results) != 1:
            raise RuntimeError("forced alignment returned an unexpected result count")
        return results[0]


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


def _load_runtime_config() -> QwenRuntimeConfig:
    """Load bounded epoch and vLLM settings and reject incompatible combinations."""
    config = QwenRuntimeConfig(
        stream_chunk_seconds=_bounded_float("QWEN_STREAM_CHUNK_SECONDS", 2.0, 0.5, 10.0),
        stream_epoch_seconds=_bounded_float("QWEN_STREAM_EPOCH_SECONDS", 60.0, 10.0, 300.0),
        epoch_context_characters=_bounded_int("QWEN_EPOCH_CONTEXT_CHARACTERS", 1000, 500, 8000),
        max_new_tokens=_bounded_int("QWEN_MAX_NEW_TOKENS", 1024, 16, 1024),
        max_model_len=_bounded_int("QWEN_MAX_MODEL_LEN", 4096, 2048, 65536),
        kv_cache_mib=_bounded_int("QWEN_KV_CACHE_MIB", 512, 512, 16384),
    )
    required_context_tokens = (
        math.ceil(config.stream_epoch_seconds * AUDIO_TOKENS_PER_SECOND)
        + config.epoch_context_characters
        + config.max_new_tokens
        + 128
    )
    if config.max_model_len < required_context_tokens:
        raise ValueError(
            "QWEN_MAX_MODEL_LEN is too small for the configured epoch, context, and output limits"
        )
    if config.kv_cache_mib * 1024 * 1024 < config.max_model_len * KV_CACHE_BYTES_PER_TOKEN:
        raise ValueError("QWEN_KV_CACHE_MIB is too small for QWEN_MAX_MODEL_LEN")
    return config


def _merge_transcript(committed: str, hypothesis: str) -> str:
    """Append an epoch hypothesis while removing an echoed committed suffix."""
    committed = str(committed or "").strip()
    hypothesis = str(hypothesis or "").strip()
    if not committed:
        return hypothesis
    if not hypothesis:
        return committed

    maximum_char_overlap = min(len(committed), len(hypothesis))
    for size in range(maximum_char_overlap, 7, -1):
        if committed[-size:].casefold() == hypothesis[:size].casefold():
            return (committed + hypothesis[size:]).strip()

    committed_words = committed.split()
    hypothesis_words = hypothesis.split()
    maximum_word_overlap = min(len(committed_words), len(hypothesis_words))
    for size in range(maximum_word_overlap, 0, -1):
        left = [
            "".join(ch for ch in word.casefold() if ch.isalnum())
            for word in committed_words[-size:]
        ]
        right = [
            "".join(ch for ch in word.casefold() if ch.isalnum())
            for word in hypothesis_words[:size]
        ]
        if all(left) and left == right:
            return " ".join(committed_words + hypothesis_words[size:]).strip()
    return f"{committed} {hypothesis}".strip()


def _novel_epoch_text(committed: str, hypothesis: str) -> str:
    """Return only transcript text not already committed by an earlier epoch."""
    committed = str(committed or "").strip()
    merged = _merge_transcript(committed, hypothesis)
    if committed and merged.startswith(committed):
        return merged[len(committed) :].strip()
    return merged


class QwenRealtimeServicer(stream_pb2_grpc.RealtimeASRServicer):
    """Expose one pinned Qwen/vLLM profile through the public realtime API."""

    def __init__(
        self,
        backend: QwenBackend,
        *,
        enricher: Optional[FinalEnricher] = None,
        epoch_seconds: float = 60.0,
        context_characters: int = 1000,
    ) -> None:
        self._backend = backend
        self._enricher = enricher
        self._epoch_samples = max(1, int(float(epoch_seconds) * SAMPLE_RATE))
        self._context_characters = max(0, int(context_characters))
        self._slots = SessionSlots(max_sessions_from_env())
        self._instance_id = serving_instance_id()

    def GetCapabilities(self, request, context):
        """Describe the fixed Qwen profile without accepting model selection."""
        return stream_pb2.RealtimeCapabilities(
            provider_profile_id=PROFILE_ID,
            windowed_realtime=False,
            native_streaming=True,
            batch=False,
            segment_timestamps=bool(self._enricher and self._enricher.supported_languages),
            word_timestamps=False,
            language_detection=True,
            supported_languages=tuple(
                QWEN_LANGUAGE_CODES[name]
                for name in self._backend.supported_languages
                if name in QWEN_LANGUAGE_CODES
            ),
            stateful=True,
            preferred_sample_rate=SAMPLE_RATE,
            maximum_audio_seconds=0,
            maximum_concurrent_sessions=self._slots.maximum,
            runtime=self._backend.runtime,
            model_revision=MODEL_REVISION,
            model_digest=MODEL_DIGEST,
            implementation_revision="xamurai-qwen-realtime-v4",
            speaker_labels=bool(self._enricher and self._enricher.supported_languages),
            aligned_diarized_languages=tuple(
                QWEN_LANGUAGE_CODES[name]
                for name in (self._enricher.supported_languages if self._enricher else ())
                if name in QWEN_LANGUAGE_CODES
            ),
        )

    def Stream(self, request_iterator, context):
        """Transcribe one public audio stream and flush a final result at EOF."""
        metadata = {
            str(key).lower(): str(value)
            for key, value in (context.invocation_metadata() or ())
        }
        opening_session_id = metadata.get("x-session-id", "").strip()
        if not opening_session_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "SESSION_ID_REQUIRED")
        if not self._slots.acquire():
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "REPLICA_FULL")

        state = None
        total_samples = 0
        epoch_samples = 0
        epoch_start_samples = 0
        epoch_number = 1
        epoch_audio: list[np.ndarray] = []
        committed_text = ""
        last_emitted_text = ""
        last_language = ""
        session_id = None
        expected_sequence = None
        language = None
        try:
            yield stream_pb2.AsrEvent(
                session_id=opening_session_id,
                type=stream_pb2.SESSION_ACCEPTED,
                provider_profile_id=PROFILE_ID,
                serving_instance_id=self._instance_id,
            )
            for chunk in request_iterator:
                if chunk.session_id != opening_session_id:
                    raise _InvalidRequest("SESSION_ID_MISMATCH")
                error = self._validate_chunk(
                    chunk,
                    session_id=session_id,
                    expected_sequence=expected_sequence,
                    language=language,
                )
                if error is not None:
                    raise _InvalidRequest(error)

                if state is None:
                    session_id = chunk.session_id
                    expected_sequence = chunk.seq
                    language = chunk.lang or None
                    state = self._backend.open(QWEN_LANGUAGE_NAMES.get(language, language), "")

                expected_sequence = chunk.seq + 1
                pcm16 = np.frombuffer(chunk.pcm16_le, dtype="<i2")
                offset = 0
                while offset < pcm16.size:
                    if epoch_samples >= self._epoch_samples:
                        state = self._backend.finish(state)
                        epoch_text = _novel_epoch_text(
                            committed_text,
                            getattr(state, "text", ""),
                        )
                        detected_language = str(getattr(state, "language", "") or "")
                        if detected_language:
                            last_language = detected_language
                        if epoch_text:
                            for event in self._final_events(
                                chunk.session_id,
                                epoch_start_samples,
                                total_samples,
                                epoch_text,
                                last_language or QWEN_LANGUAGE_NAMES.get(language or "", ""),
                                (
                                    np.concatenate(epoch_audio)
                                    if epoch_audio
                                    else np.zeros(0, dtype=np.int16)
                                ),
                                epoch_number,
                            ):
                                yield event
                            committed_text = f"{committed_text} {epoch_text}".strip()
                        logger.info(
                            "Qwen realtime epoch completed session=%s epoch=%d total_samples=%d",
                            chunk.session_id,
                            epoch_number,
                            total_samples,
                        )
                        context_text = (
                            committed_text[-self._context_characters :]
                            if self._context_characters
                            else ""
                        )
                        state = self._backend.open(
                            QWEN_LANGUAGE_NAMES.get(language, language),
                            context_text,
                        )
                        epoch_samples = 0
                        epoch_start_samples = total_samples
                        epoch_number += 1
                        epoch_audio = []
                        last_emitted_text = ""

                    take = min(pcm16.size - offset, self._epoch_samples - epoch_samples)
                    epoch_part = pcm16[offset : offset + take]
                    epoch_audio.append(epoch_part.copy())
                    state = self._backend.push(epoch_part, state)
                    offset += take
                    epoch_samples += take
                    total_samples += take
                    detected_language = str(getattr(state, "language", "") or "")
                    if detected_language:
                        last_language = detected_language
                    public_text = _novel_epoch_text(committed_text, getattr(state, "text", ""))
                    if public_text and public_text != last_emitted_text:
                        yield self._event(
                            chunk.session_id,
                            epoch_start_samples,
                            total_samples,
                            public_text,
                            last_language,
                            final=False,
                        )
                        last_emitted_text = public_text

            if state is not None and context.is_active():
                state = self._backend.finish(state)
                epoch_text = _novel_epoch_text(committed_text, getattr(state, "text", ""))
                detected_language = str(getattr(state, "language", "") or "")
                if detected_language:
                    last_language = detected_language
                for event in self._final_events(
                    session_id or "",
                    epoch_start_samples,
                    total_samples,
                    epoch_text,
                    last_language or QWEN_LANGUAGE_NAMES.get(language or "", ""),
                    np.concatenate(epoch_audio) if epoch_audio else np.zeros(0, dtype=np.int16),
                    epoch_number,
                ):
                    yield event
        except _InvalidRequest as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except grpc.RpcError as exc:
            code = exc.code()
            if context.is_active() and code != grpc.StatusCode.CANCELLED:
                logger.warning(
                    "Qwen realtime transport ended code=%s",
                    getattr(code, "name", "unknown"),
                )
        except Exception as exc:
            if context.is_active():
                logger.error("Qwen realtime stream failed error_type=%s", type(exc).__name__)
                context.abort(grpc.StatusCode.INTERNAL, "Qwen realtime inference failed")
        finally:
            self._slots.release()

    def _validate_chunk(
        self,
        chunk,
        *,
        session_id: Optional[str],
        expected_sequence: Optional[int],
        language: Optional[str],
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
        return None

    def _final_events(
        self,
        session_id: str,
        epoch_start_samples: int,
        total_samples: int,
        text: str,
        language: str,
        epoch_audio: np.ndarray,
        epoch_number: int,
    ) -> tuple[stream_pb2.AsrEvent, ...]:
        language_name = QWEN_LANGUAGE_NAMES.get(language, language)
        if (
            text
            and self._enricher is not None
            and language_name in self._enricher.supported_languages
        ):
            try:
                segments = self._enricher.enrich(
                    epoch_audio,
                    text,
                    language_name,
                    epoch_number=epoch_number,
                )
                epoch_offset_s = epoch_start_samples / SAMPLE_RATE
                return tuple(
                    self._event(
                        session_id,
                        int(round((epoch_offset_s + segment.start_s) * SAMPLE_RATE)),
                        int(round((epoch_offset_s + segment.end_s) * SAMPLE_RATE)),
                        segment.text,
                        language_name,
                        final=True,
                        speaker=segment.speaker,
                    )
                    for segment in segments
                )
            except EnrichmentFailure as exc:
                logger.warning(
                    "Qwen realtime epoch enrichment failed epoch=%d stage=%s error_type=%s",
                    epoch_number,
                    exc.stage,
                    exc.cause_type,
                )
            except Exception as exc:
                logger.warning(
                    "Qwen realtime epoch enrichment failed epoch=%d error_type=%s",
                    epoch_number,
                    type(exc).__name__,
                )
        return (
            self._event(
                session_id,
                epoch_start_samples,
                total_samples,
                text,
                language_name,
                final=True,
            ),
        )

    def _event(
        self,
        session_id: str,
        start_samples: int,
        total_samples: int,
        text: str,
        language: str,
        *,
        final: bool,
        speaker: str = "",
    ):
        return stream_pb2.AsrEvent(
            session_id=session_id,
            start_s=start_samples / SAMPLE_RATE,
            end_s=total_samples / SAMPLE_RATE,
            text=str(text or "").strip(),
            type=stream_pb2.FINAL if final else stream_pb2.PARTIAL,
            lang=QWEN_LANGUAGE_CODES.get(str(language or ""), ""),
            speaker=speaker,
            provider_profile_id=PROFILE_ID,
        )


def create_server(
    backend: Optional[QwenBackend] = None,
    *,
    enricher: Optional[FinalEnricher] = None,
    port: Optional[int] = None,
) -> grpc.Server:
    """Create the Qwen RealtimeASR server without starting its worker threads."""
    config = _load_runtime_config()
    runtime = backend or QwenVllmBackend(config)
    final_enricher = enricher
    if backend is None and final_enricher is None:
        final_enricher = QwenEpochEnricher(runtime, PyannoteDiarizer())
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    stream_pb2_grpc.add_RealtimeASRServicer_to_server(
        QwenRealtimeServicer(
            runtime,
            enricher=final_enricher,
            epoch_seconds=config.stream_epoch_seconds,
            context_characters=config.epoch_context_characters,
        ),
        server,
    )
    bind_addr = os.getenv("QWEN_RTSERVICE_BIND_ADDR", "[::]").strip() or "[::]"
    selected_port = port if port is not None else int(os.getenv("QWEN_RTSERVICE_PORT", "50052"))
    server.bound_port = server.add_insecure_port(  # type: ignore[attr-defined]
        f"{bind_addr}:{selected_port}"
    )
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
    logger.info(
        "Qwen realtime service ready port=%d",
        server.bound_port,  # type: ignore[attr-defined]
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)


if __name__ == "__main__":
    main()
