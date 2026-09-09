from __future__ import annotations

import ctypes
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_FILENAME = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
MODEL_REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"
MODEL_DIGEST = "sha256:a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae"
NEMO_SPEECH_VERSION = "0.1.0"
NEMO_SPEECH_REVISION = "4f9676226f667d14608487df744f375db87127f8"
ENDPOINTING_SILENCE_MS = 800
MAX_UTTERANCE_SECONDS = 30.0


@dataclass(frozen=True)
class TranscriptUpdate:
    text: str
    final: bool
    audio_processed_s: float
    language: str


class NativeRuntimeError(RuntimeError):
    """A deliberately detail-free native runtime failure."""


class _BackendConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("gpu", ctypes.c_int32)]


class _ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("path", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
    ]


class _StreamingConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("chunk_size", ctypes.c_float),
        ("ctc_left_padding", ctypes.c_float),
        ("ctc_right_padding", ctypes.c_float),
        ("rnnt_right_context", ctypes.c_int32),
    ]


class _BatchingConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("enable", ctypes.c_bool),
        ("max_batch_size", ctypes.c_int32),
        ("max_queue_delay_us", ctypes.c_int32),
        ("max_queue_depth", ctypes.c_int32),
        ("ingress_cohort_delay_us", ctypes.c_int32),
        ("state_arena_slots", ctypes.c_int32),
    ]


class _EndpointingConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("enable", ctypes.c_bool),
        ("vad_based", ctypes.c_bool),
        ("stop_history_eou_ms", ctypes.c_int32),
    ]


class _RecognizerConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("backend", ctypes.POINTER(_BackendConfig)),
        ("model", ctypes.POINTER(_ModelConfig)),
        ("streaming", ctypes.POINTER(_StreamingConfig)),
        ("decoder", ctypes.c_void_p),
        ("vad", ctypes.c_void_p),
        ("endpointing", ctypes.POINTER(_EndpointingConfig)),
        ("postproc", ctypes.c_void_p),
        ("diar", ctypes.c_void_p),
        ("batching", ctypes.POINTER(_BatchingConfig)),
    ]


class _RecognitionOptions(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("request_id", ctypes.c_char_p),
        ("language_code", ctypes.c_char_p),
        ("interim_results", ctypes.c_bool),
        ("enable_word_time_offsets", ctypes.c_bool),
        ("enable_automatic_punctuation", ctypes.c_bool),
        ("verbatim_transcripts", ctypes.c_bool),
        ("profanity_filter", ctypes.c_bool),
        ("stop_history_eou_ms", ctypes.c_int32),
        ("speech_contexts", ctypes.c_void_p),
        ("speech_context_count", ctypes.c_size_t),
        ("max_alternatives", ctypes.c_int32),
        ("enable_speaker_diarization", ctypes.c_bool),
        ("max_speaker_count", ctypes.c_int32),
    ]


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _verified_model_path() -> str:
    """Download exactly one immutable GGUF artifact and verify its content."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
        )
    )
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if f"sha256:{digest.hexdigest()}" != MODEL_DIGEST:
        raise RuntimeError("pinned Nemotron model artifact digest does not match")
    return str(path)


def _configure_library(library: ctypes.CDLL) -> None:
    handle_ptr = ctypes.POINTER(ctypes.c_void_p)
    library.nemo_speech_asr_recognition_options_default.argtypes = []
    library.nemo_speech_asr_recognition_options_default.restype = _RecognitionOptions
    library.nemo_speech_asr_create.argtypes = [
        ctypes.POINTER(_RecognizerConfig),
        handle_ptr,
    ]
    library.nemo_speech_asr_create.restype = ctypes.c_int
    library.nemo_speech_asr_destroy.argtypes = [ctypes.c_void_p]
    library.nemo_speech_asr_streaming_recognize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_RecognitionOptions),
        handle_ptr,
    ]
    library.nemo_speech_asr_streaming_recognize.restype = ctypes.c_int
    library.nemo_speech_asr_stream_push_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_int32,
    ]
    library.nemo_speech_asr_stream_push_f32.restype = ctypes.c_int
    library.nemo_speech_asr_stream_next.argtypes = [ctypes.c_void_p, handle_ptr]
    library.nemo_speech_asr_stream_next.restype = ctypes.c_int
    library.nemo_speech_asr_stream_force_endpoint.argtypes = [ctypes.c_void_p]
    library.nemo_speech_asr_stream_force_endpoint.restype = ctypes.c_int
    library.nemo_speech_asr_stream_finish.argtypes = [ctypes.c_void_p]
    library.nemo_speech_asr_stream_finish.restype = ctypes.c_int
    library.nemo_speech_asr_stream_close.argtypes = [ctypes.c_void_p]
    library.nemo_speech_asr_result_is_final.argtypes = [ctypes.c_void_p]
    library.nemo_speech_asr_result_is_final.restype = ctypes.c_bool
    library.nemo_speech_asr_result_audio_processed.argtypes = [ctypes.c_void_p]
    library.nemo_speech_asr_result_audio_processed.restype = ctypes.c_float
    library.nemo_speech_asr_result_alternative_count.argtypes = [ctypes.c_void_p]
    library.nemo_speech_asr_result_alternative_count.restype = ctypes.c_size_t
    library.nemo_speech_asr_result_transcript.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.nemo_speech_asr_result_transcript.restype = ctypes.c_char_p
    library.nemo_speech_asr_result_language_count.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.nemo_speech_asr_result_language_count.restype = ctypes.c_size_t
    library.nemo_speech_asr_result_language_code.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    library.nemo_speech_asr_result_language_code.restype = ctypes.c_char_p
    library.nemo_speech_asr_result_destroy.argtypes = [ctypes.c_void_p]
    library.nemo_speech_asr_version.argtypes = []
    library.nemo_speech_asr_version.restype = ctypes.c_char_p


def _check(status: int, operation: str) -> None:
    if status != 0:
        raise NativeRuntimeError(f"native ASR operation failed: {operation} ({status})")


class NativeSession:
    """One cache-aware native stream; calls must stay on one worker thread."""

    def __init__(self, library: ctypes.CDLL, recognizer: ctypes.c_void_p, request_id: str,
                 language: Optional[str]) -> None:
        self._library = library
        self._handle = ctypes.c_void_p()
        self._closed = False
        self._request_id = request_id.encode("utf-8")
        self._language = language.encode("ascii") if language else None
        self._input_audio_s = 0.0
        self._last_final_audio_s = 0.0
        options = library.nemo_speech_asr_recognition_options_default()
        options.request_id = self._request_id
        options.language_code = self._language
        options.interim_results = True
        # The stable C ABI intentionally defaults every optional request flag
        # to false. Nemotron 3.5 is self-punctuating, so this gate preserves the
        # casing and punctuation emitted by the model without a PnC sidecar.
        options.enable_automatic_punctuation = True
        _check(
            library.nemo_speech_asr_streaming_recognize(
                recognizer, ctypes.byref(options), ctypes.byref(self._handle)
            ),
            "open stream",
        )

    def push(self, pcm16_le: bytes, sample_rate: int) -> tuple[TranscriptUpdate, ...]:
        pcm16 = np.frombuffer(pcm16_le, dtype="<i2")
        samples = np.ascontiguousarray(pcm16, dtype=np.float32)
        samples *= 1.0 / 32768.0
        pointer = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _check(
            self._library.nemo_speech_asr_stream_push_f32(
                self._handle, pointer, samples.size, sample_rate
            ),
            "push audio",
        )
        self._input_audio_s += samples.size / sample_rate
        updates = self._drain()
        if self._input_audio_s - self._last_final_audio_s >= MAX_UTTERANCE_SECONDS:
            _check(
                self._library.nemo_speech_asr_stream_force_endpoint(self._handle),
                "force endpoint",
            )
            updates += self._drain()
        return updates

    def finish(self) -> tuple[TranscriptUpdate, ...]:
        _check(self._library.nemo_speech_asr_stream_finish(self._handle), "finish stream")
        return self._drain()

    def _drain(self) -> tuple[TranscriptUpdate, ...]:
        updates: list[TranscriptUpdate] = []
        while True:
            result = ctypes.c_void_p()
            _check(
                self._library.nemo_speech_asr_stream_next(
                    self._handle, ctypes.byref(result)
                ),
                "read result",
            )
            if not result.value:
                break
            try:
                alternatives = self._library.nemo_speech_asr_result_alternative_count(result)
                transcript = (
                    self._library.nemo_speech_asr_result_transcript(result, 0)
                    if alternatives
                    else None
                )
                language = ""
                if alternatives and self._library.nemo_speech_asr_result_language_count(result, 0):
                    encoded = self._library.nemo_speech_asr_result_language_code(result, 0, 0)
                    language = encoded.decode("utf-8", errors="replace") if encoded else ""
                update = TranscriptUpdate(
                    text=transcript.decode("utf-8", errors="replace") if transcript else "",
                    final=bool(self._library.nemo_speech_asr_result_is_final(result)),
                    audio_processed_s=float(
                        self._library.nemo_speech_asr_result_audio_processed(result)
                    ),
                    language=language,
                )
                updates.append(update)
                if update.final:
                    self._last_final_audio_s = max(
                        self._last_final_audio_s, update.audio_processed_s
                    )
            finally:
                self._library.nemo_speech_asr_result_destroy(result)
        return tuple(updates)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._library.nemo_speech_asr_stream_close(self._handle)


class NativeNemotronBackend:
    """A shared NeMo-Speech.cpp recognizer with independent session streams."""

    runtime = f"nemo-speech-cpp=={NEMO_SPEECH_VERSION}"

    def __init__(self, maximum_sessions: int) -> None:
        library_path = os.getenv(
            "NEMO_SPEECH_LIBRARY",
            "/opt/nemo-speech/lib/libnemo_speech_asr_c.so",
        )
        self._library = ctypes.CDLL(library_path)
        _configure_library(self._library)
        reported_version = self._library.nemo_speech_asr_version()
        version = reported_version.decode("ascii", errors="replace") if reported_version else ""
        if version != f"nemo-speech-asr {NEMO_SPEECH_VERSION}":
            raise RuntimeError("NeMo-Speech.cpp runtime version does not match the image profile")

        model_path_bytes = _verified_model_path().encode("utf-8")
        backend = _BackendConfig(
            size=ctypes.sizeof(_BackendConfig),
            gpu=_bounded_int("NEMOTRON_GPU", 0, -1, 15),
        )
        model = _ModelConfig(
            size=ctypes.sizeof(_ModelConfig),
            path=model_path_bytes,
            name=b"nemotron-3.5-asr-streaming-0.6b",
        )
        streaming = _StreamingConfig(
            size=ctypes.sizeof(_StreamingConfig),
            chunk_size=0.16,
            ctc_left_padding=1.92,
            ctc_right_padding=1.92,
            rnnt_right_context=_bounded_int("NEMOTRON_RNNT_RIGHT_CONTEXT", 1, -1, 8),
        )
        batching = _BatchingConfig(
            size=ctypes.sizeof(_BatchingConfig),
            enable=maximum_sessions > 1,
            max_batch_size=maximum_sessions,
            max_queue_delay_us=2_000,
            max_queue_depth=max(4, maximum_sessions * 2),
            ingress_cohort_delay_us=1_000,
            state_arena_slots=maximum_sessions,
        )
        endpointing = _EndpointingConfig(
            size=ctypes.sizeof(_EndpointingConfig),
            enable=True,
            vad_based=False,
            stop_history_eou_ms=ENDPOINTING_SILENCE_MS,
        )
        config = _RecognizerConfig(
            size=ctypes.sizeof(_RecognizerConfig),
            backend=ctypes.pointer(backend),
            model=ctypes.pointer(model),
            streaming=ctypes.pointer(streaming),
            decoder=None,
            vad=None,
            endpointing=ctypes.pointer(endpointing),
            postproc=None,
            diar=None,
            batching=ctypes.pointer(batching),
        )
        self._recognizer = ctypes.c_void_p()
        _check(
            self._library.nemo_speech_asr_create(
                ctypes.byref(config), ctypes.byref(self._recognizer)
            ),
            "create recognizer",
        )

    def open(self, session_id: str, language: Optional[str]) -> NativeSession:
        return NativeSession(self._library, self._recognizer, session_id, language)

    def close(self) -> None:
        recognizer = getattr(self, "_recognizer", None)
        if recognizer is not None and recognizer.value:
            self._library.nemo_speech_asr_destroy(recognizer)
            recognizer.value = None

    def __del__(self) -> None:
        self.close()
