"""Shared bindings and verified artifacts for the pinned NeMo-Speech.cpp C API."""
import ctypes
import hashlib
import os
from pathlib import Path


NEMO_SPEECH_VERSION = "0.1.0"
NEMO_SPEECH_REVISION = "4f9676226f667d14608487df744f375db87127f8"
SORTFORMER_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2"
SORTFORMER_MODEL_FILENAME = "diar_streaming_sortformer_4spk-v2.q8_0.gguf"
SORTFORMER_MODEL_REVISION = "5240a64075176943f677d30fa2171c780229f341"
SORTFORMER_MODEL_DIGEST = "sha256:0679cfeb1ce356d0dea9470b31274f4bfc7eb927497d82005483770666da998a"


class NativeRuntimeError(RuntimeError):
    """A deliberately detail-free native runtime failure."""


class BackendConfig(ctypes.Structure):
    _fields_ = [("size", ctypes.c_size_t), ("gpu", ctypes.c_int32)]


class ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("path", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
    ]


class StreamingConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("chunk_size", ctypes.c_float),
        ("ctc_left_padding", ctypes.c_float),
        ("ctc_right_padding", ctypes.c_float),
        ("rnnt_right_context", ctypes.c_int32),
    ]


class BatchingConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("enable", ctypes.c_bool),
        ("max_batch_size", ctypes.c_int32),
        ("max_queue_delay_us", ctypes.c_int32),
        ("max_queue_depth", ctypes.c_int32),
        ("ingress_cohort_delay_us", ctypes.c_int32),
        ("state_arena_slots", ctypes.c_int32),
    ]


class EndpointingConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("enable", ctypes.c_bool),
        ("vad_based", ctypes.c_bool),
        ("stop_history_eou_ms", ctypes.c_int32),
    ]


class VadConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model_path", ctypes.c_char_p),
        ("enable_masking", ctypes.c_bool),
        ("onset", ctypes.c_float),
        ("offset", ctypes.c_float),
    ]


class DiarizationConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model_path", ctypes.c_char_p),
        ("chunk_frames", ctypes.c_int32),
        ("right_context_frames", ctypes.c_int32),
        ("left_context_frames", ctypes.c_int32),
        ("fifo_frames", ctypes.c_int32),
        ("spkcache_frames", ctypes.c_int32),
        ("update_period_frames", ctypes.c_int32),
    ]


class RecognizerConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("backend", ctypes.POINTER(BackendConfig)),
        ("model", ctypes.POINTER(ModelConfig)),
        ("streaming", ctypes.POINTER(StreamingConfig)),
        ("decoder", ctypes.c_void_p),
        ("vad", ctypes.POINTER(VadConfig)),
        ("endpointing", ctypes.POINTER(EndpointingConfig)),
        ("postproc", ctypes.c_void_p),
        ("diar", ctypes.POINTER(DiarizationConfig)),
        ("batching", ctypes.POINTER(BatchingConfig)),
    ]


class RecognitionOptions(ctypes.Structure):
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


def bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def verified_model_path(model_id: str, filename: str, revision: str, expected_digest: str) -> str:
    """Download exactly one immutable GGUF artifact and verify its content."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
        )
    )
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if f"sha256:{digest.hexdigest()}" != expected_digest:
        raise RuntimeError("pinned speech model artifact digest does not match")
    return str(path)


def configure_library(library: ctypes.CDLL) -> None:
    handle_ptr = ctypes.POINTER(ctypes.c_void_p)
    library.nemo_speech_asr_recognition_options_default.argtypes = []
    library.nemo_speech_asr_recognition_options_default.restype = RecognitionOptions
    library.nemo_speech_asr_create.argtypes = [
        ctypes.POINTER(RecognizerConfig),
        handle_ptr,
    ]
    library.nemo_speech_asr_create.restype = ctypes.c_int
    library.nemo_speech_asr_recognize_f32.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(RecognitionOptions),
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32, handle_ptr,
    ]
    library.nemo_speech_asr_recognize_f32.restype = ctypes.c_int
    library.nemo_speech_asr_destroy.argtypes = [ctypes.c_void_p]
    library.nemo_speech_asr_streaming_recognize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(RecognitionOptions),
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
    library.nemo_speech_asr_result_word_count.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.nemo_speech_asr_result_word_count.restype = ctypes.c_size_t
    for name, result_type in (
        ("text", ctypes.c_char_p),
        ("start_time", ctypes.c_int32),
        ("end_time", ctypes.c_int32),
        ("speaker_tag", ctypes.c_int32),
    ):
        function = getattr(library, f"nemo_speech_asr_result_word_{name}")
        function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
        function.restype = result_type
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


def check(status: int, operation: str) -> None:
    if status != 0:
        raise NativeRuntimeError(f"native ASR operation failed: {operation} ({status})")


def load_library() -> ctypes.CDLL:
    library = ctypes.CDLL(os.getenv("NEMO_SPEECH_LIBRARY", "/opt/nemo-speech/lib/libnemo_speech_asr_c.so"))
    configure_library(library)
    if library.nemo_speech_asr_version() != f"nemo-speech-asr {NEMO_SPEECH_VERSION}".encode():
        raise RuntimeError("NeMo-Speech.cpp runtime version does not match the image profile")
    return library
