from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import nemo_speech_native as nemo


MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_FILENAME = "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"
MODEL_REVISION = "1c8deaecc64b91f034d73e08dd8b64625eb3395d"
MODEL_DIGEST = "sha256:a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae"
DEFAULT_ENDPOINTING_SILENCE_MS = 2000
MAX_UTTERANCE_SECONDS = 30.0


@dataclass(frozen=True)
class SpeakerWord:
    text: str
    start_s: float
    end_s: float
    speaker: int


@dataclass(frozen=True)
class TranscriptUpdate:
    text: str
    final: bool
    audio_processed_s: float
    language: str
    words: tuple[SpeakerWord, ...] = ()


def diarization_from_env() -> bool:
    value = os.getenv("NEMOTRON_DIARIZATION", "false").strip().lower()
    if value not in ("true", "false"):
        raise ValueError("NEMOTRON_DIARIZATION must be true or false")
    return value == "true"


class NativeSession:
    """One cache-aware native stream; calls must stay on one worker thread."""

    def __init__(self, library: ctypes.CDLL, recognizer: ctypes.c_void_p, request_id: str,
                 language: Optional[str], *, diarization: bool = False,
                 endpointing_silence_ms: int = 0) -> None:
        self._library = library
        self._handle = ctypes.c_void_p()
        self._closed = False
        self._request_id = request_id.encode("utf-8")
        self._language = language.encode("ascii") if language else None
        self._input_audio_s = 0.0
        self._last_final_audio_s = 0.0
        self._diarization = diarization
        options = library.nemo_speech_asr_recognition_options_default()
        options.request_id = self._request_id
        options.language_code = self._language
        options.interim_results = True
        options.stop_history_eou_ms = endpointing_silence_ms
        # The stable C ABI intentionally defaults every optional request flag
        # to false. Nemotron 3.5 is self-punctuating, so this gate preserves the
        # casing and punctuation emitted by the model without a PnC sidecar.
        options.enable_automatic_punctuation = True
        options.enable_word_time_offsets = diarization
        options.enable_speaker_diarization = diarization
        nemo.check(
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
        nemo.check(
            self._library.nemo_speech_asr_stream_push_f32(
                self._handle, pointer, samples.size, sample_rate
            ),
            "push audio",
        )
        self._input_audio_s += samples.size / sample_rate
        updates = self._drain()
        if self._input_audio_s - self._last_final_audio_s >= MAX_UTTERANCE_SECONDS:
            nemo.check(
                self._library.nemo_speech_asr_stream_force_endpoint(self._handle),
                "force endpoint",
            )
            updates += self._drain()
        return updates

    def finish(self) -> tuple[TranscriptUpdate, ...]:
        nemo.check(self._library.nemo_speech_asr_stream_finish(self._handle), "finish stream")
        return self._drain()

    def _drain(self) -> tuple[TranscriptUpdate, ...]:
        updates: list[TranscriptUpdate] = []
        while True:
            result = ctypes.c_void_p()
            nemo.check(
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
                final = bool(self._library.nemo_speech_asr_result_is_final(result))
                words = ()
                if self._diarization and final and alternatives:
                    words = tuple(
                        SpeakerWord(
                            text=(self._library.nemo_speech_asr_result_word_text(result, 0, i)
                                  or b"").decode("utf-8", errors="replace"),
                            start_s=self._library.nemo_speech_asr_result_word_start_time(result, 0, i) / 1000.0,
                            end_s=self._library.nemo_speech_asr_result_word_end_time(result, 0, i) / 1000.0,
                            speaker=int(self._library.nemo_speech_asr_result_word_speaker_tag(result, 0, i)),
                        )
                        for i in range(self._library.nemo_speech_asr_result_word_count(result, 0))
                    )
                update = TranscriptUpdate(
                    text=transcript.decode("utf-8", errors="replace") if transcript else "",
                    final=final,
                    audio_processed_s=float(
                        self._library.nemo_speech_asr_result_audio_processed(result)
                    ),
                    language=language,
                    words=words,
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

    runtime = f"nemo-speech-cpp=={nemo.NEMO_SPEECH_VERSION}"

    def __init__(self, maximum_sessions: int) -> None:
        self.diarization = diarization_from_env()
        self.endpointing_silence_ms = nemo.bounded_int(
            "NEMOTRON_ENDPOINTING_SILENCE_MS", DEFAULT_ENDPOINTING_SILENCE_MS,
            1, int(MAX_UTTERANCE_SECONDS * 1000),
        )
        self._library = nemo.load_library()

        model_path_bytes = nemo.verified_model_path(MODEL_ID, MODEL_FILENAME, MODEL_REVISION, MODEL_DIGEST).encode("utf-8")
        backend = nemo.BackendConfig(
            size=ctypes.sizeof(nemo.BackendConfig),
            gpu=nemo.bounded_int("NEMOTRON_GPU", 0, -1, 15),
        )
        model = nemo.ModelConfig(
            size=ctypes.sizeof(nemo.ModelConfig),
            path=model_path_bytes,
            name=b"nemotron-3.5-asr-streaming-0.6b",
        )
        streaming = nemo.StreamingConfig(
            size=ctypes.sizeof(nemo.StreamingConfig),
            chunk_size=0.16,
            ctc_left_padding=1.92,
            ctc_right_padding=1.92,
            rnnt_right_context=nemo.bounded_int("NEMOTRON_RNNT_RIGHT_CONTEXT", 1, -1, 8),
        )
        batching = nemo.BatchingConfig(
            size=ctypes.sizeof(nemo.BatchingConfig),
            enable=maximum_sessions > 1,
            max_batch_size=maximum_sessions,
            max_queue_delay_us=2_000,
            max_queue_depth=max(4, maximum_sessions * 2),
            ingress_cohort_delay_us=1_000,
            state_arena_slots=maximum_sessions,
        )
        endpointing = nemo.EndpointingConfig(
            size=ctypes.sizeof(nemo.EndpointingConfig),
            enable=True,
            vad_based=True,
            stop_history_eou_ms=self.endpointing_silence_ms,
        )
        vad = nemo.VadConfig(
            size=ctypes.sizeof(nemo.VadConfig),
            model_path=b"/opt/nemo-speech/models/silero-v6.2.0.gguf",
            enable_masking=False,
            onset=0.5,
            offset=0.3,
        )
        diar = None
        if self.diarization:
            diar_path = nemo.verified_model_path(
                nemo.SORTFORMER_MODEL_ID, nemo.SORTFORMER_MODEL_FILENAME,
                nemo.SORTFORMER_MODEL_REVISION, nemo.SORTFORMER_MODEL_DIGEST,
            ).encode("utf-8")
            # Retain the pinned runtime's streaming geometry. Zero is a valid
            # left context, so only that field needs the default sentinel.
            diar = nemo.DiarizationConfig(
                size=ctypes.sizeof(nemo.DiarizationConfig),
                model_path=diar_path,
                left_context_frames=-1,
            )
        config = nemo.RecognizerConfig(
            size=ctypes.sizeof(nemo.RecognizerConfig),
            backend=ctypes.pointer(backend),
            model=ctypes.pointer(model),
            streaming=ctypes.pointer(streaming),
            decoder=None,
            vad=ctypes.pointer(vad),
            endpointing=ctypes.pointer(endpointing),
            postproc=None,
            diar=ctypes.pointer(diar) if diar is not None else None,
            batching=ctypes.pointer(batching),
        )
        self._recognizer = ctypes.c_void_p()
        nemo.check(
            self._library.nemo_speech_asr_create(
                ctypes.byref(config), ctypes.byref(self._recognizer)
            ),
            "create recognizer",
        )

    def open(self, session_id: str, language: Optional[str], *, endpointing_silence_ms: int = 0) -> NativeSession:
        return NativeSession(
            self._library, self._recognizer, session_id, language,
            diarization=self.diarization,
            endpointing_silence_ms=endpointing_silence_ms,
        )

    def close(self) -> None:
        recognizer = getattr(self, "_recognizer", None)
        if recognizer is not None and recognizer.value:
            self._library.nemo_speech_asr_destroy(recognizer)
            recognizer.value = None

    def __del__(self) -> None:
        self.close()
