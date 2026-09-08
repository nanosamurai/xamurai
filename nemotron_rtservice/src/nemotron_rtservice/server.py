from __future__ import annotations

import logging
import os
import re
from concurrent import futures
from typing import Optional, Protocol

import grpc
import numpy as np

from nemotron_rtservice.native import (
    MODEL_DIGEST,
    MODEL_REVISION,
    NEMO_SPEECH_REVISION,
    SORTFORMER_MODEL_REVISION,
    NativeNemotronBackend,
    TranscriptUpdate,
)
from nemotron_rtservice.speakers import EMBEDDING_MODEL_REVISION, IDENTIFIER, enrollment_from_env, speaker_turns
from proto_gen import stream_pb2, stream_pb2_grpc
from xamurai_serving import SessionSlots, max_sessions_from_env, serving_instance_id


logger = logging.getLogger(__name__)

PROFILE_ID = "nemotron-3.5-asr-streaming-0.6b-nemo-speech-cpp-q8-r1"
DIARIZED_PROFILE_ID = "nemotron-3.5-asr-streaming-0.6b-sortformer-q8-r1"
ENROLLED_PROFILE_ID = "nemotron-3.5-asr-streaming-0.6b-sortformer-enrolled-q8-r1"
SAMPLE_RATE = 16_000
MAX_CHUNK_BYTES = 1_048_576
SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
LANGUAGE_LOCALES = {
    "ar": "ar-SA", "bg": "bg-BG", "cs": "cs-CZ", "da": "da-DK",
    "de": "de-DE", "el": "el-GR", "en": "en-US", "es": "es-ES",
    "et": "et-EE", "fi": "fi-FI", "fr": "fr-FR", "he": "he-IL",
    "hi": "hi-IN", "hr": "hr-HR", "hu": "hu-HU", "it": "it-IT",
    "ja": "ja-JP", "ko": "ko-KR", "lt": "lt-LT", "lv": "lv-LV",
    "nb": "nb-NO", "nl": "nl-NL", "nn": "nn-NO", "pl": "pl-PL",
    "pt": "pt-PT", "ro": "ro-RO", "ru": "ru-RU", "sk": "sk-SK",
    "sl": "sl-SI", "sv": "sv-SE", "th": "th-TH", "tr": "tr-TR",
    "uk": "uk-UA", "vi": "vi-VN", "zh": "zh-CN",
}
LOCALE_LANGUAGES = {locale.casefold(): language for language, locale in LANGUAGE_LOCALES.items()}


class _InvalidRequest(ValueError):
    """Validated public-stream error safe to return to an internal caller."""


class NemotronStream(Protocol):
    def push(self, pcm16_le: bytes, sample_rate: int) -> tuple[TranscriptUpdate, ...]: ...
    def finish(self) -> tuple[TranscriptUpdate, ...]: ...
    def close(self) -> None: ...


class NemotronBackend(Protocol):
    runtime: str

    def open(self, session_id: str, language: Optional[str]) -> NemotronStream: ...


def _public_language(language: str, fallback: str) -> str:
    value = str(language or "").strip()
    if not value:
        return fallback
    return LOCALE_LANGUAGES.get(value.casefold(), value.split("-", 1)[0].casefold())


class NemotronRealtimeServicer(stream_pb2_grpc.RealtimeASRServicer):
    """Expose one pinned cache-aware Nemotron streaming profile."""

    def __init__(self, backend: NemotronBackend, maximum_sessions: int, enrollment=None) -> None:
        self._backend = backend
        self._slots = SessionSlots(maximum_sessions)
        self._instance_id = serving_instance_id()
        self._diarization = bool(getattr(backend, "diarization", False))
        self._enrollment = enrollment
        self._profile_id = (ENROLLED_PROFILE_ID if enrollment is not None else DIARIZED_PROFILE_ID
                            ) if self._diarization else PROFILE_ID

    def GetCapabilities(self, request, context):
        return stream_pb2.RealtimeCapabilities(
            provider_profile_id=self._profile_id,
            windowed_realtime=False,
            native_streaming=True,
            batch=self._slots.maximum > 1,
            segment_timestamps=self._diarization,
            word_timestamps=False,
            language_detection=True,
            supported_languages=tuple(LANGUAGE_LOCALES),
            stateful=True,
            preferred_sample_rate=SAMPLE_RATE,
            maximum_audio_seconds=0,
            maximum_concurrent_sessions=self._slots.maximum,
            runtime=self._backend.runtime,
            model_revision=MODEL_REVISION,
            model_digest=MODEL_DIGEST,
            implementation_revision=(f"nemo-speech-cpp:{NEMO_SPEECH_REVISION}"
                                     + (f";sortformer:{SORTFORMER_MODEL_REVISION}" if self._diarization else "")
                                     + (f";wespeaker:{EMBEDDING_MODEL_REVISION}" if self._enrollment is not None else "")),
            speaker_labels=self._diarization,
        )

    def Stream(self, request_iterator, context):
        metadata = {
            str(key).lower(): str(value)
            for key, value in (context.invocation_metadata() or ())
        }
        opening_session_id = metadata.get("x-session-id", "").strip()
        if not SESSION_ID_PATTERN.fullmatch(opening_session_id):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "SESSION_ID_REQUIRED")
        if not self._slots.acquire():
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "REPLICA_FULL")

        native_stream: Optional[NemotronStream] = None
        total_samples = 0
        expected_sequence: Optional[int] = None
        language: Optional[str] = None
        last_partial = ""
        segment_start_s = 0.0
        tenant_id = None
        audio = bytearray()

        def events(update):
            nonlocal segment_start_s, last_partial
            text = update.text.strip()
            if not text or (not update.final and text == last_partial):
                return ()
            coarse = self._event(opening_session_id, update, total_samples, language or "", segment_start_s)
            result = (coarse,)
            if self._diarization and update.final:
                turns = speaker_turns(text, update.words, segment_start_s, coarse.end_s)
                if turns:
                    names = {}
                    if self._enrollment is not None and tenant_id:
                        offset = total_samples - len(audio) // 2
                        for speaker in {turn.speaker for turn in turns if turn.speaker}:
                            pieces = []
                            for turn in turns:
                                if turn.speaker != speaker:
                                    continue
                                begin = max(0, round(turn.start_s * SAMPLE_RATE) - offset)
                                end = min(len(audio) // 2, round(turn.end_s * SAMPLE_RATE) - offset)
                                if end > begin:
                                    pieces.append(np.frombuffer(bytes(audio[begin * 2:end * 2]), dtype="<i2"))
                            if pieces:
                                samples = np.concatenate(pieces).astype(np.float32) / 32768.0
                                names[speaker] = self._enrollment.identify(tenant_id, samples)
                    result = tuple(stream_pb2.AsrEvent(
                        session_id=opening_session_id, start_s=turn.start_s, end_s=turn.end_s,
                        text=turn.text.strip(), type=stream_pb2.FINAL, lang=coarse.lang,
                        speaker=names.get(turn.speaker) or (f"SPEAKER_{turn.speaker - 1:02d}" if turn.speaker else ""),
                        provider_profile_id=self._profile_id,
                    ) for turn in turns)
            if update.final:
                # Advance by the native endpoint, independently of the last
                # word's time, so silence is not replayed as the next partial.
                segment_start_s = coarse.end_s
            last_partial = "" if update.final else text
            return result

        try:
            yield stream_pb2.AsrEvent(
                session_id=opening_session_id,
                type=stream_pb2.SESSION_ACCEPTED,
                provider_profile_id=self._profile_id,
                serving_instance_id=self._instance_id,
            )
            for chunk in request_iterator:
                error = self._validate_chunk(
                    chunk,
                    session_id=opening_session_id,
                    expected_sequence=expected_sequence,
                    language=language,
                )
                if error:
                    raise _InvalidRequest(error)
                if tenant_id is None:
                    tenant_id = chunk.tenant_id
                    if tenant_id and not IDENTIFIER.fullmatch(tenant_id):
                        raise _InvalidRequest("invalid tenant ID")
                    if metadata.get("x-tenant-id", tenant_id) != tenant_id:
                        raise _InvalidRequest("tenant ID does not match stream metadata")
                elif chunk.tenant_id != tenant_id:
                    raise _InvalidRequest("tenant ID changed within realtime stream")
                if native_stream is None:
                    language = chunk.lang
                    native_stream = self._backend.open(
                        opening_session_id,
                        LANGUAGE_LOCALES.get(language) if language else None,
                    )
                expected_sequence = chunk.seq + 1
                total_samples += len(chunk.pcm16_le) // 2
                if self._enrollment is not None:
                    audio.extend(chunk.pcm16_le)
                    # One 30 s native utterance plus the largest accepted
                    # ingress chunk; no session-lifetime audio accumulation.
                    del audio[:max(0, len(audio) - 64 * SAMPLE_RATE * 2)]
                for update in native_stream.push(chunk.pcm16_le, chunk.sample_rate):
                    yield from events(update)

            if native_stream is not None and context.is_active():
                for update in native_stream.finish():
                    yield from events(update)
        except _InvalidRequest as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except grpc.RpcError as exc:
            if context.is_active() and exc.code() != grpc.StatusCode.CANCELLED:
                logger.warning("Nemotron realtime transport ended code=%s", exc.code().name)
        except Exception as exc:
            if context.is_active():
                logger.error("Nemotron realtime stream failed error_type=%s", type(exc).__name__)
                context.abort(grpc.StatusCode.INTERNAL, "Nemotron realtime inference failed")
        finally:
            if native_stream is not None:
                try:
                    native_stream.close()
                except Exception as exc:
                    logger.warning("Nemotron stream close failed error_type=%s", type(exc).__name__)
            self._slots.release()

    @staticmethod
    def _validate_chunk(chunk, *, session_id: str, expected_sequence: Optional[int],
                        language: Optional[str]) -> Optional[str]:
        if chunk.session_id != session_id:
            return "session ID changed within realtime stream"
        if expected_sequence is not None and chunk.seq != expected_sequence:
            return "audio chunk sequence is not contiguous"
        if chunk.sample_rate != SAMPLE_RATE:
            return "Nemotron realtime service accepts PCM16 mono at 16000 Hz"
        if len(chunk.pcm16_le) % 2:
            return "PCM16 payload length must be even"
        if len(chunk.pcm16_le) > MAX_CHUNK_BYTES:
            return "PCM16 payload exceeds the realtime chunk limit"
        if language is None and chunk.lang and chunk.lang not in LANGUAGE_LOCALES:
            return "Nemotron realtime service does not support the requested language code"
        if language is not None and chunk.lang != language:
            return "language changed within realtime stream"
        return None

    def _event(self, session_id: str, update: TranscriptUpdate, total_samples: int,
               fallback_language: str, segment_start_s: float):
        total_seconds = total_samples / SAMPLE_RATE
        processed = update.audio_processed_s
        end_seconds = min(total_seconds, max(0.0, processed)) if processed > 0 else total_seconds
        return stream_pb2.AsrEvent(
            session_id=session_id,
            start_s=min(end_seconds, max(0.0, segment_start_s)),
            end_s=end_seconds,
            text=update.text.strip(),
            type=stream_pb2.FINAL if update.final else stream_pb2.PARTIAL,
            lang=_public_language(update.language, fallback_language),
            provider_profile_id=self._profile_id,
        )


def create_server(backend: Optional[NemotronBackend] = None, *, port: Optional[int] = None,
                  maximum_sessions: Optional[int] = None, enrollment=None) -> grpc.Server:
    """Create the Nemotron server without starting its worker threads."""
    maximum = maximum_sessions if maximum_sessions is not None else max_sessions_from_env()
    if maximum < 1:
        raise ValueError("maximum_sessions must be positive")
    runtime = backend or NativeNemotronBackend(maximum)
    if backend is None and runtime.diarization:
        enrollment = enrollment_from_env()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max(4, maximum + 2)))
    stream_pb2_grpc.add_RealtimeASRServicer_to_server(
        NemotronRealtimeServicer(runtime, maximum, enrollment), server
    )
    bind_addr = os.getenv("NEMOTRON_RTSERVICE_BIND_ADDR", "[::]").strip() or "[::]"
    selected_port = port if port is not None else int(os.getenv("NEMOTRON_RTSERVICE_PORT", "50052"))
    server.bound_port = server.add_insecure_port(f"{bind_addr}:{selected_port}")  # type: ignore[attr-defined]
    return server


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info(
        "Loading Nemotron realtime profile=%s model_revision=%s model_digest=%s",
        PROFILE_ID,
        MODEL_REVISION,
        MODEL_DIGEST,
    )
    server = create_server()
    server.start()
    logger.info("Nemotron realtime service ready port=%d", server.bound_port)  # type: ignore[attr-defined]
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)


if __name__ == "__main__":
    main()
