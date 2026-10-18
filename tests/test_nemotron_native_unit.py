import ctypes
from unittest.mock import Mock

import pytest

from nemotron_rtservice import native
from nemotron_rtservice.native import NativeSession, SpeakerWord
from nemo_speech_native import RecognitionOptions


def test_unpatched_native_runtime_is_rejected(monkeypatch):
    library = Mock()
    library.nemo_speech_asr_version.return_value = b"nemo-speech-asr 0.1.0"
    monkeypatch.setattr(native.ctypes, "CDLL", lambda _path: library)
    with pytest.raises(RuntimeError, match="runtime version"):
        native.nemo.load_library()


@pytest.mark.parametrize("diarization", [False, True])
@pytest.mark.parametrize("configured, expected", [
    (None, 2000), ("800", 800), ("3000", 3000), ("1", 1), ("30000", 30000),
])
def test_backend_passes_endpoint_silence_to_native_config(monkeypatch, configured, expected, diarization):
    if configured is None:
        monkeypatch.delenv("NEMOTRON_ENDPOINTING_SILENCE_MS", raising=False)
    else:
        monkeypatch.setenv("NEMOTRON_ENDPOINTING_SILENCE_MS", configured)
    monkeypatch.setenv("NEMOTRON_DIARIZATION", str(diarization).lower())
    library = Mock()
    library.nemo_speech_asr_version.return_value = f"nemo-speech-asr {native.nemo.NEMO_SPEECH_VERSION}".encode()
    captured = {}

    def create(config, handle):
        config = ctypes.cast(config, ctypes.POINTER(native.nemo.RecognizerConfig)).contents
        endpointing = config.endpointing.contents
        captured.update(
            silence_ms=endpointing.stop_history_eou_ms,
            soft_after_ms=endpointing.soft_after_ms,
            soft_silence_ms=endpointing.soft_silence_ms,
            max_utterance_ms=endpointing.max_utterance_ms,
            enabled=bool(endpointing.enable),
            vad_based=bool(endpointing.vad_based),
            vad=bool(config.vad),
            vad_path=config.vad.contents.model_path,
            masking=bool(config.vad.contents.enable_masking),
            onset=config.vad.contents.onset,
            offset=config.vad.contents.offset,
            diarization=bool(config.diar),
        )
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p)).contents.value = 7
        return 0

    library.nemo_speech_asr_create.side_effect = create
    monkeypatch.setattr(native.ctypes, "CDLL", lambda _path: library)
    monkeypatch.setattr(native.nemo, "verified_model_path", lambda *args: "model.gguf")
    backend = native.NativeNemotronBackend(maximum_sessions=1)
    try:
        assert captured == {
            "silence_ms": expected,
            "soft_after_ms": 90000,
            "soft_silence_ms": 500,
            "max_utterance_ms": 120000,
            "enabled": True,
            "vad_based": True,
            "vad": True,
            "vad_path": b"/opt/nemo-speech/models/silero-v6.2.0.gguf",
            "masking": False,
            "onset": pytest.approx(0.5),
            "offset": pytest.approx(0.3),
            "diarization": diarization,
        }
    finally:
        backend.close()


@pytest.mark.parametrize("configured", ["", "abc", "2000.5", "0", "-1", "30001", "4294967296"])
def test_invalid_endpoint_silence_fails_before_loading_native_runtime(monkeypatch, configured):
    monkeypatch.setenv("NEMOTRON_ENDPOINTING_SILENCE_MS", configured)
    monkeypatch.setenv("NEMOTRON_DIARIZATION", "false")
    load_library = Mock()
    monkeypatch.setattr(native.ctypes, "CDLL", load_library)

    with pytest.raises(ValueError, match="NEMOTRON_ENDPOINTING_SILENCE_MS must be"):
        native.NativeNemotronBackend(maximum_sessions=1)

    load_library.assert_not_called()


@pytest.mark.parametrize("name,value", [
    ("NEMOTRON_MAX_UTTERANCE_SECONDS", "1"),
    ("NEMOTRON_MAX_UTTERANCE_SECONDS", "3601"),
    ("NEMOTRON_ENDPOINTING_SOFT_AFTER_SECONDS", "120"),
    ("NEMOTRON_ENDPOINTING_SOFT_AFTER_SECONDS", "0"),
    ("NEMOTRON_ENDPOINTING_SOFT_SILENCE_MS", "0"),
    ("NEMOTRON_ENDPOINTING_SOFT_SILENCE_MS", "700.5"),
])
def test_invalid_duration_policy_fails_before_loading_models(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    load = Mock()
    monkeypatch.setattr(native.nemo, "load_library", load)
    with pytest.raises(ValueError, match=name):
        native.NativeNemotronBackend(1)
    load.assert_not_called()


class _FakeLibrary:
    def __init__(self):
        self.options = None

    def nemo_speech_asr_recognition_options_default(self):
        options = RecognitionOptions()
        options.size = ctypes.sizeof(RecognitionOptions)
        return options

    def nemo_speech_asr_streaming_recognize(self, _recognizer, options, handle):
        captured = ctypes.cast(options, ctypes.POINTER(RecognitionOptions)).contents
        self.options = {
            "stop_history_eou_ms": captured.stop_history_eou_ms,
            "interim_results": bool(captured.interim_results),
            "enable_automatic_punctuation": bool(captured.enable_automatic_punctuation),
            "enable_word_time_offsets": bool(captured.enable_word_time_offsets),
            "enable_speaker_diarization": bool(captured.enable_speaker_diarization),
        }
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p)).contents.value = 1
        return 0

    def nemo_speech_asr_stream_close(self, _handle):
        return None


@pytest.mark.parametrize("silence_ms", [0, 800, 3000])
def test_native_session_requests_interim_and_self_punctuated_text(silence_ms):
    library = _FakeLibrary()

    session = NativeSession(library, ctypes.c_void_p(7), "session", "en-US", endpointing_silence_ms=silence_ms)
    try:
        assert library.options == {
            "stop_history_eou_ms": silence_ms,
            "interim_results": True,
            "enable_automatic_punctuation": True,
            "enable_word_time_offsets": False,
            "enable_speaker_diarization": False,
        }
    finally:
        session.close()


def test_native_session_requests_words_and_sortformer_together():
    library = _FakeLibrary()
    session = NativeSession(library, ctypes.c_void_p(7), 'session', 'en-US', diarization=True)
    try:
        assert library.options['enable_word_time_offsets'] is True
        assert library.options['enable_speaker_diarization'] is True
    finally:
        session.close()


def test_native_words_are_copied_and_milliseconds_converted_before_destroy():
    class ResultsLibrary(_FakeLibrary):
        def __init__(self):
            super().__init__()
            self.reads = 0
            self.destroyed = False

        def nemo_speech_asr_stream_next(self, handle, result):
            self.reads += 1
            ctypes.cast(result, ctypes.POINTER(ctypes.c_void_p)).contents.value = 42 if self.reads == 1 else None
            return 0

        def nemo_speech_asr_result_alternative_count(self, result): return 1
        def nemo_speech_asr_result_transcript(self, result, alt): return b'Hello.'
        def nemo_speech_asr_result_language_count(self, result, alt): return 0
        def nemo_speech_asr_result_is_final(self, result): return True
        def nemo_speech_asr_result_audio_processed(self, result): return 3.0
        def nemo_speech_asr_result_word_count(self, result, alt): return 1
        def nemo_speech_asr_result_word_text(self, result, alt, i): return b'Hello.'
        def nemo_speech_asr_result_word_start_time(self, result, alt, i): return 1200
        def nemo_speech_asr_result_word_end_time(self, result, alt, i): return 2300
        def nemo_speech_asr_result_word_speaker_tag(self, result, alt, i): return 4
        def nemo_speech_asr_result_destroy(self, result): self.destroyed = True

    library = ResultsLibrary()
    session = NativeSession(library, ctypes.c_void_p(7), 'session', 'en-US', diarization=True)
    try:
        result, = session._drain()
        assert library.destroyed
        assert result.words == (SpeakerWord('Hello.', 1.2, 2.3, 4),)
    finally:
        session.close()
