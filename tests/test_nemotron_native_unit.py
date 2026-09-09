import ctypes

from nemotron_rtservice.native import NativeSession, SpeakerWord, _RecognitionOptions


class _FakeLibrary:
    def __init__(self):
        self.options = None

    def nemo_speech_asr_recognition_options_default(self):
        options = _RecognitionOptions()
        options.size = ctypes.sizeof(_RecognitionOptions)
        return options

    def nemo_speech_asr_streaming_recognize(self, _recognizer, options, handle):
        captured = ctypes.cast(options, ctypes.POINTER(_RecognitionOptions)).contents
        self.options = {
            "interim_results": bool(captured.interim_results),
            "enable_automatic_punctuation": bool(captured.enable_automatic_punctuation),
            "enable_word_time_offsets": bool(captured.enable_word_time_offsets),
            "enable_speaker_diarization": bool(captured.enable_speaker_diarization),
        }
        ctypes.cast(handle, ctypes.POINTER(ctypes.c_void_p)).contents.value = 1
        return 0

    def nemo_speech_asr_stream_close(self, _handle):
        return None


def test_native_session_requests_interim_and_self_punctuated_text():
    library = _FakeLibrary()

    session = NativeSession(library, ctypes.c_void_p(7), "session", "en-US")
    try:
        assert library.options == {
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
