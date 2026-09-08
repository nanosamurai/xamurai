import ctypes

from nemotron_rtservice.native import NativeSession, _RecognitionOptions


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
