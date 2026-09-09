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
        }
    finally:
        session.close()
