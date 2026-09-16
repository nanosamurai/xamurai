"""Offline Parakeet TDT with embedded Sortformer, using the pinned native C ABI."""
import ctypes
import logging

import numpy as np
import soundfile as sf

from nemotron_rtservice import native

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
MODEL_FILENAME = "parakeet-tdt-0.6b-v3.q8_0.gguf"
MODEL_DIGEST = "sha256:e3880d0aaaaf2c308ea2c35016b2b895c423eb3fda924c1b463d1c19b7f4d32e"


class Parakeet:
    """Keep one ASR/diarization model pair warm; recognize recordings sequentially."""

    def __init__(self):
        self.library = ctypes.CDLL("/opt/nemo-speech/lib/libnemo_speech_asr_c.so")
        native._configure_library(self.library)
        if self.library.nemo_speech_asr_version() != b"nemo-speech-asr 0.1.0":
            raise RuntimeError("Parakeet requires the pinned NeMo-Speech.cpp runtime")
        self.library.nemo_speech_asr_recognize_f32.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(native._RecognitionOptions),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.nemo_speech_asr_recognize_f32.restype = ctypes.c_int
        model_path = native._verified_model_path(
            MODEL_ID, MODEL_FILENAME, MODEL_REVISION, MODEL_DIGEST,
        ).encode()
        diar_path = native._verified_model_path(
            native.SORTFORMER_MODEL_ID, native.SORTFORMER_MODEL_FILENAME,
            native.SORTFORMER_MODEL_REVISION, native.SORTFORMER_MODEL_DIGEST,
        ).encode()
        backend = native._BackendConfig(
            size=ctypes.sizeof(native._BackendConfig),
            gpu=native._bounded_int("PARAKEET_GPU", 0, -1, 15),
        )
        model = native._ModelConfig(
            size=ctypes.sizeof(native._ModelConfig), path=model_path, name=MODEL_ID.encode(),
        )
        diar = native._DiarizationConfig(
            size=ctypes.sizeof(native._DiarizationConfig), model_path=diar_path,
            left_context_frames=-1,
        )
        config = native._RecognizerConfig(
            size=ctypes.sizeof(native._RecognizerConfig), backend=ctypes.pointer(backend),
            model=ctypes.pointer(model), diar=ctypes.pointer(diar),
        )
        self.handle = ctypes.c_void_p()
        native._check(self.library.nemo_speech_asr_create(
            ctypes.byref(config), ctypes.byref(self.handle)), "create Parakeet recognizer")

    def __call__(self, path, *, tenant=None, lang=None):
        """Return text and timed anonymous speaker segments; each call has fresh speaker state."""
        with sf.SoundFile(path) as audio:
            if audio.channels != 1 or audio.samplerate != 16000:
                raise ValueError("Parakeet finalizer expects mono 16 kHz recordings")
            samples = audio.read(dtype="float32")
        if not samples.size:
            return "", []
        if not np.isfinite(samples).all():
            raise ValueError("Recording contains nonfinite samples")
        options = self.library.nemo_speech_asr_recognition_options_default()
        options.enable_word_time_offsets = True
        options.enable_automatic_punctuation = True
        options.enable_speaker_diarization = True
        # TDT detects language itself; tenant/lang never select artifacts or speaker identities.
        result = ctypes.c_void_p()
        try:
            native._check(self.library.nemo_speech_asr_recognize_f32(
                self.handle, ctypes.byref(options),
                samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), samples.size, 16000,
                ctypes.byref(result)), "recognize recording")
            text = (self.library.nemo_speech_asr_result_transcript(result, 0) or b"").decode()
            segments = []
            duration = samples.size / 16000
            previous_start = 0.0
            for i in range(self.library.nemo_speech_asr_result_word_count(result, 0)):
                word = {
                    "text": (self.library.nemo_speech_asr_result_word_text(result, 0, i) or b"").decode(),
                    "start_s": self.library.nemo_speech_asr_result_word_start_time(result, 0, i) / 1000,
                    "end_s": min(duration, self.library.nemo_speech_asr_result_word_end_time(result, 0, i) / 1000),
                }
                if not previous_start <= word["start_s"] <= word["end_s"] <= duration:
                    raise RuntimeError("Parakeet returned invalid word timing")
                previous_start = word["start_s"]
                slot = self.library.nemo_speech_asr_result_word_speaker_tag(result, 0, i)
                speaker = f"SPEAKER_{slot - 1:02d}" if 1 <= slot <= 4 else ""
                if (not segments or segments[-1]["speaker"] != speaker
                        or word["start_s"] - segments[-1]["end_s"] > 1.0):
                    segments.append(dict(start_s=word["start_s"], end_s=word["end_s"],
                                         text="", speaker=speaker, words=[]))
                segment = segments[-1]
                segment["words"].append(word)
                segment["end_s"] = max(segment["end_s"], word["end_s"])
            for segment in segments:
                segment["text"] = " ".join(w["text"] for w in segment["words"])
            logging.getLogger(__name__).info(
                "Parakeet completed duration_s=%.2f segments=%d", duration, len(segments))
            return text, segments
        finally:
            if result.value:
                self.library.nemo_speech_asr_result_destroy(result)

    def close(self):
        """Release the native recognizer after the worker exits."""
        if self.handle.value:
            self.library.nemo_speech_asr_destroy(self.handle)
            self.handle = ctypes.c_void_p()


if __name__ == "__main__":
    from finalizer_worker.finalizer_worker import main

    pipeline = Parakeet()
    try:
        main(pipeline, model=MODEL_ID)
    finally:
        pipeline.close()
