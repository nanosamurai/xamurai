"""Transcribe diarized audio crops in bounded vLLM batches."""
import logging

import numpy as np
import soundfile as sf

from xamurai_serving.qwen import (
    MODEL_ID, QWEN_LANGUAGE_NAMES, SAMPLE_RATE, PyannoteDiarizer, bounded_int, load_model,
)


class Qwen:
    def __init__(self):
        self.batch_size = bounded_int("QWEN_BATCH_SIZE", 4, 1, 32)
        self.chunk_samples = SAMPLE_RATE * bounded_int("QWEN_BATCH_CHUNK_SECONDS", 30, 1, 30)
        self.model = load_model(
            batch_size=self.batch_size,
            kv_cache_mib=bounded_int("QWEN_KV_CACHE_MIB", 1024, 512, 16384),
        )
        self.diarizer = PyannoteDiarizer()

    def __call__(self, path, *, tenant=None, lang=None):
        """Return turn-timed text; speaker identity is local to this recording/window."""
        with sf.SoundFile(path) as audio:
            if audio.channels != 1 or audio.samplerate != SAMPLE_RATE:
                raise ValueError("Qwen expects mono 16 kHz audio")
            samples = audio.read(dtype="float32")
        if not np.isfinite(samples).all():
            raise ValueError("Recording contains nonfinite samples")
        if not samples.size or not np.any(samples):
            return "", []
        pcm16 = np.clip(samples * 32768, -32768, 32767).astype(np.int16)
        turns = self.diarizer.diarize(pcm16)
        crops = []
        previous_end = 0
        for turn in turns:
            # Own each sample once, including overlapping pyannote turns.
            start = max(previous_end, round(turn.start_s * SAMPLE_RATE), 0)
            end = min(samples.size, round(turn.end_s * SAMPLE_RATE))
            for offset in range(start, end, self.chunk_samples):
                crops.append((offset, min(offset + self.chunk_samples, end), turn.speaker))
            previous_end = max(previous_end, end)

        segments = []
        # A language hint only selects a supported language, never model artifacts.
        language = QWEN_LANGUAGE_NAMES.get((lang or "").lower())
        for offset in range(0, len(crops), self.batch_size):
            batch = crops[offset:offset + self.batch_size]
            results = self.model.transcribe(
                audio=[(samples[start:end], SAMPLE_RATE) for start, end, _ in batch],
                language=language,
            )
            if len(results) != len(batch):
                raise RuntimeError("Qwen returned an unexpected batch result count")
            for (start, end, speaker), result in zip(batch, results):
                text = result.text.strip()
                if text:
                    segments.append(dict(start_s=start / SAMPLE_RATE, end_s=end / SAMPLE_RATE,
                                         text=text, speaker=speaker))
        logging.getLogger(__name__).info(
            "Qwen completed duration_s=%.2f crops=%d batch_size=%d segments=%d",
            samples.size / SAMPLE_RATE, len(crops), self.batch_size, len(segments),
        )
        return " ".join(segment["text"] for segment in segments), segments
