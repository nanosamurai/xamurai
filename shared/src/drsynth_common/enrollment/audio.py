"""Decode uploaded enrollment WAVs without passing them to native audio parsers."""
import io
import wave

import numpy as np

MAX_SAMPLE_BYTES = 32 * 1024 * 1024
MAX_SAMPLE_SECONDS = 60


def decode_enrollment_pcm(sample_bytes: bytes) -> tuple[np.ndarray, int]:
    """Return mono float32 samples and rate from an uncompressed PCM WAV.

    The manifest backends contain user uploads. Compressed WAV, floating-point
    WAV and other audio formats are rejected before any native decoder runs.
    Python's WAV reader also supports the PCM form of WAVE_FORMAT_EXTENSIBLE.
    """
    try:
        if len(sample_bytes) > MAX_SAMPLE_BYTES:
            raise ValueError("Enrollment sample exceeds byte limit")
        with wave.open(io.BytesIO(sample_bytes), "rb") as source:
            channels, width, rate, frames, compression, _ = source.getparams()
            if (compression != "NONE" or width not in (1, 2, 3, 4)
                    or channels not in (1, 2) or not 8000 <= rate <= 192000
                    or not 0 < frames <= rate * MAX_SAMPLE_SECONDS):
                raise ValueError("Unsupported PCM parameters")
            # Read one extra frame to detect an incomplete trailing frame too.
            raw = source.readframes(frames + 1)
            if len(raw) != frames * channels * width:
                raise ValueError("Truncated or incomplete PCM frames")

        if width == 1:
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
        elif width == 3:
            octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            values = (octets[:, 0].astype(np.int32)
                      | octets[:, 1].astype(np.int32) << 8
                      | octets[:, 2].astype(np.int32) << 16)
            values = (values ^ 0x800000) - 0x800000
            samples = values.astype(np.float32) / 8388608
        else:
            samples = np.frombuffer(raw, dtype=f"<i{width}").astype(np.float32) / (1 << (width * 8 - 1))
        return samples.reshape(-1, channels).mean(axis=1, dtype=np.float32), rate
    except (wave.Error, EOFError, ValueError, OverflowError) as error:
        raise ValueError("Enrollment samples must be valid uncompressed PCM WAV audio "
                         "(mono/stereo, 8–192 kHz, at most 60 seconds and 32 MiB)") from error
