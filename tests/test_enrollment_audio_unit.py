"""Exercise the uploaded-audio boundary with bytes, without native decoding."""
import io
import struct
import wave

import numpy as np
import pytest

from drsynth_common.enrollment import audio
from drsynth_common.enrollment.audio import decode_enrollment_pcm


def _wav(raw, width=2, channels=1, rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(raw)
    return output.getvalue()


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("channels", [1, 2])
def test_decode_integer_pcm_with_correct_signedness(width, channels):
    scale = 1 << (8 * width - 1)
    values = [-scale, -scale // 2, 0, scale // 2, scale - 1]
    raw = b"".join((value + (128 if width == 1 else 0)).to_bytes(
        width, "little", signed=width != 1) * channels for value in values)
    samples, rate = decode_enrollment_pcm(_wav(raw, width, channels, 22050))
    assert rate == 22050 and samples.dtype == np.float32
    np.testing.assert_allclose(samples, np.array(values, dtype=np.float32) / scale)


def test_mix_stereo_to_mono():
    samples, _ = decode_enrollment_pcm(_wav(struct.pack("<hhhh", -16384, 16384, 8192, 24576), channels=2))
    np.testing.assert_array_equal(samples, [0.0, 0.5])


def _riff(fmt, data):
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks


def test_decode_extensible_integer_pcm():
    pcm_guid = bytes.fromhex("0100000000001000800000aa00389b71")
    fmt = struct.pack("<HHIIHHHHI", 0xFFFE, 1, 16000, 32000, 2, 16, 22, 16, 1) + pcm_guid
    samples, rate = decode_enrollment_pcm(_riff(fmt, struct.pack("<hh", -16384, 16384)))
    assert rate == 16000
    np.testing.assert_array_equal(samples, [-0.5, 0.5])


@pytest.mark.parametrize("format_id", [0x11, 3, 6, 7])
def test_reject_compressed_or_float_wav_before_decoding(format_id):
    # Valid container structure, unsupported codec: IMA ADPCM, float, A-law, mu-law.
    fmt = struct.pack("<HHIIHH", format_id, 1, 16000, 32000, 2, 16)
    with pytest.raises(ValueError, match="uncompressed PCM WAV"):
        decode_enrollment_pcm(_riff(fmt, b"\x00" * 8))


@pytest.mark.parametrize("data", [b"", b"not a WAV", _wav(b""), _wav(b"\x00\x00")[:-1], _wav(b"\x00\x00\x01")])
def test_reject_empty_malformed_and_incomplete_pcm(data):
    with pytest.raises(ValueError, match="uncompressed PCM WAV"):
        decode_enrollment_pcm(data)


@pytest.mark.parametrize("rate", [1, 7999, 192001])
def test_reject_sample_rates_that_amplify_resampling_or_exceed_supported_range(rate):
    with pytest.raises(ValueError, match="uncompressed PCM WAV"):
        decode_enrollment_pcm(_wav(b"\x00\x00", rate=rate))


def test_reject_excess_duration_before_reading_or_allocating_frames(monkeypatch):
    raw = _wav(b"\x00\x00", rate=8000)
    declared = bytearray(raw)
    struct.pack_into("<I", declared, 40, 8000 * 61 * 2)
    def unexpected_read(*args):
        raise AssertionError("Oversized audio must be rejected before reading samples")
    monkeypatch.setattr(wave.Wave_read, "readframes", unexpected_read)
    with pytest.raises(ValueError, match="uncompressed PCM WAV"):
        decode_enrollment_pcm(bytes(declared))


def test_reject_byte_limit_before_parsing(monkeypatch):
    monkeypatch.setattr(audio, "MAX_SAMPLE_BYTES", 10)
    with pytest.raises(ValueError, match="uncompressed PCM WAV"):
        decode_enrollment_pcm(b"x" * 11)


def test_reject_multichannel_upload():
    with pytest.raises(ValueError, match="uncompressed PCM WAV"):
        decode_enrollment_pcm(_wav(b"\x00" * 6, channels=3))
