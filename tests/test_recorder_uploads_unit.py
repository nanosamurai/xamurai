"""Unit tests for converting uploaded media into session recordings."""
import io
import math
import os
import shutil
import struct
import uuid
import wave

import pytest

from proto_gen import stream_pb2 as pb
from recorder_worker import recorder_worker as rw
from recorder_worker import uploads

BUCKET = "recordings-bucket"
needs_ffmpeg = pytest.mark.skipif(shutil.which(uploads.FFMPEG_BIN) is None, reason="ffmpeg not installed")


def _event(session_id=None, tenant="tenant-a", key=None, bucket=BUCKET):
    session_id = session_id or str(uuid.uuid4())
    key = key or f"recordings/{tenant}/uploads/{session_id}/source"
    return pb.RecordingFinished(session_id=session_id, tenant_id=tenant, lang="en",
                                recording_url=f"s3://{bucket}/{key}")


def _wav_bytes(seconds=1.0, rate=44100, channels=2):
    frames = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        samples = (int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(frames))
        w.writeframes(b"".join(struct.pack("<h", s) * channels for s in samples))
    return buf.getvalue()


class FakeMsg:
    def __init__(self, event, headers=None):
        self._value = event.SerializeToString() if isinstance(event, pb.RecordingFinished) else event
        self._headers = headers if headers is not None else [
            ("x-outputs", b"final"), ("x-final-tracks", b"whisperx,qwen"), ("x-store-recording", b"true")]

    def value(self):
        return self._value

    def headers(self):
        return self._headers


class FakeClientError(Exception):
    """Shape of botocore ClientError without needing boto3 in unit CI."""

    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.deleted = []

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def download_file(self, bucket, key, dst):
        with open(dst, "wb") as f:
            f.write(self.objects[(bucket, key)])

    def delete_object(self, Bucket, Key):
        self.deleted.append((Bucket, Key))
        self.objects.pop((Bucket, Key), None)


class FakeProducer:
    def __init__(self):
        self.messages = []

    def produce(self, **kwargs):
        self.messages.append(kwargs)

    def poll(self, _timeout):
        return 0

    def flush(self, _timeout):
        return 0


@pytest.mark.parametrize("mutate,reason", [
    (lambda e: setattr(e, "session_id", "../x"), "invalid-session-id"),
    (lambda e: setattr(e, "tenant_id", ""), "invalid-tenant-id"),
    (lambda e: setattr(e, "recording_url", e.recording_url.replace(BUCKET, "other")), "foreign-source"),
    (lambda e: setattr(e, "recording_url", e.recording_url.replace("/source", "/audio.wav")), "foreign-source"),
    (lambda e: setattr(e, "tenant_id", "tenant-b"), "foreign-source"),
])
def test_source_location_rejects_events_outside_the_session_upload(mutate, reason):
    event = _event()
    mutate(event)
    with pytest.raises(uploads.UploadRejected) as err:
        uploads.source_location(event, BUCKET)
    assert err.value.reason == reason


def test_source_location_accepts_the_session_upload_key():
    event = _event()
    assert uploads.source_location(event, BUCKET).endswith(f"/uploads/{event.session_id}/source")


@needs_ffmpeg
def test_transcode_produces_canonical_wav(tmp_path):
    src = tmp_path / "in"
    src.write_bytes(_wav_bytes(seconds=1.0, rate=44100, channels=2))
    dst = tmp_path / "out.wav"
    samples = uploads.transcode_to_wav(str(src), str(dst))
    with wave.open(str(dst), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() == samples
    assert abs(samples - 16000) < 200


@needs_ffmpeg
def test_transcode_rejects_malformed_media(tmp_path):
    src = tmp_path / "in"
    src.write_bytes(b"RIFF\x00\x00\x00\x00WAVE" + os.urandom(4096))
    with pytest.raises(uploads.UploadRejected) as err:
        uploads.transcode_to_wav(str(src), str(tmp_path / "out.wav"))
    assert err.value.reason in ("decode-failed", "no-audio")


@needs_ffmpeg
def test_transcode_rejects_playlists_that_reference_other_files(tmp_path):
    secret = tmp_path / "secret.wav"
    secret.write_bytes(_wav_bytes(seconds=1.0))
    src = tmp_path / "in"
    src.write_text(f"#EXTM3U\n#EXTINF:1,\n{secret}\n#EXT-X-ENDLIST\n")
    with pytest.raises(uploads.UploadRejected) as err:
        uploads.transcode_to_wav(str(src), str(tmp_path / "out.wav"))
    assert err.value.reason == "decode-failed"


@needs_ffmpeg
def test_transcode_enforces_the_duration_limit(tmp_path):
    src = tmp_path / "in"
    src.write_bytes(_wav_bytes(seconds=3.0, rate=16000, channels=1))
    with pytest.raises(uploads.UploadRejected) as err:
        uploads.transcode_to_wav(str(src), str(tmp_path / "out.wav"), max_duration_s=1.0)
    assert err.value.reason == "too-long"


def test_transcode_enforces_the_timeout(tmp_path):
    python = shutil.which("python") or shutil.which("python3")
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n")
    fake = tmp_path / ("ffmpeg.cmd" if os.name == "nt" else "ffmpeg")
    if os.name == "nt":
        fake.write_text(f'@"{python}" "{script}"\n')
    else:
        fake.write_text(f'#!/bin/sh\nexec "{python}" "{script}"\n')
        fake.chmod(0o755)
    src = tmp_path / "in"
    src.write_bytes(b"x")
    with pytest.raises(uploads.UploadRejected) as err:
        uploads.transcode_to_wav(str(src), str(tmp_path / "out.wav"), timeout_s=0.5, ffmpeg_bin=str(fake))
    assert err.value.reason in ("timeout", "decode-failed")


@needs_ffmpeg
def test_process_upload_publishes_recording_and_deletes_source(tmp_path, monkeypatch):
    monkeypatch.setattr(rw, "RECORDING_STORAGE_BACKEND", "local")
    monkeypatch.setattr(rw, "RECORDING_DIR", str(tmp_path))
    event = _event()
    key = uploads.source_location(event, BUCKET)
    s3 = FakeS3({(BUCKET, key): _wav_bytes(seconds=1.0, rate=8000, channels=1)})
    producer = FakeProducer()

    assert uploads.process_upload(FakeMsg(event), producer=producer, s3=s3, bucket=BUCKET) == "published"

    assert s3.deleted == [(BUCKET, key)]
    [published] = producer.messages
    assert published["topic"] == rw.TOPIC_RECORDING_FINISHED
    finished = pb.RecordingFinished.FromString(published["value"])
    assert (finished.session_id, finished.tenant_id, finished.lang) == (event.session_id, "tenant-a", "en")
    assert finished.sample_rate == 16000 and abs(finished.duration_s - 1.0) < 0.05
    headers = dict(published["headers"])
    assert headers["x-final-tracks"] == b"whisperx,qwen"
    assert headers["x-store-recording"] == b"true"
    with wave.open(finished.recording_url[len("file://"):], "rb") as w:
        assert (w.getnchannels(), w.getframerate()) == (1, 16000)


@needs_ffmpeg
def test_process_upload_rejects_undecodable_source_and_deletes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(rw, "RECORDING_STORAGE_BACKEND", "local")
    monkeypatch.setattr(rw, "RECORDING_DIR", str(tmp_path))
    event = _event()
    key = uploads.source_location(event, BUCKET)
    s3 = FakeS3({(BUCKET, key): b"OggS" + os.urandom(2048)})
    producer = FakeProducer()

    assert uploads.process_upload(FakeMsg(event), producer=producer, s3=s3, bucket=BUCKET) == "decode-failed"
    assert producer.messages == []
    assert s3.deleted == [(BUCKET, key)]
    assert list(tmp_path.glob("*.wav")) == []


def test_process_upload_rejects_missing_too_large_and_foreign_sources(monkeypatch):
    producer = FakeProducer()
    event = _event()
    assert uploads.process_upload(FakeMsg(event), producer=producer, s3=FakeS3({}), bucket=BUCKET) == "missing-source"

    key = uploads.source_location(event, BUCKET)
    monkeypatch.setattr(uploads, "MAX_SOURCE_BYTES", 10)
    s3 = FakeS3({(BUCKET, key): b"x" * 11})
    assert uploads.process_upload(FakeMsg(event), producer=producer, s3=s3, bucket=BUCKET) == "too-large"
    assert s3.deleted == [(BUCKET, key)]

    foreign = _event(bucket="other")
    s3 = FakeS3({})
    assert uploads.process_upload(FakeMsg(foreign), producer=producer, s3=s3, bucket=BUCKET) == "foreign-source"
    assert s3.deleted == [], "A foreign object is never touched"
    assert uploads.process_upload(FakeMsg(b"\xff\xff"), producer=producer, s3=s3, bucket=BUCKET) == "invalid-event"
    assert producer.messages == []


def test_process_upload_propagates_storage_outages_for_retry():
    class BrokenS3(FakeS3):
        def head_object(self, Bucket, Key):
            raise ConnectionError("storage unavailable")

    s3 = BrokenS3({})
    with pytest.raises(ConnectionError):
        uploads.process_upload(FakeMsg(_event()), producer=FakeProducer(), s3=s3, bucket=BUCKET)
    assert s3.deleted == [], "The source stays for the retry"
