"""Turn uploaded media files into normal session recordings.

The BFF stores an uploaded file unchanged in the recordings bucket and
publishes a ``RecordingFinished`` message on ``recordings.uploaded``. Its
``recording_url`` points at that source object; its Kafka headers carry the
same stream controls as ``audio.raw`` (``x-final-tracks``,
``x-store-recording``).

This module downloads the source, decodes it with FFmpeg into the same 16 kHz
mono PCM16 WAV that live sessions produce, stores it with the normal recording
writer and publishes the normal ``recordings.finished`` event. Final-track
workers therefore need no change.

Security: the source is untrusted.
- The event must point at ``<bucket>/.../<tenant>/uploads/<session>/source`` in
  the configured bucket, so an event cannot make the worker read other objects.
- FFmpeg runs without a shell, reads only the local temporary copy (``file``
  protocol only, allow-listed demuxers, so playlists cannot fetch other files
  or URLs), is killed after a timeout and is stopped at the duration limit.
- The source object is deleted after processing, whether decoding succeeded or
  not. Only a failure of storage or Kafka is retried.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import wave
from typing import Optional

from confluent_kafka import TopicPartition

from proto_gen import stream_pb2
from drsynth_common.stream_controls import parse_stream_controls_from_kafka_headers

from recorder_worker import recorder_worker as rw

logger = logging.getLogger("recorder_worker.uploads")


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y")


TOPIC_RECORDING_UPLOADED = os.getenv("KAFKA_TOPIC_RECORDING_UPLOADED", "recordings.uploaded")
UPLOADS_ENABLED = _env_bool("RECORDER_UPLOADS_ENABLED", "true")
FFMPEG_BIN = os.getenv("RECORDER_FFMPEG_BIN", "ffmpeg")
MAX_SOURCE_BYTES = int(os.getenv("RECORDER_UPLOAD_MAX_BYTES", str(512 * 1024 * 1024)))
MAX_DURATION_S = float(os.getenv("RECORDER_UPLOAD_MAX_DURATION_S", "14400"))
TRANSCODE_TIMEOUT_S = float(os.getenv("RECORDER_UPLOAD_TIMEOUT_S", "1800"))
RETRY_BACKOFF_S = 5.0

# FFmpeg demuxer names that match the containers the BFF accepts.
ALLOWED_DEMUXERS = "wav,mp3,flac,ogg,mov,matroska,aac,amr,aiff,asf,caf,avi"

_READ_CHUNK = 64 * 1024


class UploadRejected(Exception):
    """A permanent problem with one upload. The message is committed, not retried."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def source_location(event: stream_pb2.RecordingFinished, bucket: str) -> str:
    """Return the S3 key of the uploaded source, or raise UploadRejected.

    The key must be in ``bucket`` and end with
    ``<tenant_id>/uploads/<session_id>/source``.
    """
    try:
        uuid.UUID(event.session_id)
    except ValueError:
        raise UploadRejected("invalid-session-id") from None
    if not event.tenant_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", event.tenant_id):
        raise UploadRejected("invalid-tenant-id")
    m = re.fullmatch(r"s3://([^/]+)/(.+)", event.recording_url or "")
    if not m or m.group(1) != bucket:
        raise UploadRejected("foreign-source")
    key = m.group(2)
    suffix = f"{event.tenant_id}/uploads/{event.session_id}/source"
    if key != suffix and not key.endswith("/" + suffix):
        raise UploadRejected("foreign-source")
    return key


def transcode_to_wav(src_path: str, dst_path: str, *, max_duration_s: float = MAX_DURATION_S,
                     timeout_s: float = TRANSCODE_TIMEOUT_S, ffmpeg_bin: str = FFMPEG_BIN) -> int:
    """Decode the first audio stream of ``src_path`` into a 16 kHz mono PCM16 WAV.

    Returns the number of samples written. Raises UploadRejected with reason
    ``timeout``, ``too-long``, ``decode-failed`` or ``no-audio``.
    """
    cmd = [
        ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file", "-format_whitelist", ALLOWED_DEMUXERS,
        "-i", "file:" + src_path,
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", str(rw.NUM_CHANNELS), "-ar", str(rw.DEFAULT_SR), "-c:a", "pcm_s16le",
        "-t", f"{max_duration_s + 1:.3f}",
        "-f", "s16le", "pipe:1",
    ]
    max_bytes = int(max_duration_s * rw.DEFAULT_SR) * rw.SAMPLE_WIDTH_BYTES
    timed_out = threading.Event()
    total = 0
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)

    def _kill_on_timeout():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, _kill_on_timeout)
    timer.daemon = True
    timer.start()
    try:
        with wave.open(dst_path, "wb") as out:
            out.setnchannels(rw.NUM_CHANNELS)
            out.setsampwidth(rw.SAMPLE_WIDTH_BYTES)
            out.setframerate(rw.DEFAULT_SR)
            while True:
                chunk = proc.stdout.read(_READ_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise UploadRejected("too-long")
                out.writeframes(chunk)
        returncode = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
    if timed_out.is_set():
        raise UploadRejected("timeout")
    if returncode != 0:
        raise UploadRejected("decode-failed")
    samples = total // rw.SAMPLE_WIDTH_BYTES
    if samples == 0:
        raise UploadRejected("no-audio")
    return samples


def _download_source(s3, bucket: str, key: str, dst_path: str) -> None:
    """Download the source object after checking its size. Missing or too large is permanent."""
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        # botocore ClientError carries the S3 error code in .response.
        code = str((getattr(e, "response", None) or {}).get("Error", {}).get("Code"))
        if code in ("404", "NoSuchKey", "NotFound"):
            raise UploadRejected("missing-source") from None
        raise
    if int(head.get("ContentLength", 0)) > MAX_SOURCE_BYTES:
        raise UploadRejected("too-large")
    s3.download_file(bucket, key, dst_path)


def _delete_source(s3, bucket: str, key: str) -> None:
    try:
        s3.delete_object(Bucket=bucket, Key=key)
    except Exception:
        logger.warning("Failed to delete uploaded source s3://%s/%s", bucket, key, exc_info=True)


def _store_and_publish(event, msg, wav_path: str, producer) -> None:
    """Store the decoded WAV with the normal writer and publish recordings.finished."""
    controls = parse_stream_controls_from_kafka_headers(msg.headers() or None)
    rec = rw.SessionRecording(
        session_id=event.session_id,
        tenant_id=event.tenant_id,
        sample_rate=rw.DEFAULT_SR,
        lang=event.lang,
        writer=rw.make_writer_for_session(event.session_id, event.tenant_id, rw.DEFAULT_SR),
        trace_headers=msg.headers(),
        store_recording=bool(controls.store_recording),
        final_tracks=controls.final_tracks,
    )
    with wave.open(wav_path, "rb") as src:
        while True:
            frames = src.readframes(_READ_CHUNK)
            if not frames:
                break
            rec.append_pcm16(frames)
    rw.finalize_session(event.session_id, rec, producer)
    if producer.flush(30.0) > 0:
        raise RuntimeError("recordings.finished was not acknowledged")


def process_upload(msg, *, producer, s3, bucket: Optional[str] = None) -> str:
    """Process one ``recordings.uploaded`` message.

    Returns ``"published"`` or the reason the upload was rejected. Rejections
    are permanent and the caller commits the message. Other exceptions (storage
    or Kafka unavailable) propagate so the caller retries the message.
    """
    bucket = bucket or rw.S3_BUCKET
    event = stream_pb2.RecordingFinished()
    try:
        event.ParseFromString(msg.value() or b"")
        key = source_location(event, bucket)
    except UploadRejected as e:
        logger.error("Rejected upload event: reason=%s", e.reason)
        return e.reason
    except Exception:
        logger.error("Rejected upload event: reason=invalid-event")
        return "invalid-event"

    log_ctx = {"session_id": event.session_id, "tenant_id": event.tenant_id}
    logger.info("Processing uploaded source: %s", log_ctx)
    workdir = tempfile.mkdtemp(prefix="upload_")
    finished = False
    try:
        controls = parse_stream_controls_from_kafka_headers(msg.headers() or None)
        if not controls.want_final:
            raise UploadRejected("final-output-disabled")
        src_path = os.path.join(workdir, "source")
        wav_path = os.path.join(workdir, "audio.wav")
        _download_source(s3, bucket, key, src_path)
        samples = transcode_to_wav(src_path, wav_path)
        _store_and_publish(event, msg, wav_path, producer)
        finished = True
        logger.info("Uploaded source converted: %s duration_s=%.2f", log_ctx, samples / rw.DEFAULT_SR)
        return "published"
    except UploadRejected as e:
        finished = True
        logger.error("Uploaded source rejected: %s reason=%s", log_ctx, e.reason)
        return e.reason
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if finished:
            _delete_source(s3, bucket, key)


def run(producer, stop_event: threading.Event) -> None:
    """Consume ``recordings.uploaded`` until ``stop_event`` is set."""
    consumer = rw.make_consumer(
        group_id=f"{rw.GROUP_ID}-uploads",
        **{"max.poll.interval.ms": int((TRANSCODE_TIMEOUT_S + 600) * 1000)},
    )
    consumer.subscribe([TOPIC_RECORDING_UPLOADED])
    s3 = rw.make_s3_client()
    logger.info("Upload conversion ready: topic=%s", TOPIC_RECORDING_UPLOADED)
    try:
        while not stop_event.is_set():
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.warning("Kafka error on %s: %s", TOPIC_RECORDING_UPLOADED, msg.error())
                stop_event.wait(RETRY_BACKOFF_S)
                continue
            try:
                process_upload(msg, producer=producer, s3=s3)
                consumer.commit(msg, asynchronous=False)
            except Exception:
                logger.exception("Upload conversion failed; retrying the message")
                consumer.seek(TopicPartition(msg.topic(), msg.partition(), msg.offset()))
                stop_event.wait(RETRY_BACKOFF_S)
    finally:
        consumer.close()


def start(producer, stop_event: threading.Event) -> Optional[threading.Thread]:
    """Start the upload consumer thread when enabled and S3 is configured."""
    if not UPLOADS_ENABLED:
        logger.info("Upload conversion disabled (RECORDER_UPLOADS_ENABLED=false)")
        return None
    if not rw.S3_BUCKET:
        logger.warning("Upload conversion needs S3_BUCKET; uploads will not be processed")
        return None
    thread = threading.Thread(target=run, args=(producer, stop_event), name="uploads", daemon=True)
    thread.start()
    return thread
