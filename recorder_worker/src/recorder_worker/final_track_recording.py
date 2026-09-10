"""Opt-in immutable recording assembly for a frozen final-track plan."""

import os
import tempfile
import uuid
import wave
from pathlib import Path
from drsynth_common.final_tracks import ContractError, MAX_AUDIO_SECONDS, read_plan
from drsynth_common.final_track_artifacts import S3Artifacts, s3_client


def resolve_plan(chunk, headers, controls):
    """Fail before accepting audio when a planned session cannot be supported."""
    plan = read_plan(headers, chunk.tenant_id, chunk.session_id)
    if plan is not None:
        if os.getenv("FINAL_TRACKS_ENABLED") != "true":
            raise ContractError("final_tracks_disabled")
        if not controls.want_final or not controls.store_recording:
            raise ContractError("unsupported_retention_or_outputs")
        if chunk.sample_rate != 16000 or len(chunk.pcm16_le) % 2:
            raise ContractError("unsupported_audio_geometry")
        if os.getenv("RECORDING_STORAGE_BACKEND") != "s3":
            raise ContractError("final_tracks_require_s3")
    return plan


class ImmutableRecordingWriter:
    """Upload one bounded PCM WAV to a new tenant/session-scoped object key."""

    def __init__(self, plan):
        self.plan = plan
        self._directory = tempfile.TemporaryDirectory(prefix="final-recording-")
        self._path = Path(self._directory.name) / "audio.wav"
        self._wave = wave.open(str(self._path), "wb")
        self._wave.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        self._samples = 0
        self.source = None

    def append_pcm16(self, pcm):
        if len(pcm) % 2 or self._samples + len(pcm) // 2 > MAX_AUDIO_SECONDS * 16000:
            raise ContractError("recording_limit_exceeded")
        self._wave.writeframes(pcm)
        self._samples += len(pcm) // 2

    def close_and_get_url(self):
        if self.source is None:
            self._wave.close()
            if not self._samples:
                raise ContractError("empty_recording")
            store = S3Artifacts(s3_client(), os.environ["S3_BUCKET"],
                                recording_prefix=os.getenv("FINAL_RECORDING_PREFIX", "recordings"))
            self.source = store.put_audio(self._path, self.plan["tenant_id"],
                                          self.plan["session_id"], str(uuid.uuid4()))
            self._directory.cleanup()
        return self.source.storage_uri
