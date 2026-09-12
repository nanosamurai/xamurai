"""Tenant-scoped object storage for source audio only."""

from __future__ import annotations

import os
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path

from proto_gen import stream_pb2 as pb
from drsynth_common.final_tracks import (
    ContractError, file_digest, require_uuid,
)


def s3_client():
    """Build a bounded S3 client using operator config or workload identity."""
    import boto3
    from botocore.config import Config

    kwargs = dict(region_name=os.getenv("S3_REGION") or "us-east-1",
                  config=Config(connect_timeout=5, read_timeout=60,
                                retries={"max_attempts": 3, "mode": "standard"},
                                s3={"addressing_style": "path" if os.getenv(
                                    "S3_FORCE_PATH_STYLE", "true").lower() == "true" else "virtual"}))
    if os.getenv("S3_ENDPOINT"):
        kwargs["endpoint_url"] = os.environ["S3_ENDPOINT"]
    if os.getenv("S3_ACCESS_KEY") and os.getenv("S3_SECRET_KEY"):
        kwargs.update(aws_access_key_id=os.environ["S3_ACCESS_KEY"],
                      aws_secret_access_key=os.environ["S3_SECRET_KEY"])
    return boto3.client("s3", **kwargs)


class S3Artifacts:
    """Validate and transfer immutable recording audio.

    The source bucket/prefix is operator-controlled; event URLs cannot select
    other storage. Transcript contents never enter this class.
    """

    def __init__(self, client, bucket: str, recording_prefix="recordings"):
        if not bucket or any(c in bucket for c in "/\\:%"):
            raise ValueError("invalid configured bucket")
        for prefix in (recording_prefix,):
            if not prefix or any(part in ("", ".", "..") for part in prefix.split("/")):
                raise ValueError("invalid configured prefix")
            if any(c in prefix for c in "\\:%"):
                raise ValueError("invalid configured prefix")
        self.client, self.bucket = client, bucket
        self.recording_prefix = recording_prefix

    @classmethod
    def from_env(cls):
        """Use the same source bucket configuration as the recorder."""
        return cls(s3_client(), os.getenv("S3_BUCKET") or os.getenv("RECORDING_S3_BUCKET", ""),
                   os.getenv("FINAL_RECORDING_PREFIX") or "recordings")

    def source_key(self, tenant_id, session_id, artifact_id):
        """Construct the sole allowed key for this recording generation."""
        return "/".join([self.recording_prefix, require_uuid(tenant_id),
                         require_uuid(session_id), require_uuid(artifact_id) + ".wav"])

    def validate_source(self, source, tenant_id, session_id):
        """Reject cross-tenant/path references without touching object storage."""
        key = self.source_key(tenant_id, session_id, source.artifact_id)
        if source.storage_uri != f"s3://{self.bucket}/{key}":
            raise ContractError("source_scope_mismatch")
        return key

    def put_audio(self, path: Path, tenant_id, session_id, artifact_id) -> pb.AudioArtifact:
        """Store one finalized PCM WAV under a write-once source key."""
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ContractError("unsupported_audio_format")
            sample_count = wav.getnframes()
        digest = file_digest(path)
        key = self.source_key(tenant_id, session_id, artifact_id)
        with path.open("rb") as body:
            response = self.client.put_object(
                Bucket=self.bucket, Key=key, Body=body, IfNoneMatch="*",
                ContentType="audio/wav", Metadata={"sha256": digest})
        return pb.AudioArtifact(
            artifact_id=artifact_id, storage_uri=f"s3://{self.bucket}/{key}",
            sha256=digest, size_bytes=path.stat().st_size, sample_rate=16000,
            sample_count=sample_count, media_type="audio/wav",
            version_id=response.get("VersionId", ""))

    @contextmanager
    def audio_file(self, source, tenant_id, session_id):
        """Download bounded bytes, verify digest/geometry, always remove temp audio."""
        from botocore.exceptions import ClientError

        key = self.validate_source(source, tenant_id, session_id)
        args = dict(Bucket=self.bucket, Key=key)
        if source.version_id:
            args["VersionId"] = source.version_id
        try:
            response = self.client.get_object(**args)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("NoSuchKey", "NoSuchVersion", "404"):
                raise ContractError("source_missing") from None
            raise
        body = response["Body"]
        try:
            if response["ContentLength"] != source.size_bytes:
                raise ContractError("audio_size_mismatch")
            with tempfile.TemporaryDirectory(prefix="final-track-") as directory:
                path = Path(directory) / "audio.wav"
                count = 0
                with path.open("wb") as out:
                    for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                        count += len(chunk)
                        if count > source.size_bytes:
                            raise ContractError("audio_size_mismatch")
                        out.write(chunk)
                if count != source.size_bytes or file_digest(path) != source.sha256:
                    raise ContractError("audio_digest_mismatch")
                try:
                    with wave.open(str(path), "rb") as wav:
                        if ((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes())
                                != (1, 2, source.sample_rate, source.sample_count)):
                            raise ContractError("audio_format_mismatch")
                except (wave.Error, EOFError):
                    raise ContractError("invalid_wav") from None
                yield path
        finally:
            body.close()
