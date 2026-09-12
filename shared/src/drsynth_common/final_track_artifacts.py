"""Tenant-scoped S3 storage for immutable final-track inputs and outcomes."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path

from proto_gen import stream_pb2 as pb
from drsynth_common.final_tracks import (
    ContractError, MAX_EVENT_BYTES, MAX_TRANSCRIPT_BYTES, canonical_json,
    file_digest, require_uuid,
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
    """Validate exact source keys and publish one immutable outcome per result.

    The source bucket/prefix is operator-controlled; event URLs cannot select
    other storage. Outputs use the same bucket and a separate fixed prefix.
    """

    def __init__(self, client, bucket: str, recording_prefix="recordings", result_prefix="final-tracks"):
        if not bucket or any(c in bucket for c in "/\\:%"):
            raise ValueError("invalid configured bucket")
        for prefix in (recording_prefix, result_prefix):
            if not prefix or any(part in ("", ".", "..") for part in prefix.split("/")):
                raise ValueError("invalid configured prefix")
            if any(c in prefix for c in "\\:%"):
                raise ValueError("invalid configured prefix")
        self.client, self.bucket = client, bucket
        self.recording_prefix, self.result_prefix = recording_prefix, result_prefix

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

    def result_base(self, event):
        """Derive a tenant/source/track/run namespace from validated identities."""
        from drsynth_common.final_tracks import NAME
        if not NAME.fullmatch(event.track_id):
            raise ContractError("invalid_track")
        return "/".join([self.result_prefix, require_uuid(event.tenant_id),
                         require_uuid(event.session_id), require_uuid(event.source.artifact_id),
                         event.track_id, require_uuid(event.run_id)])

    def _read_json(self, key, max_bytes):
        """Read bounded JSON or None on a genuinely absent object."""
        from botocore.exceptions import ClientError
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        try:
            if response["ContentLength"] > max_bytes:
                raise ContractError("stored_artifact_too_large")
            data = response["Body"].read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ContractError("stored_artifact_too_large")
            return json.loads(data)
        finally:
            response["Body"].close()

    def read_outcome(self, event):
        """Recover the exact accepted protobuf and primary-projection bytes."""
        key = f"{self.result_base(event)}/{require_uuid(event.result_id)}.outcome.json"
        value = self._read_json(key, 2 * MAX_TRANSCRIPT_BYTES)
        if value is None:
            return None
        try:
            canonical = base64.b64decode(value["canonical"], validate=True)
            legacy = base64.b64decode(value["legacy"], validate=True) if value["legacy"] else None
            actual = pb.FinalTrackResult.FromString(canonical)
            if len(canonical) > MAX_EVENT_BYTES or (legacy and len(legacy) > MAX_TRANSCRIPT_BYTES):
                raise ValueError()
            for name in ("tenant_id", "session_id", "plan_id", "track_id", "profile_id",
                         "run_id", "result_id", "primary"):
                if getattr(event, name) != getattr(actual, name):
                    raise ValueError()
            if actual.source != event.source or actual.status not in ("succeeded", "failed"):
                raise ValueError()
            return actual, canonical, legacy
        except (KeyError, ValueError, TypeError):
            raise ContractError("stored_outcome_mismatch") from None

    def put_transcript(self, event, transcript: dict):
        """Write immutable attempt output; losing attempts remain unreferenced."""
        import hashlib
        data = canonical_json(transcript)
        if len(data) > MAX_TRANSCRIPT_BYTES:
            raise ContractError("transcript_too_large")
        key = f"{self.result_base(event)}/{require_uuid(event.attempt_id)}.transcript.json"
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data,
                               ContentType="application/json", IfNoneMatch="*")
        return f"s3://{self.bucket}/{key}", hashlib.sha256(data).hexdigest()

    def publish_once(self, event, legacy):
        """Atomically select the first terminal outcome, then return its bytes."""
        from botocore.exceptions import ClientError
        canonical = event.SerializeToString(deterministic=True)
        if len(canonical) > MAX_EVENT_BYTES:
            raise ContractError("outcome_too_large")
        value = canonical_json(dict(
            canonical=base64.b64encode(canonical).decode("ascii"),
            legacy=base64.b64encode(legacy).decode("ascii") if legacy else None))
        key = f"{self.result_base(event)}/{require_uuid(event.result_id)}.outcome.json"
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=value,
                                   ContentType="application/json", IfNoneMatch="*")
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in ("PreconditionFailed", "412"):
                raise
        result = self.read_outcome(event)
        if result is None:
            raise RuntimeError("outcome disappeared after conditional publication")
        return result
