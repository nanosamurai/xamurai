from __future__ import annotations

import abc
import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .models import SpeakerManifest, TenantSpeakerIndex

logger = logging.getLogger(__name__)


class EnrollmentProvider(abc.ABC):
    """Abstract source of tenant speaker enrollment data."""

    @abc.abstractmethod
    def list_speaker_ids(self, tenant_id: str) -> list[str]:
        """Return all speaker IDs for a tenant."""

    @abc.abstractmethod
    def load_speaker_manifest(self, tenant_id: str, speaker_id: str) -> SpeakerManifest:
        """Load and parse `speaker.json` for a speaker."""

    @abc.abstractmethod
    def open_url(self, url: str) -> bytes:
        """Fetch a sample by URL and return its bytes.

        Implementations may support `s3://`, `file://`, etc.
        """


@dataclass(frozen=True, slots=True)
class LocalEnrollmentProvider(EnrollmentProvider):
    """Local filesystem enrollment provider.

    Layout:
      <root>/<tenant_id>/speakers/<speaker_id>/speaker.json

    This is intended for dev/testing.
    """

    root_dir: str

    def _speaker_json_path(self, tenant_id: str, speaker_id: str) -> Path:
        return Path(self.root_dir) / str(tenant_id) / "speakers" / str(speaker_id) / "speaker.json"

    def _index_path(self, tenant_id: str) -> Path:
        return Path(self.root_dir) / str(tenant_id) / "speakers" / "index.json"

    def list_speaker_ids(self, tenant_id: str) -> list[str]:
        # Prefer index.json if present.
        idx = self._index_path(tenant_id)
        if idx.exists():
            try:
                text = idx.read_text(encoding="utf-8")
                parsed = TenantSpeakerIndex.loads(text)
                out = [e.speaker_id for e in parsed.speakers if e.speaker_id]
                # If index.json is corrupt/empty, fall back to directory scan.
                if out:
                    return out
            except Exception:
                logger.warning("Failed to read local speaker index %s, falling back to scan", idx, exc_info=True)

        base = Path(self.root_dir) / str(tenant_id) / "speakers"
        if not base.exists():
            return []

        out: list[str] = []
        for p in base.iterdir():
            if not p.is_dir():
                continue
            sj = p / "speaker.json"
            if sj.exists():
                out.append(p.name)
        return sorted(out)

    def load_speaker_manifest(self, tenant_id: str, speaker_id: str) -> SpeakerManifest:
        path = self._speaker_json_path(tenant_id, speaker_id)
        text = path.read_text(encoding="utf-8")
        m = SpeakerManifest.loads(text)
        m.validate()
        return m

    def open_url(self, url: str) -> bytes:
        if url.startswith("file://"):
            path = url[len("file://"):]
            return Path(path).read_bytes()
        # Treat bare paths as local.
        return Path(url).read_bytes()


@dataclass(frozen=True, slots=True)
class S3EnrollmentProvider(EnrollmentProvider):
    """S3 (or S3-compatible) enrollment provider.

    We assume enrollment objects exist under:
      s3://<bucket>/<prefix>/<tenant_id>/speakers/<speaker_id>/speaker.json

    and optionally:
      s3://<bucket>/<prefix>/<tenant_id>/speakers/index.json

    Notes:
    - Uses boto3; import is lazy so non-S3 deployments don't pay the dependency.
    - `open_url` supports s3:// and file://.
    """

    bucket: str
    prefix: str = ""
    endpoint: str = ""  # optional (minio/ceph)
    region: str = ""  # optional
    access_key: str = ""  # optional
    secret_key: str = ""  # optional
    force_path_style: bool = True

    def _client(self):
        import boto3
        from botocore.config import Config

        session_kwargs = {}
        if self.region:
            session_kwargs["region_name"] = self.region
        if self.access_key and self.secret_key:
            session_kwargs["aws_access_key_id"] = self.access_key
            session_kwargs["aws_secret_access_key"] = self.secret_key

        addressing_style = "path" if self.force_path_style else "virtual"
        cfg = Config(s3={"addressing_style": addressing_style})

        client_kwargs = {"config": cfg}
        if self.endpoint:
            client_kwargs["endpoint_url"] = self.endpoint

        return boto3.client("s3", **session_kwargs, **client_kwargs)

    def _join(self, *parts: str) -> str:
        return "/".join([p.strip("/") for p in parts if p is not None and str(p).strip("/")])

    def _key_for_index(self, tenant_id: str) -> str:
        return self._join(self.prefix, tenant_id, "speakers", "index.json")

    def _key_for_speaker_manifest(self, tenant_id: str, speaker_id: str) -> str:
        return self._join(self.prefix, tenant_id, "speakers", speaker_id, "speaker.json")

    def list_speaker_ids(self, tenant_id: str) -> list[str]:
        s3 = self._client()

        # Prefer index.json.
        try:
            obj = s3.get_object(Bucket=self.bucket, Key=self._key_for_index(tenant_id))
            text = obj["Body"].read().decode("utf-8")
            idx = TenantSpeakerIndex.loads(text)
            out = [e.speaker_id for e in idx.speakers if e.speaker_id]
            if out:
                return out
        except Exception:
            # fall back to prefix scan
            pass

        prefix = self._join(self.prefix, tenant_id, "speakers") + "/"
        paginator = s3.get_paginator("list_objects_v2")
        ids = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for c in page.get("Contents") or []:
                key = c.get("Key") or ""
                if not key.endswith("/speaker.json"):
                    continue
                # .../speakers/<speaker_id>/speaker.json
                parts = key.split("/")
                if len(parts) < 2:
                    continue
                speaker_id = parts[-2]
                if speaker_id:
                    ids.add(speaker_id)

        return sorted(ids)

    def load_speaker_manifest(self, tenant_id: str, speaker_id: str) -> SpeakerManifest:
        s3 = self._client()
        obj = s3.get_object(Bucket=self.bucket, Key=self._key_for_speaker_manifest(tenant_id, speaker_id))
        text = obj["Body"].read().decode("utf-8")
        m = SpeakerManifest.loads(text)
        m.validate()
        return m

    def open_url(self, url: str) -> bytes:
        if url.startswith("file://"):
            return Path(url[len("file://"):]).read_bytes()

        if not url.startswith("s3://"):
            # allow using provider for local urls too
            return Path(url).read_bytes()

        # parse s3://bucket/key
        rest = url[len("s3://"):]
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise ValueError(f"Invalid s3 url: {url}")

        s3 = self._client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
