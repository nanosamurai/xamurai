from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class SpeakerSample:
    """One enrollment sample for a speaker.

    The canonical source of truth is object storage (S3) URL.

    Notes:
    - Workers/services treat samples as immutable blobs.
    - Optional fields help with debugging/cache validation.
    """

    url: str
    sample_id: Optional[str] = None
    sha256: Optional[str] = None
    duration_s: Optional[float] = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SpeakerSample":
        return SpeakerSample(
            url=str(d.get("url") or ""),
            sample_id=(str(d["sample_id"]) if d.get("sample_id") is not None else None),
            sha256=(str(d["sha256"]) if d.get("sha256") is not None else None),
            duration_s=(float(d["duration_s"]) if d.get("duration_s") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class SpeakerManifest:
    """Per-speaker manifest stored as JSON (speaker.json)."""

    speaker_id: str
    label: str
    updated_at: Optional[str] = None  # ISO timestamp string
    samples: tuple[SpeakerSample, ...] = ()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SpeakerManifest":
        samples = d.get("samples") or []
        return SpeakerManifest(
            speaker_id=str(d.get("speaker_id") or ""),
            label=str(d.get("label") or ""),
            updated_at=(str(d["updated_at"]) if d.get("updated_at") is not None else None),
            samples=tuple(SpeakerSample.from_dict(x) for x in samples if isinstance(x, dict)),
        )

    @staticmethod
    def loads(text: str) -> "SpeakerManifest":
        return SpeakerManifest.from_dict(json.loads(text))

    def validate(self) -> None:
        if not self.speaker_id:
            raise ValueError("SpeakerManifest missing speaker_id")
        if not self.label:
            raise ValueError("SpeakerManifest missing label")
        if not self.samples:
            raise ValueError(f"SpeakerManifest speaker_id={self.speaker_id} has no samples")
        for s in self.samples:
            if not s.url:
                raise ValueError(f"SpeakerManifest speaker_id={self.speaker_id} has sample with empty url")

    def updated_at_dt(self) -> Optional[datetime]:
        if not self.updated_at:
            return None
        try:
            # Accept e.g. 2026-02-20T10:00:00Z
            v = self.updated_at.replace("Z", "+00:00")
            return datetime.fromisoformat(v)
        except Exception:
            return None


@dataclass(frozen=True, slots=True)
class TenantSpeakerIndexEntry:
    """Optional index entry for enumerating speakers under a tenant."""

    speaker_id: str
    label: str
    updated_at: Optional[str] = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TenantSpeakerIndexEntry":
        return TenantSpeakerIndexEntry(
            speaker_id=str(d.get("speaker_id") or ""),
            label=str(d.get("label") or ""),
            updated_at=(str(d["updated_at"]) if d.get("updated_at") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class TenantSpeakerIndex:
    """Optional tenant-level index (index.json)."""

    tenant_id: str
    speakers: tuple[TenantSpeakerIndexEntry, ...] = ()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TenantSpeakerIndex":
        speakers = d.get("speakers") or []
        return TenantSpeakerIndex(
            tenant_id=str(d.get("tenant_id") or ""),
            speakers=tuple(TenantSpeakerIndexEntry.from_dict(x) for x in speakers if isinstance(x, dict)),
        )

    @staticmethod
    def loads(text: str) -> "TenantSpeakerIndex":
        return TenantSpeakerIndex.from_dict(json.loads(text))
