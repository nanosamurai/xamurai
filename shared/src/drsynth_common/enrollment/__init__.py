"""Tenant-specific speaker enrollment utilities.

This package provides:
- Data models for per-speaker enrollment manifests stored in S3 (or locally)
- Provider interfaces for fetching manifests + WAV samples
- A TTL/LRU cache that materializes per-tenant enrolled speaker embeddings

Design goals:
- Keep workers/services *stateless* with respect to enrollment data.
- Avoid DB dependencies in Python services; data is discovered via object storage.
- Allow eventual consistency (cache TTL) and cheap refresh.

See: docs/enrolled-speakers-multi-tenant.md
"""

from .models import SpeakerManifest, SpeakerSample, TenantSpeakerIndex, TenantSpeakerIndexEntry
from .providers import EnrollmentProvider, LocalEnrollmentProvider, S3EnrollmentProvider
from .cache import EnrollmentCache, EnrollmentSnapshot

__all__ = [
    "SpeakerManifest",
    "SpeakerSample",
    "TenantSpeakerIndex",
    "TenantSpeakerIndexEntry",
    "EnrollmentProvider",
    "LocalEnrollmentProvider",
    "S3EnrollmentProvider",
    "EnrollmentCache",
    "EnrollmentSnapshot",
]
