from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .providers import EnrollmentProvider

logger = logging.getLogger(__name__)


EmbeddingFn = Callable[[bytes], np.ndarray]


@dataclass(frozen=True, slots=True)
class EnrollmentSnapshot:
    """Materialized enrollment state for a tenant."""

    tenant_id: str
    embeddings_by_label: Dict[str, np.ndarray]
    loaded_at_s: float


class EnrollmentCache:
    """TTL + LRU cache for enrolled speaker embeddings.

    - Keyed by tenant_id
    - On cache miss / TTL expiry, fetches all speakers for the tenant via provider
      and computes mean embedding per human label.

    Observability notes:
    - We log when a tenant snapshot is loaded and how many speaker labels were produced.
    - We intentionally do not log secrets; sample URLs may be logged at DEBUG.

    This is intentionally simple (eventual consistency). TTL is expected to be
    on the order of minutes.
    """

    def __init__(
        self,
        *,
        provider: EnrollmentProvider,
        embed_fn: EmbeddingFn,
        ttl_s: float = 300.0,
        max_tenants: int = 128,
    ) -> None:
        self._provider = provider
        self._embed_fn = embed_fn
        self._ttl_s = float(ttl_s)
        self._max_tenants = int(max_tenants)
        self._lru: "OrderedDict[str, EnrollmentSnapshot]" = OrderedDict()

    def invalidate(self, tenant_id: str) -> None:
        self._lru.pop(str(tenant_id), None)

    def get(self, tenant_id: str) -> EnrollmentSnapshot:
        tid = str(tenant_id)
        now = time.time()

        snap = self._lru.get(tid)
        if snap and (now - snap.loaded_at_s) <= self._ttl_s:
            # bump LRU
            self._lru.move_to_end(tid)
            logger.debug(
                "EnrollmentCache hit tenant=%s labels=%d age_s=%.1f",
                tid,
                len(snap.embeddings_by_label),
                now - snap.loaded_at_s,
            )
            return snap

        # reload
        logger.info("EnrollmentCache load tenant=%s (ttl_s=%.1f)", tid, self._ttl_s)
        t0 = time.time()
        snap = self._load_tenant(tid)
        dt = time.time() - t0

        self._lru[tid] = snap
        self._lru.move_to_end(tid)

        # enforce max size
        while len(self._lru) > self._max_tenants:
            evicted_tid, _ = self._lru.popitem(last=False)
            logger.info("EnrollmentCache evict tenant=%s (max_tenants=%d)", evicted_tid, self._max_tenants)

        if snap.embeddings_by_label:
            logger.info(
                "EnrollmentCache loaded tenant=%s labels=%d in %.2fs",
                tid,
                len(snap.embeddings_by_label),
                dt,
            )
        else:
            logger.warning(
                "EnrollmentCache loaded tenant=%s but produced 0 embeddings (check manifests/samples)",
                tid,
            )

        return snap

    def _load_tenant(self, tenant_id: str) -> EnrollmentSnapshot:
        speaker_ids = self._provider.list_speaker_ids(tenant_id)
        logger.info("EnrollmentCache: tenant=%s discovered speaker_ids=%d", tenant_id, len(speaker_ids))

        embeddings_by_label: Dict[str, np.ndarray] = {}

        for speaker_id in speaker_ids:
            try:
                manifest = self._provider.load_speaker_manifest(tenant_id, speaker_id)
            except Exception:
                logger.warning(
                    "Failed to load speaker manifest tenant=%s speaker_id=%s", tenant_id, speaker_id, exc_info=True
                )
                continue

            label = (manifest.label or "").strip()
            if not label:
                logger.warning("Skipping speaker with empty label tenant=%s speaker_id=%s", tenant_id, speaker_id)
                continue

            if not manifest.samples:
                logger.warning("Skipping speaker with 0 samples tenant=%s speaker_id=%s", tenant_id, speaker_id)
                continue

            logger.debug(
                "EnrollmentCache: tenant=%s speaker_id=%s label=%s samples=%d",
                tenant_id,
                speaker_id,
                label,
                len(manifest.samples),
            )

            embs = []
            for s in manifest.samples:
                try:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Embedding enrollment sample tenant=%s speaker_id=%s url=%s",
                            tenant_id,
                            speaker_id,
                            s.url,
                        )
                    b = self._provider.open_url(s.url)
                    emb = self._embed_fn(b)
                    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                    if emb.size == 0:
                        continue
                    embs.append(emb)
                except Exception:
                    logger.warning(
                        "Failed to embed enrollment sample tenant=%s speaker_id=%s url=%s",
                        tenant_id,
                        speaker_id,
                        s.url,
                        exc_info=True,
                    )

            if not embs:
                logger.warning(
                    "No usable embeddings for speaker tenant=%s speaker_id=%s label=%s",
                    tenant_id,
                    speaker_id,
                    label,
                )
                continue

            # mean + normalize
            arr = np.stack(embs, axis=0)
            mean = arr.mean(axis=0)
            denom = float(np.linalg.norm(mean) + 1e-12)
            mean = mean / denom

            # if duplicate label exists, last-write-wins (but log)
            if label in embeddings_by_label:
                logger.warning(
                    "Duplicate speaker label for tenant=%s: label=%s. Overwriting.", tenant_id, label
                )
            embeddings_by_label[label] = mean

        return EnrollmentSnapshot(
            tenant_id=tenant_id,
            embeddings_by_label=embeddings_by_label,
            loaded_at_s=time.time(),
        )
