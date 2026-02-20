import json
from pathlib import Path

import numpy as np

from drsynth_common.enrollment import (
    EnrollmentCache,
    LocalEnrollmentProvider,
)


def _fake_embed_fn(data: bytes) -> np.ndarray:
    # deterministic: embed = normalized histogram of byte values (tiny)
    if not data:
        return np.zeros(4, dtype=np.float32)
    s = int(sum(data))
    v = np.array([
        (s % 97) / 97.0,
        (s % 89) / 89.0,
        (s % 83) / 83.0,
        (s % 79) / 79.0,
    ], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def test_enrollment_cache_local_provider(tmp_path: Path):
    tenant_id = "t-1"
    speaker_id = "spk-1"

    # local layout:
    # <root>/<tenant>/speakers/<speaker_id>/speaker.json
    base = tmp_path / tenant_id / "speakers" / speaker_id
    base.mkdir(parents=True)

    # create a fake sample file
    sample_path = base / "sample1.wav"
    sample_path.write_bytes(b"abc123")

    manifest = {
        "speaker_id": speaker_id,
        "label": "Dr Novak",
        "updated_at": "2026-02-20T10:00:00Z",
        "samples": [
            {"url": str(sample_path)},
        ],
    }
    (base / "speaker.json").write_text(json.dumps(manifest), encoding="utf-8")

    provider = LocalEnrollmentProvider(root_dir=str(tmp_path))
    cache = EnrollmentCache(provider=provider, embed_fn=_fake_embed_fn, ttl_s=300.0, max_tenants=10)

    snap = cache.get(tenant_id)
    assert snap.tenant_id == tenant_id
    assert "Dr Novak" in snap.embeddings_by_label
    emb = snap.embeddings_by_label["Dr Novak"]
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (4,)
    assert np.isfinite(emb).all()


def test_enrollment_cache_ttl_and_invalidate(tmp_path: Path, monkeypatch):
    tenant_id = "t-1"
    speaker_id = "spk-1"
    base = tmp_path / tenant_id / "speakers" / speaker_id
    base.mkdir(parents=True)

    sample_path = base / "sample1.wav"
    sample_path.write_bytes(b"aaa")

    manifest = {
        "speaker_id": speaker_id,
        "label": "SpeakerA",
        "samples": [{"url": str(sample_path)}],
    }
    (base / "speaker.json").write_text(json.dumps(manifest), encoding="utf-8")

    provider = LocalEnrollmentProvider(root_dir=str(tmp_path))

    # force time control
    t = {"now": 1000.0}

    def fake_time():
        return t["now"]

    import drsynth_common.enrollment.cache as cache_mod

    monkeypatch.setattr(cache_mod.time, "time", fake_time)

    cache = EnrollmentCache(provider=provider, embed_fn=_fake_embed_fn, ttl_s=10.0, max_tenants=10)

    snap1 = cache.get(tenant_id)
    assert "SpeakerA" in snap1.embeddings_by_label

    # change sample bytes but within TTL should not reload
    sample_path.write_bytes(b"bbb")
    t["now"] = 1005.0
    snap2 = cache.get(tenant_id)
    assert snap2.loaded_at_s == snap1.loaded_at_s

    # after TTL, should reload
    t["now"] = 1015.0
    snap3 = cache.get(tenant_id)
    assert snap3.loaded_at_s != snap1.loaded_at_s

    # invalidate forces reload next time
    cache.invalidate(tenant_id)
    t["now"] = 1016.0
    snap4 = cache.get(tenant_id)
    assert snap4.loaded_at_s == 1016.0
