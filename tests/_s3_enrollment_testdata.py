"""Helpers for LocalStack-backed enrollment integration tests.

We keep this in tests/ so it can be reused across rtservice + whisperx_worker tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import soundfile as sf


SR = 16000


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    n_out = int(round(float(x.size) * float(sr_out) / float(sr_in)))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)
    t_old = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)


def make_enrollment_wav_from_test_audio(
    *,
    src_wav_path: str | Path,
    out_wav_path: str | Path,
    seconds: float = 5.0,
) -> None:
    """Create a small, valid PCM16 WAV sample from the test wav."""

    src_wav_path = Path(src_wav_path)
    out_wav_path = Path(out_wav_path)

    x, sr = sf.read(src_wav_path, dtype="float32")
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x[:, 0]

    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if int(sr) != SR:
        x = _resample_linear(x, int(sr), SR)

    n = int(seconds * SR)
    if x.size < n:
        reps = int(np.ceil(float(n) / float(max(1, x.size))))
        x = np.tile(x, reps)

    x = x[:n]
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav_path, x, SR, subtype="PCM_16")


def upload_tenant_enrollment_to_s3(
    *,
    s3_client: Any,
    bucket: str,
    prefix: str,
    tenant_id: str,
    speaker_id: str,
    label: str,
    sample_wav_path: str | Path,
) -> Dict[str, str]:
    """Upload a minimal enrollment structure to S3.

    Returns dict with keys: speaker_manifest_key, sample_key, sample_url
    """

    sample_wav_path = Path(sample_wav_path)

    base = "/".join([p.strip("/") for p in [prefix, tenant_id, "speakers", speaker_id] if p])

    sample_key = f"{base}/samples/sample.wav"
    manifest_key = f"{base}/speaker.json"

    s3_client.put_object(
        Bucket=bucket,
        Key=sample_key,
        Body=sample_wav_path.read_bytes(),
        ContentType="audio/wav",
    )

    sample_url = f"s3://{bucket}/{sample_key}"
    speaker_manifest = {
        "speaker_id": speaker_id,
        "label": label,
        "updated_at": "2026-01-01T00:00:00Z",
        "samples": [
            {
                "url": sample_url,
                "sample_id": "sample",
            }
        ],
    }

    s3_client.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=json.dumps(speaker_manifest).encode("utf-8"),
        ContentType="application/json",
    )

    # Optional index.json to speed up listing
    index_key = "/".join([p.strip("/") for p in [prefix, tenant_id, "speakers", "index.json"] if p])
    index = {
        "tenant_id": tenant_id,
        "speakers": [
            {
                "speaker_id": speaker_id,
                "label": label,
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ],
    }
    s3_client.put_object(
        Bucket=bucket,
        Key=index_key,
        Body=json.dumps(index).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "speaker_manifest_key": manifest_key,
        "sample_key": sample_key,
        "sample_url": sample_url,
        "index_key": index_key,
    }
