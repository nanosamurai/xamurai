"""Verified Mistral-format artifacts; no remote Python or client model selection."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    model_id: str
    revision: str
    languages: tuple[str, ...]
    files: dict[str, str]

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", self.profile_id):
            raise ValueError("invalid profile_id")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.model_id):
            raise ValueError("model_id must be a Hugging Face repository ID")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("model revision must be an immutable commit")
        if set(self.files) != {"params.json", "tekken.json", "consolidated.safetensors"}:
            raise ValueError("profile must contain exactly the three Mistral-format artifacts")
        if any(not re.fullmatch(r"[0-9a-f]{64}", v) for v in self.files.values()):
            raise ValueError("every artifact requires a SHA-256 digest")
        if not self.languages or len(set(self.languages)) != len(self.languages):
            raise ValueError("languages must be a nonempty unique list")
        if any(not re.fullmatch(r"[a-z]{2,3}", v) for v in self.languages):
            raise ValueError("languages must use public short codes")

    @property
    def digest(self):
        return "sha256:" + self.files["consolidated.safetensors"]


DEFAULT_PROFILE = ModelProfile(
    profile_id="voxtral-mini-4b-realtime-2602-vllm-r1",
    model_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
    revision="2769294da9567371363522aac9bbcfdd19447add",
    languages=("en", "fr", "es", "de", "ru", "zh", "ja", "it", "pt", "nl", "ar", "hi", "ko"),
    files={
        "params.json": "2ace010ebf7f0b62c60747d91c6d140e3c7238632d3e9c63d60a2bd2065ea301",
        "tekken.json": "8434af1d39eba99f0ef46cf1450bf1a63fa941a26933a1ef5dbbf4adf0d00e44",
        "consolidated.safetensors": "263f178fe752c90a2ae58f037a95ed092db8b14768b0978b8c48f66979c8345d",
    },
)


def load_profile():
    path = os.getenv("VOXTRAL_MODEL_MANIFEST")
    if not path:
        return DEFAULT_PROFILE
    manifest = Path(path)
    if manifest.stat().st_size > 16_384:
        raise ValueError("model manifest is too large")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["languages"] = tuple(data["languages"])
    return ModelProfile(**data)


def verify_file(path: Path, digest: str):
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"model artifact checksum mismatch: {path.name}")


def resolve_model(profile: ModelProfile) -> str:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(
        repo_id=profile.model_id, revision=profile.revision,
        allow_patterns=list(profile.files),
    ))
    for name, digest in profile.files.items():
        verify_file(path / name, digest)
    return str(path)
