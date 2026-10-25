"""Verify the default diarization and enrollment checkpoints before loading them.

Pyannote explicitly enables pickle compatibility for its checkpoints. Keep that
decision inside its loader; never change torch.load or its process-wide allowlist.
Operator-selected custom pipelines remain an explicit model trust decision.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
_CONFIG = (
    DIARIZATION_MODEL, "84fd25912480287da0247647c3d2b4853cb3ee5d", "config.yaml",
    "04ad9cd59a93c3a7c754200ecc9e1c4ba87bf1657ef8a4debf7555e711daeeda",
)
_SEGMENTATION = (
    "pyannote/segmentation-3.0", "e66f3d3b9eb0873085418a7b813d3b369bf160bb", "pytorch_model.bin",
    "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298",
)
_SPEAKER_EMBEDDING = (
    "pyannote/wespeaker-voxceleb-resnet34-LM", "837717ddb9ff5507820346191109dc79c958d614", "pytorch_model.bin",
    "366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56",
)
_PLDA_TRANSFORM = (
    "pyannote/speaker-diarization-community-1", "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    "plda/xvec_transform.npz", "325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f",
)
_PLDA = (
    "pyannote/speaker-diarization-community-1", "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    "plda/plda.npz", "9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255",
)
_ENROLLMENT = (
    "pyannote/embedding", "4db4899737a38b2d618bbd74350915aa10293cb2", "pytorch_model.bin",
    "4bcec986de13da7af7ac88736572692359950df63669989c4f78b294934c9089",
)


def _verified_file(artifact: tuple[str, str, str, str], token: str) -> str:
    """Return one pinned HF file after checking its complete SHA-256 digest.

    Check cached files on every load too. A mismatch raises before a checkpoint
    parser runs; failed downloads propagate without falling back to an unpinned ID.
    """
    from huggingface_hub import hf_hub_download

    repo, revision, filename, expected = artifact
    path = Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename, token=token))
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"Model artifact checksum mismatch: {repo}/{filename}")
    return str(path)


def load_diarization_pipeline(model_id: str, token: str):
    """Load the verified default pipeline or an operator-trusted custom pipeline.

    The default config, nested checkpoints and PLDA data are immutable and checked
    before Pyannote sees them. Custom IDs preserve the existing operator setting;
    their revisions, nested artifacts and trust remain the operator's responsibility.
    """
    from pyannote.audio import Pipeline

    if model_id != DIARIZATION_MODEL:
        return Pipeline.from_pretrained(model_id, token=token)

    import yaml

    config_path = _verified_file(_CONFIG, token)
    segmentation_path = _verified_file(_SEGMENTATION, token)
    embedding_path = _verified_file(_SPEAKER_EMBEDDING, token)
    # Pyannote 4 loads PLDA even for the legacy agglomerative pipeline. Supplying
    # the verified directory prevents its implicit, unversioned Hub download.
    transform_path = Path(_verified_file(_PLDA_TRANSFORM, token))
    plda_path = Path(_verified_file(_PLDA, token))
    if transform_path.parent != plda_path.parent:
        raise RuntimeError("Verified PLDA files have different directories")
    with Path(config_path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    params = config["pipeline"]["params"]
    if params.get("segmentation") != _SEGMENTATION[0] or params.get("embedding") != _SPEAKER_EMBEDDING[0]:
        raise RuntimeError("Pinned diarization config has unexpected checkpoint references")
    params["segmentation"] = {"checkpoint": segmentation_path}
    params["embedding"] = embedding_path
    params["plda"] = {"checkpoint": str(plda_path.parent)}
    return Pipeline.from_pretrained(config, token=token)


def load_enrollment_model(token: str):
    """Load the verified default enrollment checkpoint without changing Torch defaults."""
    from pyannote.audio import Model

    return Model.from_pretrained(_verified_file(_ENROLLMENT, token), token=token)
