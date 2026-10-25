"""Check checkpoint integrity and preserve PyTorch's safe loading boundary."""
import hashlib
import importlib
import io
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from drsynth_common import pyannote_models as models


def _artifact(path, contents):
    path.write_bytes(contents)
    return ("fixture/model", "a" * 40, path.name, hashlib.sha256(contents).hexdigest())


@pytest.fixture
def loaders(monkeypatch, tmp_path):
    downloads, calls = [], []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = download
    audio = ModuleType("pyannote.audio")
    audio.Pipeline = SimpleNamespace(from_pretrained=lambda *args, **kwargs: calls.append((args, kwargs)))
    audio.Model = SimpleNamespace(from_pretrained=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    return downloads, calls


def test_verify_download_revision_and_bytes(loaders, tmp_path):
    artifact = _artifact(tmp_path / "weights.bin", b"verified checkpoint")
    assert models._verified_file(artifact, "test") == str(tmp_path / "weights.bin")
    assert loaders[0] == [{"repo_id": artifact[0], "revision": artifact[1],
                           "filename": artifact[2], "token": "test"}]


def test_cached_checkpoint_mismatch_is_rejected(loaders, tmp_path):
    artifact = _artifact(tmp_path / "weights.bin", b"verified checkpoint")
    (tmp_path / "weights.bin").write_bytes(b"modified checkpoint")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        models._verified_file(artifact, "test")
    assert loaders[1] == []


@pytest.mark.parametrize("tampered", ["segmentation.bin", "speaker.bin", "xvec_transform.npz", "plda.npz"])
def test_nested_pipeline_checkpoints_are_verified_before_loading(loaders, tmp_path, monkeypatch, tampered):
    segmentation = _artifact(tmp_path / "segmentation.bin", b"segmentation")
    speaker = _artifact(tmp_path / "speaker.bin", b"speaker")
    transform = _artifact(tmp_path / "xvec_transform.npz", b"transform")
    plda = _artifact(tmp_path / "plda.npz", b"plda")
    config = _artifact(tmp_path / "config.yaml", b"pipeline:\n  params:\n    segmentation: fixture/model\n    embedding: fixture/model\n")
    monkeypatch.setattr(models, "_CONFIG", config)
    monkeypatch.setattr(models, "_SEGMENTATION", segmentation)
    monkeypatch.setattr(models, "_SPEAKER_EMBEDDING", speaker)
    monkeypatch.setattr(models, "_PLDA_TRANSFORM", transform)
    monkeypatch.setattr(models, "_PLDA", plda)
    models.load_diarization_pipeline(models.DIARIZATION_MODEL, "test")
    params = loaders[1][0][0][0]["pipeline"]["params"]
    assert params == {"segmentation": {"checkpoint": str(tmp_path / "segmentation.bin")},
                      "embedding": str(tmp_path / "speaker.bin"),
                      "plda": {"checkpoint": str(tmp_path)}}
    assert len(loaders[0]) == 5

    loaders[1].clear()
    (tmp_path / tampered).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        models.load_diarization_pipeline(models.DIARIZATION_MODEL, "test")
    assert loaders[1] == []


def test_enrollment_checkpoint_is_verified_before_loading(loaders, tmp_path, monkeypatch):
    artifact = _artifact(tmp_path / "enrollment.bin", b"embedding")
    monkeypatch.setattr(models, "_ENROLLMENT", artifact)
    models.load_enrollment_model("test")
    assert loaders[1] == [((str(tmp_path / "enrollment.bin"),), {"token": "test"})]


def test_custom_model_remains_an_explicit_operator_choice(loaders):
    models.load_diarization_pipeline("operator/custom-pipeline", "test")
    assert loaders[0] == []
    assert loaders[1] == [(("operator/custom-pipeline",), {"token": "test"})]


def test_service_imports_do_not_replace_torch_loader_or_expand_globals():
    code = """
import sys
from types import ModuleType, SimpleNamespace
torch = ModuleType('torch')
def original(*args, **kwargs):
    pass
added = []
torch.load = original
torch.serialization = SimpleNamespace(add_safe_globals=lambda values: added.extend(values))
sys.modules['torch'] = torch
import rtservice.engine
import whisperx_worker.pipeline
assert torch.load is original
assert added == []
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30)


class _UnsafeCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({str(self.marker)!r}).touch()",)


@pytest.mark.integration
@pytest.mark.parametrize("service_module", ["rtservice.engine", "whisperx_worker.pipeline"])
def test_torch_rejects_pickle_execution_after_service_imports(tmp_path, service_module):
    torch = pytest.importorskip("torch")
    importlib.import_module(service_module)

    marker = tmp_path / "checkpoint-executed"
    checkpoint = io.BytesIO()
    torch.save(_UnsafeCheckpoint(marker), checkpoint)
    for kwargs in ({}, {"weights_only": True}):
        checkpoint.seek(0)
        with pytest.raises(pickle.UnpicklingError):
            torch.load(checkpoint, **kwargs)
        assert not marker.exists()
