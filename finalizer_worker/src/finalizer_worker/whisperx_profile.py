"""Immutable model artifacts and bounded runtime settings for the first profile."""

from pathlib import Path
from drsynth_common.final_tracks import ContractError, file_digest

PROFILE_ID = "whisperx-medium-final-r1"
ASR = {"model_id": "Systran/faster-whisper-medium",
       "revision": "08e178d48790749d25932bbc082711ddcfdfbc4f",
       "filename": "model.bin",
       "sha256": "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae"}
ALIGNMENT = {
    "cs": {"model_id": "comodoro/wav2vec2-xls-r-300m-cs-250",
           "revision": "73e2b9004f35d3ca79316a624e7edc3cdaa45d40",
           "filename": "model.safetensors",
           "sha256": "99376277738caade67bf57b7dfab362b3172323b4724c0db4c51a56254340099"},
    "en": {"model_id": "facebook/wav2vec2-base-960h",
           "revision": "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
           "filename": "model.safetensors",
           "sha256": "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a"},
}
VAD_SHA256 = "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"
RUNTIME = {"whisperx": "3.8.6", "faster-whisper": "1.2.1", "pyannote.audio": "4.0.7",
           "torch": "2.8.0+cu128", "torchaudio": "2.8.0+cu128",
           "transformers": "4.57.6", "huggingface-hub": "0.36.2"}


def model_snapshot(descriptor):
    """Load only pinned weights/configuration, with an independently checked digest."""
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(descriptor["model_id"], revision=descriptor["revision"],
                                  allow_patterns=[descriptor["filename"], "*.json", "vocabulary.*"]))
    if file_digest(path / descriptor["filename"]) != descriptor["sha256"]:
        raise ContractError("model_digest_mismatch")
    return str(path)


def descriptor():
    """Describe configured provenance; execution capabilities are reported separately."""
    return {"profile_id": PROFILE_ID, "runtime": RUNTIME, "asr": ASR,
            "vad": {"backend": "whisperx-pyannote", "sha256": VAD_SHA256,
                    "onset": 0.5, "offset": 0.363, "chunk_size_s": 30},
            "alignment": ALIGNMENT, "alignment_min_coverage": 0.7,
            "diarization": {"model_id": "pyannote/speaker-diarization-3.1",
                            "revision": "84fd25912480287da0247647c3d2b4853cb3ee5d"},
            "compute_type": "float16", "device": "cuda", "batch_size": 16,
            "speaker_assignment": "segment-overlap", "enrollment": False}
