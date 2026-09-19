"""Pinned Qwen/vLLM model loading and pyannote shared by realtime and workers."""
import hashlib
import importlib.metadata
import logging
import os
import time
from pathlib import Path

import numpy as np
from drsynth_common.diarization_assign import DiarizationSegment
from drsynth_common.pyannote_telemetry import disable_pyannote_telemetry

disable_pyannote_telemetry()
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
MODEL_REVISION = "c4468bdb552ddc559e464f6081e22dd4034f2e68"
MODEL_DIGEST = "sha256:79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
ALIGNMENT_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNMENT_MODEL_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
ALIGNMENT_MODEL_DIGEST = "sha256:47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7"
SAMPLE_RATE = 16000
AUDIO_TOKENS_PER_SECOND = 13
KV_CACHE_BYTES_PER_TOKEN = 114688
QWEN_LANGUAGE_CODES = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian": "it",
    "Korean": "ko",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Turkish": "tr",
    "Hindi": "hi",
    "Malay": "ms",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Polish": "pl",
    "Czech": "cs",
    "Filipino": "fil",
    "Persian": "fa",
    "Greek": "el",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Macedonian": "mk",
}
QWEN_LANGUAGE_NAMES = {code: name for name, code in QWEN_LANGUAGE_CODES.items()}


def _verified_snapshot(repo_id: str, revision: str, expected_digest: str) -> str:
    """Download one immutable model snapshot and verify its safetensors payload."""
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(repo_id=repo_id, revision=revision)
    weights = Path(model_path) / "model.safetensors"
    if not weights.is_file():
        raise RuntimeError(f"pinned model artifact is incomplete: {repo_id}")
    digest = hashlib.sha256()
    with weights.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if f"sha256:{digest.hexdigest()}" != expected_digest:
        raise RuntimeError(
            f"pinned model artifact digest does not match the service profile: {repo_id}"
        )
    return model_path


def bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def load_model(*, max_new_tokens=1024, max_model_len=4096, kv_cache_mib=512,
               batch_size=1, align=False):
    """Load the pinned vLLM profile, optionally with its forced aligner."""
    import torch
    from qwen_asr import Qwen3ASRModel

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen requires an NVIDIA CUDA device")
    model_path = _verified_snapshot(MODEL_ID, MODEL_REVISION, MODEL_DIGEST)
    aligner_path = (_verified_snapshot(ALIGNMENT_MODEL_ID, ALIGNMENT_MODEL_REVISION,
                                      ALIGNMENT_MODEL_DIGEST) if align else None)
    return Qwen3ASRModel.LLM(
        model=model_path, max_new_tokens=max_new_tokens, max_model_len=max_model_len,
        kv_cache_memory_bytes=kv_cache_mib * 1024 * 1024,
        max_num_seqs=batch_size, max_inference_batch_size=batch_size,
        limit_mm_per_prompt={"audio": 1}, tensor_parallel_size=1,
        trust_remote_code=False, disable_log_stats=True,
        forced_aligner=aligner_path,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0",
                               "trust_remote_code": False},
    )


DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
DIARIZATION_MODEL_REVISION = "84fd25912480287da0247647c3d2b4853cb3ee5d"
DIARIZATION_CONFIG_DIGEST = (
    "sha256:04ad9cd59a93c3a7c754200ecc9e1c4ba87bf1657ef8a4debf7555e711daeeda"
)
SEGMENTATION_MODEL_ID = "pyannote/segmentation-3.0"
SEGMENTATION_MODEL_REVISION = "e66f3d3b9eb0873085418a7b813d3b369bf160bb"
SEGMENTATION_MODEL_DIGEST = (
    "sha256:da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298"
)
EMBEDDING_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBEDDING_MODEL_REVISION = "837717ddb9ff5507820346191109dc79c958d614"
EMBEDDING_MODEL_DIGEST = "sha256:366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56"


def _verified_hf_file(
    repo_id: str,
    revision: str,
    filename: str,
    expected_digest: str,
    token: str,
) -> str:
    """Download and verify one immutable Hugging Face artifact."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
        )
    )
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if f"sha256:{digest.hexdigest()}" != expected_digest:
        raise RuntimeError(f"pinned model artifact digest does not match: {repo_id}/{filename}")
    return str(path)


class PyannoteDiarizer:
    """Load the fixed pyannote pipeline and diarize one Qwen recording or window."""

    def __init__(self) -> None:
        import torch
        import yaml
        from pyannote.audio import Pipeline

        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("Set HF_TOKEN for the fixed pyannote diarization profile")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen diarization requires an NVIDIA CUDA device")

        config_path = _verified_hf_file(
            DIARIZATION_MODEL_ID,
            DIARIZATION_MODEL_REVISION,
            "config.yaml",
            DIARIZATION_CONFIG_DIGEST,
            token,
        )
        segmentation_path = _verified_hf_file(
            SEGMENTATION_MODEL_ID,
            SEGMENTATION_MODEL_REVISION,
            "pytorch_model.bin",
            SEGMENTATION_MODEL_DIGEST,
            token,
        )
        embedding_path = _verified_hf_file(
            EMBEDDING_MODEL_ID,
            EMBEDDING_MODEL_REVISION,
            "pytorch_model.bin",
            EMBEDDING_MODEL_DIGEST,
            token,
        )
        with Path(config_path).open("r", encoding="utf-8") as source:
            config = yaml.safe_load(source)
        pipeline_params = config.get("pipeline", {}).get("params", {})
        if pipeline_params.get("segmentation") != SEGMENTATION_MODEL_ID:
            raise RuntimeError("pinned pyannote pipeline has an unexpected segmentation model")
        if pipeline_params.get("embedding") != EMBEDDING_MODEL_ID:
            raise RuntimeError("pinned pyannote pipeline has an unexpected embedding model")
        pipeline_params["segmentation"] = {"checkpoint": segmentation_path}
        pipeline_params["embedding"] = embedding_path
        pipeline = Pipeline.from_pretrained(config, token=token)
        if pipeline is None:
            raise RuntimeError("pinned pyannote diarization pipeline could not be loaded")
        pipeline.to(torch.device("cuda"))
        self._pipeline = pipeline
        self.runtime = f"pyannote-audio=={importlib.metadata.version('pyannote-audio')}"

    def diarize(self, pcm16: np.ndarray) -> tuple[DiarizationSegment, ...]:
        import torch

        wave = pcm16.astype(np.float32) / 32768.0
        started = time.monotonic()
        output = self._pipeline(
            {
                "waveform": torch.from_numpy(wave.copy()).unsqueeze(0),
                "sample_rate": 16000,
            }
        )
        annotation = getattr(output, "speaker_diarization", output)
        if not hasattr(annotation, "itertracks"):
            raise RuntimeError("pyannote returned no speaker annotation")
        turns = tuple(
            sorted(
                (
                    DiarizationSegment(
                        start_s=float(turn.start),
                        end_s=float(turn.end),
                        speaker=str(speaker),
                    )
                    for turn, _, speaker in annotation.itertracks(yield_label=True)
                    if float(turn.end) > float(turn.start) and str(speaker).strip()
                ),
                key=lambda turn: (turn.start_s, turn.end_s, turn.speaker),
            )
        )
        logger.info(
            "Qwen diarization completed turns=%d elapsed_seconds=%.3f",
            len(turns),
            time.monotonic() - started,
        )
        return turns
