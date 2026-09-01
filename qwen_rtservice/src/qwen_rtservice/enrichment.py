from __future__ import annotations

import importlib.metadata
import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from drsynth_common.diarization_assign import (
    DiarizationSegment,
    TimeSegment,
    assign_speaker_by_overlap,
)
from drsynth_common.pyannote_telemetry import disable_pyannote_telemetry


disable_pyannote_telemetry()

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True, slots=True)
class AlignedWord(TimeSegment):
    text: str


@dataclass(frozen=True, slots=True)
class EnrichedSegment(TimeSegment):
    text: str
    speaker: str


class EnrichmentFailure(RuntimeError):
    """Sanitized enrichment failure classification safe for service logs."""

    def __init__(self, stage: str, cause_type: str) -> None:
        super().__init__(f"Qwen epoch enrichment failed during {stage}")
        self.stage = stage
        self.cause_type = cause_type


class AlignmentBackend(Protocol):
    alignment_languages: tuple[str, ...]

    def align(self, pcm16: np.ndarray, text: str, language: str): ...


class DiarizationBackend(Protocol):
    runtime: str

    def diarize(self, pcm16: np.ndarray) -> tuple[DiarizationSegment, ...]: ...


class FinalEnricher(Protocol):
    supported_languages: tuple[str, ...]

    def enrich(
        self,
        pcm16: np.ndarray,
        text: str,
        language: str,
        *,
        epoch_number: int,
    ) -> tuple[EnrichedSegment, ...]: ...


class PyannoteDiarizer:
    """Load the fixed pyannote pipeline and diarize one bounded Qwen epoch."""

    def __init__(self) -> None:
        import torch
        import yaml
        from pyannote.audio import Pipeline

        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("Set HF_TOKEN for the fixed pyannote diarization profile")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen realtime diarization requires an NVIDIA CUDA device")

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
            "Qwen epoch diarization completed turns=%d elapsed_seconds=%.3f",
            len(turns),
            time.monotonic() - started,
        )
        return turns


class QwenEpochEnricher:
    """Align committed Qwen text, join words to pyannote turns, and coalesce it."""

    def __init__(self, aligner: AlignmentBackend, diarizer: DiarizationBackend) -> None:
        self._aligner = aligner
        self._diarizer = diarizer
        self.supported_languages = tuple(aligner.alignment_languages)

    def enrich(
        self,
        pcm16: np.ndarray,
        text: str,
        language: str,
        *,
        epoch_number: int,
    ) -> tuple[EnrichedSegment, ...]:
        duration_s = pcm16.size / 16000.0
        started = time.monotonic()
        logger.info(
            "Qwen epoch enrichment started audio_samples=%d transcript_characters=%d language=%s",
            pcm16.size,
            len(text),
            language,
        )
        try:
            raw_words = self._aligner.align(pcm16, text, language)
        except Exception as exc:
            raise EnrichmentFailure("alignment", type(exc).__name__) from exc
        words: list[AlignedWord] = []
        previous_start = 0.0
        for item in raw_words:
            raw_start_s = float(item.start_time)
            raw_end_s = float(item.end_time)
            unit = str(item.text or "").strip()
            if not unit or not math.isfinite(raw_start_s) or not math.isfinite(raw_end_s):
                continue
            start_s = max(0.0, min(duration_s, raw_start_s))
            end_s = max(start_s, min(duration_s, raw_end_s))
            if end_s <= start_s:
                continue
            if start_s < previous_start:
                raise EnrichmentFailure("alignment", "NonMonotonicResult")
            words.append(AlignedWord(start_s=start_s, end_s=end_s, text=unit))
            previous_start = start_s
        if not words:
            raise EnrichmentFailure("alignment", "EmptyResult")

        try:
            words = _restore_transcript_text(words, text)
        except Exception as exc:
            raise EnrichmentFailure("alignment", type(exc).__name__) from exc
        logger.info(
            "Qwen epoch alignment completed words=%d elapsed_seconds=%.3f",
            len(words),
            time.monotonic() - started,
        )
        try:
            turns = self._diarizer.diarize(pcm16)
        except Exception as exc:
            raise EnrichmentFailure("diarization", type(exc).__name__) from exc
        if not turns:
            raise EnrichmentFailure("diarization", "EmptyResult")
        try:
            speakers = assign_speaker_by_overlap(
                segments=words,
                diarization=turns,
                default_speaker="",
                min_coverage=0.1,
            )
        except Exception as exc:
            raise EnrichmentFailure("speaker_join", type(exc).__name__) from exc

        segments: list[EnrichedSegment] = []
        for word, raw_speaker in zip(words, speakers):
            if not raw_speaker:
                word_midpoint = (word.start_s + word.end_s) / 2
                raw_speaker = min(
                    turns,
                    key=lambda turn: min(
                        abs(word_midpoint - turn.start_s),
                        abs(word_midpoint - turn.end_s),
                    ),
                ).speaker
            speaker = f"EPOCH_{epoch_number:04d}/{raw_speaker}"
            if segments and segments[-1].speaker == speaker:
                previous = segments[-1]
                segments[-1] = EnrichedSegment(
                    start_s=previous.start_s,
                    end_s=max(previous.end_s, word.end_s),
                    text=f"{previous.text}{word.text}",
                    speaker=speaker,
                )
            else:
                segments.append(
                    EnrichedSegment(
                        start_s=word.start_s,
                        end_s=word.end_s,
                        text=word.text,
                        speaker=speaker,
                    )
                )
        if not segments or any(segment.end_s <= segment.start_s for segment in segments):
            raise EnrichmentFailure("speaker_join", "InvalidRange")
        return tuple(replace(segment, text=segment.text.strip()) for segment in segments)


def _restore_transcript_text(words: list[AlignedWord], transcript: str) -> list[AlignedWord]:
    """Restore spaces and punctuation removed by the aligner's tokenizer."""
    normalized_characters: list[str] = []
    original_indexes: list[int] = []
    for original_index, character in enumerate(transcript):
        if character.isalnum():
            folded = character.casefold()
            normalized_characters.extend(folded)
            original_indexes.extend([original_index] * len(folded))
    normalized_transcript = "".join(normalized_characters)

    positions: list[int] = []
    normalized_cursor = 0
    for word in words:
        normalized_word = "".join(
            character.casefold() for character in word.text if character.isalnum()
        )
        match_index = normalized_transcript.find(normalized_word, normalized_cursor)
        if not normalized_word or match_index < 0:
            raise RuntimeError("forced-alignment units do not match the committed transcript")
        positions.append(original_indexes[match_index])
        normalized_cursor = match_index + len(normalized_word)

    restored: list[AlignedWord] = []
    for index, word in enumerate(words):
        text_start = 0 if index == 0 else positions[index]
        text_end = positions[index + 1] if index + 1 < len(words) else len(transcript)
        restored.append(replace(word, text=transcript[text_start:text_end]))
    return restored
