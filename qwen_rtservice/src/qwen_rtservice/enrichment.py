from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from drsynth_common.diarization_assign import (
    DiarizationSegment,
    TimeSegment,
    assign_speaker_by_overlap,
)

logger = logging.getLogger(__name__)


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
