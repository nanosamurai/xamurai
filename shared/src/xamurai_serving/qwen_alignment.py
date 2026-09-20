"""Validate Qwen word timings and retain the original transcript characters."""
import math
from dataclasses import dataclass, replace

from drsynth_common.diarization_assign import TimeSegment


@dataclass(frozen=True, slots=True)
class AlignedWord(TimeSegment):
    text: str


def aligned_words(raw_words, transcript: str, duration_s: float) -> list[AlignedWord]:
    """Clip valid alignment units to audio bounds and restore spacing/punctuation."""
    words = []
    previous_start = 0.0
    for item in raw_words:
        start_s, end_s = float(item.start_time), float(item.end_time)
        text = str(item.text or "").strip()
        if not text or not math.isfinite(start_s) or not math.isfinite(end_s):
            continue
        start_s = max(0.0, min(duration_s, start_s))
        end_s = max(start_s, min(duration_s, end_s))
        if end_s <= start_s:
            continue
        if start_s < previous_start:
            raise ValueError("Nonmonotonic Qwen alignment")
        words.append(AlignedWord(start_s=start_s, end_s=end_s, text=text))
        previous_start = start_s
    if not words:
        raise ValueError("Empty Qwen alignment")

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
