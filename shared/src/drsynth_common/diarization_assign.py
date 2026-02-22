from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True, slots=True)
class TimeSegment:
    start_s: float
    end_s: float

    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))


@dataclass(frozen=True, slots=True)
class DiarizationSegment(TimeSegment):
    speaker: str


def _overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker_by_overlap(
    *,
    segments: Iterable[TimeSegment],
    diarization: Iterable[DiarizationSegment],
    default_speaker: str = "",
    min_coverage: float = 0.1,
) -> list[str]:
    """Assign a speaker label to each ASR segment based on diarization overlap.

    Args:
        segments: ASR segments (start/end)
        diarization: diarization segments (start/end/speaker)
        default_speaker: fallback speaker when no overlap
        min_coverage: require that best speaker overlap covers at least this
            fraction of the segment duration; otherwise return default.

    Returns:
        A list of speakers aligned with `segments`.
    """

    diar = list(diarization)
    out: list[str] = []

    for s in segments:
        dur = s.duration_s()
        if dur <= 0:
            out.append(default_speaker)
            continue

        best_spk: Optional[str] = None
        best_ov = 0.0

        for d in diar:
            ov = _overlap_s(s.start_s, s.end_s, d.start_s, d.end_s)
            if ov > best_ov:
                best_ov = ov
                best_spk = d.speaker

        if best_spk is None:
            out.append(default_speaker)
            continue

        cov = best_ov / dur if dur > 0 else 0.0
        if cov < float(min_coverage):
            out.append(default_speaker)
        else:
            out.append(best_spk)

    return out
