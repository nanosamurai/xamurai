from drsynth_common.diarization_assign import (
    DiarizationSegment,
    TimeSegment,
    assign_speaker_by_overlap,
)


def test_assign_speaker_by_overlap_basic():
    diar = [
        DiarizationSegment(start_s=0.0, end_s=1.0, speaker="A"),
        DiarizationSegment(start_s=1.0, end_s=2.0, speaker="B"),
    ]

    segs = [
        TimeSegment(start_s=0.2, end_s=0.8),
        TimeSegment(start_s=1.2, end_s=1.8),
    ]

    out = assign_speaker_by_overlap(segments=segs, diarization=diar, default_speaker="")
    assert out == ["A", "B"]


def test_assign_speaker_by_overlap_min_coverage_default():
    diar = [
        DiarizationSegment(start_s=0.0, end_s=0.05, speaker="A"),
    ]

    segs = [
        TimeSegment(start_s=0.0, end_s=1.0),
    ]

    out = assign_speaker_by_overlap(segments=segs, diarization=diar, default_speaker="?", min_coverage=0.2)
    assert out == ["?"]
