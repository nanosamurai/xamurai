# whisperx_worker diarization (refined speaker turns)

This document describes how `whisperx_worker` assigns speakers in **refined** output (`transcripts.refined`) and what configuration knobs exist to improve speaker turn quality.

## Background: why refined diarization can look “collapsed”

`whisperx_worker` runs per-slice (e.g. 60s) inference:

1. WhisperX ASR produces **text segments** with `[start_s, end_s]`.
2. Pyannote diarization produces **speaker time spans** for the same slice.
3. By default, `whisperx_worker` assigns **one speaker label per ASR segment** based on overlap.

For long refinement windows, WhisperX sometimes returns **very coarse ASR segments** (e.g. 1–2 segments for 30–60s of speech). In that case, the “one speaker per ASR segment” strategy collapses multiple back-and-forth turns into a single speaker label.

## Diarization-driven split mode (recommended when refined turns are wrong)

To improve speaker turn quality in refined output, `whisperx_worker` can use diarization turns to **drive segmentation**:

- run pyannote diarization on the slice
- merge diarization turns (merge tiny gaps, drop very short turns)
- re-transcribe each turn with WhisperX
- emit refined segments with `speaker=<diar speaker (or enrolled mapped label)>`

This produces multiple speaker turns within a single refinement window.

## RefinedEvent semantics (window-level)

Refined output is **window-based**:

- The worker buffers audio per session and cuts it into refinement windows
  (default `WHISPERX_SLICE_SECONDS=60`, overrideable per-stream via Kafka header
  `x-refinement-window-sec`).
- The worker emits **exactly one** Kafka message (`RefinedEvent`) per window slice.

Within that single `RefinedEvent`, speaker turns / silence-separated parts are represented
as `segments[]` (`SessionTranscriptSegment`). This avoids the previous ambiguity where one
refinement window could yield multiple `RefinedEvent` messages (one per segment), which
made refined behave too similarly to the session-level **final** transcript.

Important fields:
- `start_s`, `end_s` (window boundaries)
- `window_sec`, `slice_index`, `flush_reason`
- `segments[]` (per turn: `start_s`, `end_s`, `text`, `speaker`)
- `full_text` (concatenation convenience)

Backwards compatibility:
- `start_s`/`end_s` are set to the window boundaries.
- `text` is set to `full_text`.
- `speaker` is left empty; speaker labels are per segment (`segments[].speaker`).

### Configuration (environment variables)

Main toggle:

- `WHISPERX_DIAR_SPLIT_MODE` (`on|off`, default: `off`)

Guardrails / tuning:

- `WHISPERX_DIAR_SPLIT_MIN_TURN_SEC` (default: `0.7`)
  - diarization turns shorter than this are dropped
- `WHISPERX_DIAR_SPLIT_MERGE_GAP_SEC` (default: `0.15`)
  - consecutive turns by the same speaker with a gap <= this value are merged
- `WHISPERX_DIAR_SPLIT_MAX_TURNS` (default: `40`)
  - hard cap to avoid compute amplification
- `WHISPERX_DIAR_SPLIT_MAX_AUDIO_SEC` (default: `90.0`)
  - if a slice is longer than this, the worker falls back to the default overlap assignment

Notes:
- Split mode triggers only when diarization sees **2+ unique speakers** in the slice.
- This mode increases compute cost (more WhisperX transcribe calls per slice). The knobs above are meant to bound it.

## Security / abuse considerations

Split mode can amplify compute on highly fragmented diarization output.

We mitigate this via:
- minimum turn duration
- maximum number of turns
- maximum slice duration eligible for split mode

In multi-tenant environments, ensure resource quotas and scaling are configured appropriately.
