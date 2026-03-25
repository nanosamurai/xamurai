# Word-level timing (karaoke / playback highlighting)

This document describes how **word-level timestamps** are produced and carried through the system so the UI can implement karaoke-style highlighting during playback.

## Where the data lives

### Kafka (`transcripts.final`)

The `finalizer_worker` publishes a `SessionTranscript` protobuf message to the `transcripts.final` topic.

Each transcript segment can optionally contain word-level timing:

```proto
message SessionTranscriptSegment {
  double start_s = 1;
  double end_s   = 2;
  string text    = 3;
  string speaker = 4;
  repeated WordAlignment words = 5; // optional
}

message WordAlignment {
  double start_s = 1;
  double end_s   = 2;
  string text    = 3;
}
```

`words[]` is produced when WhisperX alignment succeeds; otherwise it may be empty.

### Sidecar JSON artifact (next to WAV)

For `file://...` recordings, `finalizer_worker` writes a JSON transcript next to the WAV. The JSON mirrors the protobuf shape.

Example segment:

```json
{
  "start_s": 12.34,
  "end_s": 14.10,
  "text": "hello how are you",
  "speaker": "SPEAKER_00",
  "words": [
    {"start_s": 12.34, "end_s": 12.60, "text": "hello"},
    {"start_s": 12.61, "end_s": 12.80, "text": "how"},
    {"start_s": 12.81, "end_s": 12.95, "text": "are"},
    {"start_s": 12.96, "end_s": 13.10, "text": "you"}
  ]
}
```

### Postgres (`session_transcripts.segments` jsonb)

Persistence is handled by **samuraipersistor** (separate repo). It consumes `transcripts.final` and stores the transcript into the `session_transcripts` table.

The `segments` jsonb column should store the same structure, i.e. each segment object should include a `words` array when present.

## Implementation notes

- Word timings come from `whisperx.align()`.
- The pipeline currently assigns **speaker per segment** (diarization overlap on segment spans). Words do not carry speaker info.
- Payload size can grow for long sessions (more data per segment). We rely on Kafka zstd compression, but we should watch broker message size limits if sessions become very long.

## Dependency: samuraipersistor update required

Xamurai only publishes to Kafka; to make word timing available in the UI via DB APIs, `samuraipersistor` must be updated to:

1. Read `SessionTranscriptSegment.words` from the protobuf.
2. Include it in the serialized JSON written into `session_transcripts.segments`.
