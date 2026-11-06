# Uploaded files

SamuraiBFF accepts an existing audio or video file as a session
(`POST /api/sessions/upload`). The BFF does not decode the file. It stores the
bytes unchanged and asks `recorder_worker` to convert them. After conversion
the session follows the same path as a live session.

## Flow

1. The BFF stores the file at
   `s3://<recordings-bucket>/<prefix>/<tenant_id>/uploads/<session_id>/source`.
2. The BFF publishes a `RecordingFinished` message to `recordings.uploaded`
   (key: session ID). `recording_url` points at the source object. Kafka
   headers carry the same stream controls as `audio.raw` (`x-outputs`,
   `x-final-tracks`, `x-store-recording`).
3. `recorder_worker` consumes the message on a separate thread and consumer
   group (`<KAFKA_GROUP_ID_RECORDER>-uploads`). A long conversion does not delay
   live session audio.
4. FFmpeg decodes the first audio stream to 16 kHz mono PCM16. The worker stores
   it with the normal recording writer (same key layout and WAV format as a live
   session) and publishes the normal `recordings.finished` event.
5. The worker deletes the source object. Final-track workers, Persistor and
   webhooks need no change.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `KAFKA_TOPIC_RECORDING_UPLOADED` | `recordings.uploaded` | Input topic |
| `RECORDER_UPLOADS_ENABLED` | `true` | Starts the upload consumer; also needs `S3_BUCKET` |
| `RECORDER_UPLOAD_MAX_BYTES` | `536870912` | Largest source object the worker downloads |
| `RECORDER_UPLOAD_MAX_DURATION_S` | `14400` | Longest decoded audio |
| `RECORDER_UPLOAD_TIMEOUT_S` | `1800` | FFmpeg run time limit |
| `RECORDER_FFMPEG_BIN` | `ffmpeg` | FFmpeg executable |

The source is always read from `S3_BUCKET`, also when
`RECORDING_STORAGE_BACKEND=local` writes the converted recording to disk.

## Failures

These problems are permanent. The worker logs the reason with the session and
tenant IDs, deletes the source and commits the message. No recording or
transcript is produced:

- the event does not point at `<tenant_id>/uploads/<session_id>/source` in
  `S3_BUCKET` (the worker does not read or delete the object);
- the source is missing or larger than `RECORDER_UPLOAD_MAX_BYTES`;
- FFmpeg fails, finds no audio, runs past the timeout, or the audio is longer
  than `RECORDER_UPLOAD_MAX_DURATION_S`;
- the message disables the final output.

A storage or Kafka outage is not permanent. The worker keeps the source and
retries the same message after five seconds.

## Security

The file is untrusted.

- FFmpeg runs as a list of arguments without a shell. It reads only the local
  temporary copy (`-protocol_whitelist file`) through allow-listed demuxers
  (`-format_whitelist`), so playlists cannot read other files or URLs.
- The temporary directory has a random name; the client file name is never
  used.
- The FFmpeg Debian package version is pinned in `recorder_worker/Dockerfile`.
  Bump `FFMPEG_VERSION` when Debian publishes a security fix.
- Size, duration and time limits bound the work per file. Set container CPU and
  memory limits for the recorder as well.
