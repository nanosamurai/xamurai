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
| `KAFKA_TOPIC_RECORDING_UPLOAD_FAILED` | `recordings.upload_failed` | Output topic for rejected uploads |
| `RECORDER_UPLOADS_ENABLED` | `true` | Starts the upload consumer; also needs `S3_BUCKET` |
| `RECORDER_UPLOAD_MAX_BYTES` | `536870912` | Largest source object the worker downloads |
| `RECORDER_UPLOAD_MAX_DURATION_S` | `14400` | Longest decoded audio |
| `RECORDER_UPLOAD_TIMEOUT_S` | `1800` | FFmpeg run time limit |
| `RECORDER_FFMPEG_BIN` | `ffmpeg` | FFmpeg executable |

The source is always read from `S3_BUCKET`, also when
`RECORDING_STORAGE_BACKEND=local` writes the converted recording to disk.

## Failures

These problems are permanent. No recording or transcript is produced. The
reason code is in brackets:

- the event does not point at `<tenant_id>/uploads/<session_id>/source` in
  `S3_BUCKET` (`foreign-source`; the worker does not read or delete the
  object), or has an invalid session ID, tenant ID or encoding
  (`invalid-session-id`, `invalid-tenant-id`, `invalid-event`);
- the source is missing (`missing-source`) or larger than
  `RECORDER_UPLOAD_MAX_BYTES` (`too-large`);
- FFmpeg fails (`decode-failed`), finds no audio (`no-audio`), runs past the
  timeout (`timeout`), or the audio is longer than
  `RECORDER_UPLOAD_MAX_DURATION_S` (`too-long`);
- the message disables the final output (`final-output-disabled`).

The worker logs the reason with the session and tenant IDs. When the event
names a valid session ID and tenant ID, it publishes a JSON event to
`recordings.upload_failed` (key: session ID) and waits for the broker ack:

```json
{"schema_version": 1, "session_id": "<uuid>", "tenant_id": "<tenant_id>",
 "reason": "decode-failed", "created_at": "2026-11-08T10:00:00.000Z"}
```

SamuraiBFF consumes it and sets the session status to `failed`. The event and
the logs contain IDs and the reason code only, never file content. After the
ack, the worker deletes the source and commits the message.

A storage or Kafka outage is not permanent, also when the failure event is not
acknowledged. The worker keeps the source and retries the same message after
five seconds.

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
