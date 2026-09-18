# Parakeet refinement

`parakeet_worker.refinement` runs the same pinned Parakeet TDT v3 / Sortformer
pipeline as the [finalizer](parakeet-finalizer.md) on semi-batch audio windows.
Build with `docker buildx bake --load parakeet-refinement`; its four-line
Dockerfile reuses the finalizer image and changes the entrypoint/track default.
The publication matrix includes `xamurai-parakeet-refinement`.

The shared `xamurai_serving.refinement` accepts a warm callable returning text
and segment dictionaries, as finalization does. WhisperX adapts its existing
unaligned segment tuples; Parakeet passes its existing pipeline directly.
There is one buffering/recovery implementation in `refinement_runtime`.
No Torch, alignment model, new topic, protobuf field or database migration is
required for Parakeet. Rebuild both WhisperX images after this shared-code move.

## Configuration and contract

The image defaults `REFINEMENT_TRACK_ID=parakeet`, yielding consumer group
`refinement.parakeet`. `KAFKA_GROUP_ID` overrides it; replicas of a track share
one group, different tracks must use different groups. WhisperX retains its
existing group. Existing Kafka TLS, topic and tracing settings apply.

- `REFINEMENT_SLICE_SECONDS`: default 60, overridden by the session's existing
  `x-refinement-window-sec` header.
- `REFINEMENT_IDLE_SECONDS`: default 30; flush the short tail after input idles.
- `REFINEMENT_READY_QUEUE_MAX`: default 256 queued windows, one inference thread.
- Existing `WHISPERX_SLICE_SECONDS`, `WHISPERX_IDLE_SECONDS` and
  `WHISPERX_READY_QUEUE_MAX` remain fallbacks for compatibility.
- `PARAKEET_GPU`: same GPU selection as the finalizer. No HF token is required.

Only selected `x-refinement-tracks` consume `audio.raw`; `x-outputs` must include
refinement. Omitted selection still means WhisperX. Each window emits one
`RefinedEvent` with track `parakeet` and model `nvidia/parakeet-tdt-0.6b-v3`.
Window bounds come from cumulative PCM samples. Both segment and word times
are shifted into session coordinates, including idle tails. Kafka must
acknowledge publication before buffered work can finish. The runtime preserves
active-session offsets across skips, failures and rebalances; migration 019
deduplicates replay. See [refinement recovery limits](refinement-tracks-spike.md).
Shared tracing uses `refinement.slice` with a track attribute and the existing
`kafka.produce transcripts.refined` child span.

Sortformer starts fresh for each window: `SPEAKER_00` in different windows need
not be the same person. No cross-window speaker matching or enrolled names are
claimed. Language is detected by Parakeet; the session hint stays metadata.
Full-session finalization remains independently selectable.

## Validation and security

The real-model Compose smoke in Nanosamurai checks both real refinement tracks,
two 10-second windows and a 3.125-second tail, live delivery, session-relative
word timing, model identity, filtered history, input/conflicting-output replay,
individual/default/disabled selections, silence and tenant denial. Existing
native integration tests exercise repeated independent calls and invalid audio.
The general refinement smoke still supplies synthetic inference directly to
the shared runtime for failure, replica and rebalance coverage.

Model revisions, hashes, native runtime and attribution remain those of the
finalizer. The worker runs as UID 10003, publishes no host port, needs no S3
credentials, and removes temporary WAVs even on failure. Transcript contents
are not added to logs. Existing audio-retention/replay and idle-resume limits
still apply; a new session should be used after completion. Qualification with
short public fixtures does not establish capacity for arbitrary concurrency.

Local validation on 2026-09-17: 122 lightweight regression checks and both
real native Parakeet integration tests passed. Rebuilt Parakeet/WhisperX images
passed Nanosamurai's real refinement and finalizer Compose probes, including
replay, silence, live/history routing and playback. Worker Python is 91 lines
smaller after the shared-runtime extraction. Deployment evidence belongs to
`nanodeploy/docs/asr-platform/parakeet-refinement-spike.md`.
The existing shared-runtime Compose smoke also passed owner loss, interleaved
reconstruction, replicas, bounded queues, independent failure/restart, database
retry and tenant rejection with the updated inference callback.
