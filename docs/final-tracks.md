# Opt-in completed-recording final tracks

The first asynchronous profile is `whisperx-medium-final-r1`: WhisperX 3.8.6,
the pinned faster-whisper medium weights, the bundled pyannote VAD, pinned
Czech/English alignment, and the existing pinned pyannote 3.1 diarization loader.
The descriptor lives in `finalizer_worker/whisperx_profile.py`. Other alignment
languages retain ASR with an explicit degradation. Enrollment is disabled.

`BatchTrack.describe()` describes the configured composite. `process(AudioInput,
RunContext)` returns a normalized transcript, actual capabilities, provenance,
and degradations. The current unit is the complete recording; refinement can
later use the same audio/source and run identities with bounded sample windows.
No refinement execution or distributed speech-stage graph is introduced here.

The BFF freezes one plan per session. `x-asr-plan` contains bounded canonical JSON,
`x-asr-plan-id` its UUID, and `x-final-track-ids` the selected logical tracks.
The recorder copies that plan into the additive `RecordingFinished.final_plan`
and creates `source` with a new artifact UUID, SHA-256, byte/sample counts and S3
version where available. Consumers verify typed fields against the headers.

Each `(final, track_id, profile_id)` deployment owns a group; replicas share that
group. Every group consumes `recordings.finished`, skips unselected recordings
before storage/model access, and publishes terminal `FinalTrackResult` protobufs
to **`transcripts.final-tracks`**. Schema version is in the envelope, never the
topic name. This shared-topic/filtering arrangement is intended to remain.
Only primary successes also publish the unchanged `SessionTranscript` body to
`transcripts.final`, with additive result/source headers for SQL deduplication.
Legacy finalizers skip planned inputs. Missing plan headers mean legacy only
when the typed source/plan are also absent.

The recorder requires S3, mono 16 kHz PCM16, retained audio, and at most 600 seconds
for opted-in sessions. Set `FINAL_TRACKS_ENABLED=true` on the recorder and the
generic finalizer; set `FINAL_TRACK_ID` and `FINAL_PROFILE_ID` on each deployment.
The fixed source prefix is `recordings`, overridable consistently through
`FINAL_RECORDING_PREFIX`; `S3_BUCKET` and existing S3/Kafka TLS configuration apply.
The test profile `test-final-r1` additionally requires
`FINAL_TRACK_TEST_PROFILE_ENABLED=true`; it is not a second real provider.

Workers keep polling Kafka during inference, bound local inference to one job,
retry provider failures at most three times, and publish failure outcomes when
those attempts fail. A conditional S3 outcome manifest chooses the first durable
outcome. Replay republishes those exact bytes without inference. Kafka output
acknowledgements precede the input commit. Execution before the manifest is
at least once; neither model execution nor cross-topic delivery is exactly once.
Malformed identity/routing and storage outages do not advance the input offset.

Source downloads allow only the configured bucket and the exact tenant/session/
artifact key, enforce size/sample/digest bounds, and remove temporary audio.
Track workers never delete the shared recording. Use isolated, consented local
fixtures; coordinated retention cleanup and recording-assembly crash recovery
remain outside this completed-recording spike. Per-attempt orphan artifacts
require cleanup with the isolated evaluation data. Do not enable broad use of
the feature before that lifecycle work is defined.

Qualification passed (2026-09-10): 17 focused contract/replay tests and all 128
tests selected by `-m 'not integration'` pass. The full Windows selection
hits a native dependency crash in the existing realtime model initialization;
the final-track model qualification runs in the Linux service image. The
consented Czech fixture passed real WhisperX, alignment and diarization: six
segments, all three timing/speaker capabilities, no degradations. Generated
three-second silence succeeded empty. The first run including model loading
took 103.145 seconds; PyTorch reported 2,976,341,504 peak allocated GPU bytes
during processing (this excludes CTranslate2 allocations and is not total GPU
usage). The complete Kafka/S3/SQL/Compose flow passed with a real primary,
independent successful/failed test tracks, two replicas on two source partitions,
exact replay bytes, stable row counts, database outage recovery, and process exit
after manifest creation but before Kafka publication. The existing recording
Range API also passed. The normal WhisperX function produced six segments,
31 words and six speaker-labelled segments on the same fixture. Its cached
initial/warm runs took 16.657/1.502 seconds; the new worker measured
16.171/1.459 seconds. This is a compatibility sample, not a benchmark.

Publication scans recognize only the known synthetic UUID tails in the shared
final-track vector, scoped to that file and rule; other values remain scanned.

The lightweight pull-request CI gate includes `tests/test_final_tracks_unit.py`
alongside the existing realtime suites, using fake model and storage adapters.

