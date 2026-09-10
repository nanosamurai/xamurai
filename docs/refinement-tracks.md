# Bounded refinement tracks

`recorder_worker.refinement_windows` consumes `audio.raw` once in group
`refinement-window-producer`, writes immutable fixed windows to S3, and emits
`RefinementWindow` on `audio.refinement-windows`. Run the existing
`finalizer_worker.track_worker` with `ASR_TRACK_STAGE=refined` and
`REFINEMENT_TRACK_ID` / `REFINEMENT_PROFILE_ID`. Each track/profile has its own
group; replicas share it. The real profile is `whisperx-medium-refined-r1`, using
the same pinned WhisperX/VAD/alignment/diarization composite as final tracks.
The test profile `test-refined-r1` requires
`REFINEMENT_TRACK_TEST_PROFILE_ENABLED=true` and is not another speech model.

The frozen BFF plan adds `refinement_tracks` and `refinement_window_samples`.
`x-refinement-track-ids` must agree with the canonical `x-asr-plan` header and
the typed source plan. Empty final selection is valid for refinement-only use.
The legacy refinement loops skip planned refinement input before inference.

Windows own half-open sample ranges with no overlap/context. Source generation
is the plan ID. `unit_id` is `fixed-<window_samples>:<start>:<end>`; a run UUIDv5
uses canonical JSON `[tenant,session,plan,"refined",track,profile]`, and result
UUIDv5 uses that run plus `<unit_id>:1`. Source UUIDv5 uses the plan plus unit.
Revision 1 is immutable; retry never creates a newer revision. Future replacement
must target the same run/unit and cannot supersede another track.

`transcripts.refined-tracks` carries the existing bounded result envelope
(`FinalTrackResult` retains its wire type name) with `stage=refined` and the
typed `refinement_window`. Successful primary windows additionally publish
`RefinedEvent` on `transcripts.refined`. Segment/word times are absolute within
the session. Primary headers carry the accepted outcome for idempotent SQL;
secondary and failed results never enter the compatibility stream. Legacy
`flush_reason=idle` represents either EOF or idle closure; canonical outcomes
retain the exact reason. No new inference service or scheduler is introduced.

Open sessions pin their initial Kafka offsets, including across interleaved
sessions in a partition. Rebalance/restart reconstructs from retained raw audio.
Each window and close boundary has a conditional immutable S3 manifest, so a
retry republishes identical bytes. A normal BFF EOF follows all accepted audio.
`REFINEMENT_IDLE_SECONDS` (default 30) is a fallback only after catching up with
Kafka. A closed generation cannot accept more audio. Shutdown/rebalance never
flushes partial buffers. One complete window is held until another sample or
closure so exact-multiple sessions retain a terminal window.

Limits: mono PCM16 16 kHz, retained audio, 10–600 second fixed windows, ten-minute
sessions, at most 32 open sessions per producer. Sequence must start at 1 and
remain contiguous. Malformed input/storage failure stops without committing
uncertain work. Raw retention must exceed session duration plus outage/recovery
time; this spike makes no unbounded-session throughput claim.

S3 uses the configured bucket, `refinement-windows/<tenant>/<session>/` sources
and `refined-tracks/` outcomes. Workers validate exact keys, size, format and
digest before inference. Temporary WAVs are removed. Shared source deletion,
derived-artifact retention and orphan cleanup remain lifecycle qualification
work; use consented local fixtures until that work is implemented.

Run `pytest -q tests/test_final_tracks_unit.py tests/test_refinement_tracks_unit.py`.
These cover restart/replay, publication failure, interleaved commit frontiers,
absolute timestamps, track failure isolation and tenant/plan/sequence tampering.
Regenerate the additive Python messages with
`buf generate proto --template proto/buf.gen.python.yaml`; the pinned compiler
keeps smoke consumers compatible with protobuf 6.x. No RPC methods changed.
Anonymous speaker labels are local to each window; cross-window identity is
not inferred from a repeated label.
