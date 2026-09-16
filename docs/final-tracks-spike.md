# Final track spike validation

The worker's `main(transcribe=None, model=MODEL)` entry point accepts a pipeline
returning the existing text/segment dictionaries. The default lazily loads
WhisperX with alignment enabled. Model-specific entry points share selection,
storage, acknowledged publication and replay handling without importing WhisperX.
Test-only synthetic callers pass their inference function directly to `main`.

Validated on 2026-09-13 on `implement-lean-final-tracks`.

- Lightweight regression suite: 71 passed, including stream controls, serving
  admission, realtime segmentation/isolation and provider adapters. The command
  uses the README's lightweight test selection plus `test_stream_controls_unit.py`.
- Rebuilt `xamurai-finalizer-worker:lean-tracks` and ran both the production
  recorder and finalizer from it in Nanosamurai's local Compose stack.
- Full Compose smoke passed: real WhisperX `medium` inference and alignment,
  synthetic peer track, silence, selection/skip, input/output replay, independent
  worker failure/restart, source download recovery, DB retry, tenant denial, BFF text/segments/playback,
  retained shared audio, single-track deletion and no local JSON sidecar.

The second track replaces inference only in a mounted test script; it exercises
the production worker loop without adding a production synthetic mode. This
proves multi-track plumbing, not another model's quality. Test ports bind only
to localhost and build contexts exclude credentials and scratch files.

The repeatable overlay, assertions and runbook live in Nanosamurai at
`docker-compose.final-tracks-smoke.yml`, `smoke-tests/final-tracks/` and
`docs/final-tracks-spike.md`, mirrored in Nanodeploy. The rollout and full smoke
now target the original Compose project `nanosamurai` and its Postgres 18 database,
with migrations 017/018 applied there. Its 117 original transcripts and 51
recording records were preserved; obsolete metadata was backed up before cleanup.
Realtime, ordinary refinement, recorder and finalizer use local images from this
branch. The last active original WhisperX final consumer group is retained.
The earlier experimental Compose project is stopped.

Browser result tabs and durable per-track failure reporting remain later work.
