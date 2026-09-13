# Final track spike validation

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
`docs/final-tracks-spike.md`, mirrored in Nanodeploy. This run used a separate
validation database in the existing Compose infrastructure. Following explicit
approval, migration 018 and a separate local cleanup removed the retained DB's
obsolete experimental schema. All original transcript/recording content was
preserved, and the full Compose smoke passed again against that retained DB.
Local realtime and ordinary refinement images were also rebuilt from this branch.

Browser result tabs and durable per-track failure reporting remain later work.
