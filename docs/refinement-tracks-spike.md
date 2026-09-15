# Refinement track qualification

Validated on 2026-09-14 on `implement-lean-refinement-tracks`, branched from
`implement-lean-final-tracks`.

72 lightweight regression/publication checks passed.

The full Nanosamurai Compose smoke passed on its original Postgres 18 stack:
real WhisperX medium and test-only synthetic inference; two tracks with 10-second
adjacent windows and a 3.125-second tail; live BFF delivery and history filtering;
selection snapshots/defaults/skips; first-result replay; interleaved sessions;
same-track replicas and Kafka owner loss; a one-job queue; independent worker
failure/restart; database retry and tenant rejection. An ordinary realtime,
refined and final session plus recording/range playback also passed after
restoring the normal local Compose settings. No synthetic workers remain running.

Migration 019 was preflighted and tested in a rolled-back schema before being
applied through the normal deployment runner. All 182 pre-existing transcripts
and 63 recording records passed row-hash comparisons after qualification. The
injected database function/trigger was removed. Updated service image IDs were
verified, and every published local port binds to 127.0.0.1.

The repeatable overlay/probe and complete evidence live in Nanosamurai's
`docker-compose.refinement-tracks-smoke.yml`, `smoke-tests/refinement-tracks/`
and `docs/refinement-tracks-spike.md`, mirrored in Nanodeploy. Production workers
have no synthetic mode, no new Kafka topics and no new window/result storage.

The worker's conservative checkpoint retains each active session's first input
offset until its idle tail is acknowledged. Recovery can repeat inference and
requires the original audio to remain in Kafka. Idle timeout still means audio
completion; resuming an evicted session can restart its timing origin. A durable
end/resume contract, browser track selection/tabs and per-track failure status
remain follow-up work. Deployment migration copies must stay byte-identical;
this service does not own the deployment migration ledger.

The post-merge WhisperX integration run on 2026-09-15 found a missing
`defaultdict` import in legacy speaker enrollment. The runtime consolidation
removed the import while `_init_diarization_models()` still used it. Restore
that import so the worker can load local enrollment samples and publish results.
The existing local-enrollment integration test covers this regression; the
ordinary Compose smoke used S3 enrollment and did not exercise this path.
