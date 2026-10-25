# Default speech model artifacts

Faster-Whisper and WhisperX verify the default Pyannote 3.1 pipeline config,
segmentation checkpoint, speaker checkpoint, PLDA data and enrollment checkpoint before
loading them. Immutable revisions and SHA-256 values live in
`shared/src/drsynth_common/pyannote_models.py`. Cached files are checked too;
a mismatch is never retried through an unpinned model ID.
Pyannote 4 also loads the community-1 PLDA data for this legacy pipeline. Those
two files are pinned and passed as local paths, avoiding implicit Hub downloads.
The model account must have access to the upstream community-1 repository too.

Pyannote needs pickle compatibility for these upstream checkpoints. Its loader
enables that for its own call. Xamurai does not replace `torch.load`, disable its
safer default, or add process-wide deserialization globals. Keep model caches
private to the service identity and separate from uploaded recordings.

`RT_DIAR_MODEL` and `WHISPERX_DIAR_MODEL` still accept custom operator-selected
pipelines. Those custom configurations and their nested artifacts are trusted
operator inputs; the default-profile hash checks do not cover them. Verify their
revisions and files separately before deployment. Clients cannot select arbitrary
model repositories through the realtime API.

Realtime Silero VAD is installed from the pinned `silero-vad` wheel during the
image build. Startup does not fetch and execute a Torch Hub repository.

Run `pytest -q tests/test_model_artifact_security.py` in a Torch-enabled test
environment. It checks modified cached files, nested checkpoint verification,
unchanged Torch defaults and rejection of a checkpoint with pickle side effects.
Real diarization/enrollment and audio tests remain required when changing pins.
