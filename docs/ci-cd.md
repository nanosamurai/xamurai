# Xamurai CI and image publication

Xamurai owns validation and container-image publication for its four speech
services. Full-stack orchestration, infrastructure provisioning, and
environment-specific release configuration are intentionally outside this
repository.

## Pull request validation

Pull requests targeting `master` run the lightweight CI workflow. It:

- installs the pinned dependencies from `requirements.ci.txt`
- runs unit tests that do not require Kafka, model downloads, or a GPU

Separate integration workflows cover the two ML dependency stacks:

- `integration-fastwhisper.yml` validates realtime gRPC behavior
- `integration-whisperx.yml` validates WhisperX refinement, diarization, and
  local/S3 enrollment behavior

The integration workflows disable pyannote telemetry and use constrained model
settings suitable for CI. Gated model access is provided through the minimum
required `HF_TOKEN` repository secret.

The repository ruleset for `master` must require these successful checks before
merge:

- `Python unit tests (lightweight)`
- `rtservice integration (faster-whisper + pyannote)`
- `whisperx integration (WhisperX + pyannote)`
- `scan`

The ruleset must also require pull requests to be up to date with `master` and
must not permit routine bypass of required checks. Workflow triggers make the
checks run; the repository ruleset makes them merge prerequisites.

Repository secrets are not available to workflows triggered from forks. Never
use `pull_request_target` to execute untrusted pull request code with secrets;
Xamurai integration tests that require `HF_TOKEN` must run from a trusted
repository branch.

## Image publication

On pushes to `master`, `publish-image.yml` first reuses the unit, Gitleaks, and
both integration workflows to validate the exact merged commit. Only after all
four gates succeed does it build each service from its own Dockerfile and
publish an immutable `sha-<git-sha>` tag:

- `ghcr.io/nanosamurai/xamurai-rtservice`
- `ghcr.io/nanosamurai/xamurai-whisperx-worker`
- `ghcr.io/nanosamurai/xamurai-recorder-worker`
- `ghcr.io/nanosamurai/xamurai-finalizer-worker`

The workflow grants only `contents: read` and `packages: write`. Authentication
uses the workflow-scoped `GITHUB_TOKEN`; no external registry credential is
stored in the repository.

`packages: write` is limited to the image publication job. Validation jobs are
read-only and receive only the specific `GITLEAKS_LICENSE` or `HF_TOKEN` secret
they require. A failed, cancelled, or misconfigured gate skips every image
build and push. The four image matrix entries remain independent after the
shared gate succeeds, so one Dockerfile failure does not cancel the other
service builds.

## Ownership boundary

Service Dockerfiles, dependency locks, unit tests, integration tests, and image
publication belong here. Stack wiring and local Community Edition smoke tests
belong in the public
[`nanosamurai/nanosamurai`](https://github.com/nanosamurai/nanosamurai)
repository.

This boundary keeps service changes independently testable and prevents
environment topology or deployment credentials from leaking into the public
service source.

## Release checks

Before publishing or promoting an image:

1. Run lightweight and applicable integration tests.
2. Run gitleaks against the complete branch history.
3. Build all four Dockerfiles from the repository root.
4. Confirm generated protobuf bindings match `proto/stream.proto`.
5. Use immutable image tags; do not publish mutable production references from
   unreviewed commits.
