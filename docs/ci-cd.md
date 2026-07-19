# Xamurai CI and image publication

Xamurai owns validation and container-image publication for its four speech
services. Full-stack orchestration, infrastructure provisioning, and
environment-specific release configuration are intentionally outside this
repository.

## Continuous integration

The lightweight CI workflow runs on pull requests and pushes to `master`. It:

- installs the pinned dependencies from `requirements.ci.txt`
- runs unit tests that do not require Kafka, model downloads, or a GPU

Separate integration workflows cover the two ML dependency stacks:

- `integration-fastwhisper.yml` validates realtime gRPC behavior
- `integration-whisperx.yml` validates WhisperX refinement, diarization, and
  local/S3 enrollment behavior

The integration workflows disable pyannote telemetry and use constrained model
settings suitable for CI. Gated model access is provided through the minimum
required `HF_TOKEN` repository secret.

## Image publication

On pushes to `master`, `publish-image.yml` builds each service from its own
Dockerfile and publishes an immutable `sha-<git-sha>` tag:

- `ghcr.io/nanosamurai/xamurai-rtservice`
- `ghcr.io/nanosamurai/xamurai-whisperx-worker`
- `ghcr.io/nanosamurai/xamurai-recorder-worker`
- `ghcr.io/nanosamurai/xamurai-finalizer-worker`

The workflow grants only `contents: read` and `packages: write`. Authentication
uses the workflow-scoped `GITHUB_TOKEN`; no external registry credential is
stored in the repository.

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
