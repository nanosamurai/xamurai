# Contributing

Thanks for your interest in contributing!

## Development workflow
- Create a feature branch from `master`.
- Keep commits small and focused.
- Add/update tests when changing behavior.

## Running checks locally

### Python unit tests
Use the repo's existing test commands (see CI workflows under `.github/workflows/`).

### Docker builds
This repo builds multiple images (rtservice + workers). To validate locally:

```bash
docker build -f rtservice/Dockerfile -t xamurai-rtservice:local .
docker buildx bake --load whisperx-worker
docker build -f recorder_worker/Dockerfile -t xamurai-recorder-worker:local .
docker buildx bake --load finalizer-worker
```

## Security
- Never commit secrets (tokens, API keys, private keys).
- Prefer `.env.example` over `.env`.
- CI runs secret scanning (gitleaks). Treat failures as blockers.