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
docker build -f whisperx_worker/Dockerfile -t xamurai-whisperx-worker:local .
docker build -f recorder_worker/Dockerfile -t xamurai-recorder-worker:local .
docker build -f finalizer_worker/Dockerfile -t xamurai-finalizer-worker:local .
```

## Security
- Never commit secrets (tokens, API keys, private keys).
- Prefer `.env.example` over `.env`.
- CI runs secret scanning (gitleaks). Treat failures as blockers.