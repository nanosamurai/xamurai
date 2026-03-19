
# Local GPU enablement (Docker Desktop / Windows)

This repo's service images install CUDA-enabled PyTorch wheels (e.g. `torch==2.8.0+cu128`).
However, **CUDA will only be available at runtime if Docker is started with GPU access**.

## Quick check (host)

```bat
nvidia-smi
```

You should see your GPU listed.

## Quick check (Docker)

If Docker is configured correctly for NVIDIA, this should work:

```bat
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Enable GPU for `rtservice`

By default, `docker-compose.yml` runs `rtservice` on CPU. To opt-in to GPU,
use the provided compose override file:

```bat
docker compose -f docker-compose.yml -f docker-compose.gpu.override.yml up -d --build rtservice
```

Then validate:

```bat
docker exec xamurai-rtservice python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected output: `True 1` (or higher device count).

## Common failure modes

### `torch.cuda.is_available() == False` in the container

This typically means **the container was not started with GPU access**.
In Compose you must set `gpus: all` (or an equivalent device request).

### Docker runtime missing NVIDIA

Check Docker runtimes:

```bat
docker info --format "Runtimes={{json .Runtimes}} Default={{.DefaultRuntime}}"
```

You should see an `nvidia` runtime available.

### Docker Desktop configuration

Docker Desktop on Windows generally requires:

- WSL2 backend enabled
- Recent NVIDIA driver installed on the host
- NVIDIA Container Toolkit integration available to Docker

If the `nvidia/cuda` container test above fails, fix Docker/NVIDIA first.
