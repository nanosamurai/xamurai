# syntax=docker/dockerfile:1

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Ensure monorepo imports work
ENV PYTHONPATH=/app:/app/shared/src:/app/rtservice/src:/app/whisperx_worker/src:/app/recorder_worker/src:/app/finalizer_worker/src

WORKDIR /app

# System deps required by rtservice stack (audio + ML)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# Install deps. Torch wheels are CUDA-enabled when available.
# NOTE: if CUDA wheels for py3.13 are unavailable, this layer will fail.
# 1) Install torch from the PyTorch CUDA index
# Keep torch pinned to reduce resolver conflicts across services.
RUN pip install \
      "torch==2.8.0+cu128" \
      "torchaudio==2.8.0+cu128" \
      --index-url https://download.pytorch.org/whl/cu128

# 2) Install the rest from PyPI
RUN pip install \
      numpy \
      scipy \
      soundfile \
      confluent-kafka \
      protobuf \
      grpcio \
      grpcio-tools \
      faster-whisper \
      pyannote-audio

# Copy monorepo
COPY . /app

# Non-root runtime
RUN useradd -m -u 10001 appuser \
    && chown -R 10001:10001 /app
USER 10001

EXPOSE 50052

# Required at runtime:
# - HF_TOKEN
# Optional:
# - RT_GRPC_PORT (default 50052)
# - ENROLL_DIR (default enrolled_speakers)
# - FW_LANG_DEFAULT
CMD ["python", "-m", "rtservice.server"]
