# syntax=docker/dockerfile:1

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

ENV PYTHONPATH=/app:/app/shared/src:/app/rtservice/src:/app/whisperx_worker/src:/app/recorder_worker/src:/app/finalizer_worker/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

# 1) Torch from the PyTorch CUDA index
# WhisperX pins torch~=2.8.0, so we align with that.
RUN pip install \
      "torch==2.8.0+cu128" \
      "torchaudio==2.8.0+cu128" \
      --index-url https://download.pytorch.org/whl/cu128

# 2) WhisperX worker deps from PyPI
RUN pip install \
      numpy \
      scipy \
      soundfile \
      confluent-kafka \
      protobuf \
      whisperx

COPY . /app

RUN useradd -m -u 10001 appuser \
    && mkdir -p /data/hf_cache \
    && chown -R 10001:10001 /app /data
USER 10001

# Env vars:
# - KAFKA_BOOTSTRAP, KAFKA_TOPIC_AUDIO, KAFKA_TOPIC_REFINED
# - WHISPERX_SLICE_SECONDS, WHISPERX_IDLE_SECONDS, WHISPERX_COMPUTE_TYPE
# - KAFKA_GROUP_ID
CMD ["python", "-m", "whisperx_worker.whisperx_worker"]
