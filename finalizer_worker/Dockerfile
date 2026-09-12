# syntax=docker/dockerfile:1

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

ENV PYTHONPATH=/app:/app/shared/src:/app/rtservice/src:/app/whisperx_worker/src:/app/recorder_worker/src:/app/finalizer_worker/src:/app/qwen_rtservice/src

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

# finalizer imports run_whisperx from whisperx_worker, so we install whisperx deps too.
# 1) Torch from the PyTorch CUDA index
# WhisperX pins torch~=2.8.0, so we align with that.
RUN pip install \
      "torch==2.8.0+cu128" \
      "torchaudio==2.8.0+cu128" \
      --index-url https://download.pytorch.org/whl/cu128

# 2) Rest from PyPI
RUN pip install \
      numpy \
      scipy \
      soundfile \
      confluent-kafka \
      protobuf \
      boto3 \
      whisperx==3.8.6 \
      faster-whisper==1.2.1 \
      pyannote.audio==4.0.7 \
      transformers==4.57.6 \
      huggingface-hub==0.36.2 \
      opentelemetry-api \
      opentelemetry-sdk \
      opentelemetry-exporter-otlp

COPY . /app

RUN useradd -m -u 10001 appuser \
    && mkdir -p /data/recordings \
    && mkdir -p /data/hf_cache \
    && chown -R 10001:10001 /app /data
USER 10001

VOLUME ["/data/recordings"]

# FINAL_TRACKS_ENABLED=true selects the S3-backed generic final-track consumer.
CMD ["python", "-m", "finalizer_worker.finalizer_worker"]
