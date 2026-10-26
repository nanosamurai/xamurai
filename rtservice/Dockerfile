# syntax=docker/dockerfile:1

FROM python:3.13.17-slim-trixie@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Ensure monorepo imports work
ENV PYTHONPATH=/app:/app/shared/src:/app/rtservice/src:/app/whisperx_worker/src:/app/recorder_worker/src

WORKDIR /app

# System deps required by rtservice stack (audio + ML)
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY constraints.active-speech.txt /opt/constraints.active-speech.txt

RUN python -m venv /opt/venv \
    && pip install --upgrade -c /opt/constraints.active-speech.txt pip==26.2.2 setuptools wheel

# Install deps. Torch wheels are CUDA-enabled when available.
# NOTE: if CUDA wheels for py3.13 are unavailable, this layer will fail.
# 1) Install torch from the PyTorch CUDA index
# Keep torch pinned to reduce resolver conflicts across services.
RUN pip install --no-deps \
      "torch==2.8.0+cu128" \
      "torchaudio==2.8.0+cu128" \
      --index-url https://download.pytorch.org/whl/cu128

# 2) Resolve Torch's dependencies and the application packages from PyPI.
# The CUDA index is used only for the two explicitly selected Torch wheels.
# NOTE: pyannote-audio depends on omegaconf/hydra for some checkpoint configs.
RUN pip install -c /opt/constraints.active-speech.txt \
      numpy \
      scipy \
      soundfile \
      confluent-kafka \
      protobuf \
      grpcio \
      "faster-whisper==1.2.0" \
      "silero-vad==6.2.3" \
      "torchcodec==0.7.0" \
      pyannote-audio==4.0.7 \
      omegaconf \
      hydra-core \
      boto3 \
      prometheus-client \
      opentelemetry-api \
      opentelemetry-sdk \
      opentelemetry-exporter-otlp-proto-grpc \
    && pip check

FROM dependencies AS runtime

RUN /opt/venv/bin/python -m pip uninstall -y pip wheel \
    && /usr/local/bin/python -m pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.13/ensurepip

# Copy monorepo
COPY . /app

# Non-root runtime
RUN useradd -m -u 10001 appuser \
    && mkdir -p /data/hf_cache \
    && chown -R 10001:10001 /app /data
USER 10001

EXPOSE 50052

# Required at runtime:
# - HF_TOKEN
# Optional:
# - RT_GRPC_PORT (default 50052)
# - ENROLL_DIR (default enrolled_speakers)
# - FW_LANG_DEFAULT
CMD ["python", "-m", "rtservice.server"]
