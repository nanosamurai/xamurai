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
# Keep torch pinned to reduce resolver conflicts across services. The Torch pins
# in constraints.active-speech.txt stop step 2 from replacing this CUDA build.
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

# CTranslate2 (Faster-Whisper) loads CUDA 12 cuBLAS by soname and does not look
# inside Python packages. The provider imports Torch first, which preloads the
# library, but point the loader at the cuBLAS wheel installed for Torch so GPU
# inference does not depend on that import order. The directory stays
# root-owned, so the runtime user cannot plant libraries there.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.13/site-packages/nvidia/cublas/lib

# Fail the build if the image cannot run Faster-Whisper on a GPU. CTranslate2
# loads cuBLAS only on first GPU use, so read the cuBLAS soname from its library
# and load it before importing Torch; this exercises the loader path above and
# follows CTranslate2 if a release moves to another CUDA major version. Torch
# must ship that same CUDA major. TorchCodec fails to import when its native
# library does not match the installed Torch. None of these checks needs a GPU.
RUN python -I -B - <<'EOF'
import ctypes
import pathlib
import re

import ctranslate2

ct2_libs = pathlib.Path(ctranslate2.__file__).parent.with_name("ctranslate2.libs")
sonames = {
    name.decode()
    for lib in ct2_libs.glob("libctranslate2*.so*")
    for name in re.findall(rb"libcublas\.so\.\d+", lib.read_bytes())
}
if len(sonames) != 1:
    raise SystemExit(f"expected one cuBLAS soname in CTranslate2, found {sorted(sonames)}")
(soname,) = sonames
ctypes.CDLL(soname)

import torch

cuda_major = soname.rsplit(".", 1)[1]
if (torch.version.cuda or "").split(".")[0] != cuda_major:
    raise SystemExit(f"CTranslate2 needs CUDA {cuda_major}, found Torch {torch.__version__}")

import torchcodec.decoders
EOF

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
