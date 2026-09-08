# syntax=docker/dockerfile:1

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS native-builder

ARG NEMO_SPEECH_REVISION=4f9676226f667d14608487df744f375db87127f8
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install --yes --no-install-recommends \
      ca-certificates \
      cmake \
      g++-13 \
      gcc-13 \
      git \
      libsentencepiece-dev \
      ninja-build \
      pkg-config \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
RUN git init \
    && git remote add origin https://github.com/NVIDIA/NeMo-Speech.cpp.git \
    && git fetch --depth 1 origin "${NEMO_SPEECH_REVISION}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${NEMO_SPEECH_REVISION}" \
    && GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 ggml \
    && bash scripts/apply-ggml-patches.sh

# Build only the stable ASR C ABI and the CUDA libraries it needs. The two
# architectures cover the production T4 target (sm75) and local Blackwell
# validation (sm120) without shipping an unused server framework.
RUN cmake -G Ninja -S /work -B /work/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="75;120" \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_NCCL=OFF \
      -DGGML_CUDA_GRAPHS=ON \
      -DGGML_NATIVE=OFF \
      -DNEMO_SPEECH_CUBLAS_SHIM=OFF \
      -DNEMO_SPEECH_GGML_PATCHED=ON \
      -DNEMO_SPEECH_BUILD_ASR=ON \
      -DNEMO_SPEECH_BUILD_DIAR=ON \
      -DNEMO_SPEECH_BUILD_TTS=OFF \
      -DNEMO_SPEECH_BUILD_NMT=OFF \
      -DNEMO_SPEECH_BUILD_CLI=OFF \
      -DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF \
      -DNEMO_SPEECH_BUILD_HTTP=OFF \
      -DNEMO_SPEECH_BUILD_GRPC=OFF \
      -DNEMO_SPEECH_BUILD_EXAMPLES=OFF \
      -DNEMO_SPEECH_BUILD_TESTS=OFF \
      -DNEMO_SPEECH_BUILD_TOOLS=OFF \
      -DNEMO_SPEECH_WITH_FLASHLIGHT=OFF \
      -DNEMO_SPEECH_WITH_NORM=OFF \
    && cmake --build /work/build --parallel 2 \
    && cmake --install /work/build --prefix /opt/nemo-speech

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/shared/src:/app/nemotron_rtservice/src \
    LD_LIBRARY_PATH=/opt/nemo-speech/lib \
    HF_HOME=/data/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install --yes --no-install-recommends \
      ca-certificates \
      libgomp1 \
      libsentencepiece0 \
      libsndfile1 \
      python3 \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --retries 5 --timeout 120 \
      "grpcio==1.76.0" \
      "huggingface-hub==0.36.0" \
      "numpy==2.2.3" \
      "protobuf==6.33.0" \
      "onnxruntime==1.23.2" \
      "kaldi-native-fbank==1.22.3" \
      "soundfile==0.13.1" \
      "scipy==1.15.3" \
      "boto3==1.43.89" \
    && /opt/venv/bin/pip check

ENV PATH=/opt/venv/bin:${PATH}

COPY --from=native-builder /opt/nemo-speech /opt/nemo-speech
COPY proto_gen /app/proto_gen
COPY shared/src /app/shared/src
COPY nemotron_rtservice/src /app/nemotron_rtservice/src

RUN useradd --create-home --uid 10003 rtservice \
    && mkdir -p /data/huggingface \
    && chown -R 10003:10003 /app /data/huggingface

USER 10003

EXPOSE 50052

HEALTHCHECK --interval=10s --timeout=3s --start-period=15m --retries=3 \
  CMD ["python", "-m", "nemotron_rtservice.healthcheck"]

ENTRYPOINT []
CMD ["python", "-m", "nemotron_rtservice.server"]
