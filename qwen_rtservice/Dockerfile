# syntax=docker/dockerfile:1

FROM qwenllm/qwen3-asr@sha256:fb75b775f089e06e5a1aaebffd421e37505cc630d50c86d889d95ffa45a7e16a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/shared/src:/app/qwen_rtservice/src \
    HF_HOME=/data/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYANNOTE_METRICS_ENABLED=0 \
    DO_NOT_TRACK=1 \
    VLLM_NO_USAGE_STATS=1

WORKDIR /app

# The upstream image inherits PyGObject through Ubuntu's software-properties
# packages without pycairo. This service uses none of them, so remove the
# packages through their owning package manager instead of adding graphics libs.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
      ca-certificates \
      libgomp1 \
      libsndfile1 \
    && apt-get purge --yes \
      software-properties-common \
      python3-software-properties \
      python3-gi \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade \
      "qwen-asr[vllm]==0.0.6" \
      "vllm==0.14.0" \
      "pyannote-audio==4.0.7" \
      "grpcio==1.76.0" \
      "protobuf==6.33.0" \
    && pip check

COPY proto_gen /app/proto_gen
COPY shared/src /app/shared/src
COPY qwen_rtservice/src /app/qwen_rtservice/src

RUN useradd --create-home --uid 10002 rtservice \
    && mkdir -p /data/huggingface \
    && chown -R 10002:10002 /app /data/huggingface

USER 10002

EXPOSE 50052

HEALTHCHECK --interval=10s --timeout=3s --start-period=10m --retries=3 \
  CMD ["python", "-m", "qwen_rtservice.healthcheck"]

ENTRYPOINT []
CMD ["python", "-m", "qwen_rtservice.server"]
