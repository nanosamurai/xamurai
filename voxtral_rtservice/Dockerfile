# syntax=docker/dockerfile:1
# vLLM 0.32.0, immutable multi-architecture image manifest.
FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/shared/src:/app/voxtral_rtservice/src \
    HF_HOME=/data/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    DO_NOT_TRACK=1

WORKDIR /app
RUN apt-get update \
    && apt-get install --yes --no-install-recommends --only-upgrade \
      ca-certificates libssl3t64 openssl libbrotli1 linux-libc-dev \
      dirmngr gnupg gnupg-utils gnupg2 gpg gpg-agent gpgconf gpgsm gpgv keyboxd \
    && rm -rf /var/lib/apt/lists/*
COPY voxtral_rtservice/requirements.txt voxtral_rtservice/check_dependencies.py /build/
# The vendor image contains an unused newer OTLP-common distribution alongside
# its 1.40 telemetry stack; remove that orphan, retaining the compatible stack.
RUN python3 -m pip uninstall -y opentelemetry-exporter-otlp-common \
    && python3 -m pip install --no-cache-dir -r /build/requirements.txt \
    && python3 /build/check_dependencies.py
COPY proto_gen /app/proto_gen
COPY shared/src/xamurai_serving /app/shared/src/xamurai_serving
COPY voxtral_rtservice/src /app/voxtral_rtservice/src
RUN useradd --create-home --uid 10004 voxtral \
    && mkdir -p /data/huggingface \
    && chown -R 10004:10004 /app /data/huggingface
USER 10004
EXPOSE 50052
HEALTHCHECK --interval=10s --timeout=3s --start-period=15m --retries=3 \
  CMD ["python3", "-m", "voxtral_rtservice.healthcheck"]
ENTRYPOINT []
CMD ["python3", "-m", "voxtral_rtservice.server"]
