# syntax=docker/dockerfile:1

FROM python:3.13.17-slim-trixie@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 AS dependencies

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

ENV PYTHONPATH=/app:/app/shared/src:/app/rtservice/src:/app/whisperx_worker/src:/app/recorder_worker/src

WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY constraints.active-speech.txt /opt/constraints.active-speech.txt

RUN python -m venv /opt/venv \
    && pip install --upgrade -c /opt/constraints.active-speech.txt pip==26.2.2 setuptools wheel

RUN pip install -c /opt/constraints.active-speech.txt \
      numpy \
      confluent-kafka \
      protobuf \
      boto3 \
      opentelemetry-api \
      opentelemetry-sdk \
      opentelemetry-exporter-otlp \
    && pip check

FROM dependencies AS runtime

# Installers and their vendored copies are not needed by the worker.
RUN /opt/venv/bin/python -m pip uninstall -y pip wheel \
    && /usr/local/bin/python -m pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.13/ensurepip

COPY . /app

RUN useradd -m -u 10001 appuser \
    && mkdir -p /data/recordings \
    && chown -R 10001:10001 /app /data

USER 10001

# Default volume mount for local recordings
VOLUME ["/data/recordings"]

# Env vars:
# - KAFKA_BOOTSTRAP, KAFKA_TOPIC_AUDIO
# - KAFKA_TOPIC_RECORDING_FINISHED, KAFKA_GROUP_ID_RECORDER
# - RECORDER_IDLE_SECONDS
# - RECORDING_STORAGE_BACKEND (local|s3)
# - RECORDING_DIR (default /data/recordings)
# - S3_ENDPOINT (optional), S3_BUCKET, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY, S3_FORCE_PATH_STYLE
# - S3_PREFIX (optional)
CMD ["python", "-m", "recorder_worker.recorder_worker"]
