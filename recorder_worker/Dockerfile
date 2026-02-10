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
        libsndfile1 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

RUN pip install \
      numpy \
      soundfile \
      confluent-kafka \
      protobuf \
      boto3

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
