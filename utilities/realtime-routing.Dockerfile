# syntax=docker/dockerfile:1

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/shared/src

WORKDIR /app

RUN pip install --no-cache-dir \
      "grpcio==1.76.0" \
      "protobuf==6.33.0"

COPY proto_gen /app/proto_gen
COPY shared/src /app/shared/src
COPY utilities/realtime_routing_server.py /app/realtime_routing_server.py

RUN useradd --create-home --uid 10003 routing
USER 10003

HEALTHCHECK --interval=2s --timeout=1s --retries=20 \
  CMD ["python", "-c", "import socket; socket.create_connection(('127.0.0.1', 50052), 1).close()"]

CMD ["python", "/app/realtime_routing_server.py"]
