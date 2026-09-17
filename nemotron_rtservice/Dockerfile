# syntax=docker/dockerfile:1
FROM nemo-speech-native

COPY nemotron_rtservice/src /app/nemotron_rtservice/src
ENV PYTHONPATH=/app:/app/shared/src:/app/nemo_speech_native/src:/app/nemotron_rtservice/src

EXPOSE 50052

HEALTHCHECK --interval=10s --timeout=3s --start-period=15m --retries=3 \
  CMD ["python", "-m", "nemotron_rtservice.healthcheck"]

ENTRYPOINT []
CMD ["python", "-m", "nemotron_rtservice.server"]
