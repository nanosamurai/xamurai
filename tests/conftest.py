# tests/conftest.py

import sys
from pathlib import Path
from testcontainers.kafka import KafkaContainer

# Ensure project root (the directory that contains rtservice/, bff/, etc.) is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

@pytest.fixture(scope="session")
def kafka_bootstrap():
    """
    Start a Kafka container for integration tests and yield the bootstrap URL.

    Requires: pip install testcontainers[kafka]
    """
    # You can change the image if you prefer confluent's:
    # KafkaContainer("confluentinc/cp-kafka:latest")
    with KafkaContainer("apache/kafka:latest") as kafka:
        yield kafka.get_bootstrap_server()
