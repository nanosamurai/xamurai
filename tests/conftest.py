# tests/conftest.py
import os
import time

import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
import sys
from pathlib import Path

# Ensure project root (the directory that contains rtservice/, bff/, etc.) is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.append(os.path.join(ROOT, "whisperx_worker", "src"))
sys.path.append(os.path.join(ROOT, "proto_gen"))

KAFKA_IMAGE = os.getenv("TEST_KAFKA_IMAGE", "apache/kafka:latest")

@pytest.fixture(scope="session")
def kafka_bootstrap():
    """
    Start Apache Kafka (KRaft) using apache/kafka:latest via a generic container.

    Reuses a minimal single-broker KRaft config similar to what you'd have in docker-compose.
    """

    # Adjust env vars to match your docker-compose.yml!
    container = (
        DockerContainer(KAFKA_IMAGE)
        .with_bind_ports(9092, 9092)
        .with_bind_ports(9093, 9093)
        .with_env("KAFKA_NODE_ID", "1")
        .with_env("KAFKA_PROCESS_ROLES", "broker,controller")
        .with_env(
            "KAFKA_LISTENERS",
            "PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093",
        )
        .with_env(
            "KAFKA_ADVERTISED_LISTENERS",
            # from the host we’ll connect via localhost + mapped port
            "PLAINTEXT://localhost:9092",
        )
        .with_env("KAFKA_CONTROLLER_LISTENER_NAMES", "CONTROLLER")
        .with_env(
            "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP",
            "CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT",
        )
        .with_env("KAFKA_CONTROLLER_QUORUM_VOTERS", "1@localhost:9093")
        .with_env("KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR", "1")
        .with_env("KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR", "1")
        .with_env("KAFKA_TRANSACTION_STATE_LOG_MIN_ISR", "1")
        .with_env("KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS", "0")
    )

    with container as kafka:
        # Optionally wait for a log line that indicates Kafka is ready.
        # You may need to tweak the pattern after seeing actual logs.
        try:
            wait_for_logs(
                kafka,
                "Kafka Server started",
                timeout=120,
                interval=2,
            )
        except Exception as e:
            # Dump logs for debugging if something goes wrong
            stdout, stderr = kafka.get_logs()
            print("=== Kafka STDOUT ===")
            print(stdout.decode("utf-8", errors="ignore"))
            print("=== Kafka STDERR ===")
            print(stderr.decode("utf-8", errors="ignore"))
            raise

        bootstrap = "localhost:9092"
        print(f"[tests] Kafka bootstrap: {bootstrap}")
        yield bootstrap

@pytest.fixture(scope="session")
def kafka_bootstrap_existing():
    """
    Use an already-running Kafka, typically started via docker-compose.

    e.g. docker compose up kafka
    """
    return os.getenv("TEST_KAFKA_BOOTSTRAP", "localhost:9092")