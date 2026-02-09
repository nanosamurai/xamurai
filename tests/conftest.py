import os
import re
import sys
from pathlib import Path

import pytest


# Ensure repo root is on sys.path so imports like `rtservice.server` work in tests
# when running without editable installs.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Ensure src roots are on sys.path for monorepo-style layout.
# This matches what editable installs would provide.
for rel in [
    "rtservice/src",
    "recorder_worker/src",
    "whisperx_worker/src",
    "finalizer_worker/src",
    "shared/src",
]:
    p = (_REPO_ROOT / rel).resolve()
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _normalize_bootstrap(value: str) -> str:
    # testcontainers sometimes returns e.g. "PLAINTEXT://localhost:12345"
    return re.sub(r"^[A-Z]+://", "", value)


def _ensure_topics(kafka_bootstrap: str, topics: list[str]) -> None:
    """Ensure Kafka topics exist.

    Test brokers may have auto-create disabled, so tests must create topics explicitly.
    This helper is intentionally lightweight and best-effort.
    """

    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": kafka_bootstrap})
    existing = admin.list_topics(timeout=10).topics.keys()

    new_topics = [
        NewTopic(t, num_partitions=1, replication_factor=1)
        for t in topics
        if t not in existing
    ]

    if not new_topics:
        return

    fs = admin.create_topics(new_topics)
    for _t, f in fs.items():
        try:
            f.result(10)
        except Exception:
            # Topic may already exist due to races; not fatal.
            pass


@pytest.fixture(scope="session")
def kafka_bootstrap() -> str:
    """Kafka bootstrap address for integration tests.

    Default behavior:
    - Prefer an explicitly provided `KAFKA_BOOTSTRAP` (lets you run against real Kafka)
    - Otherwise start Kafka via Testcontainers and return its bootstrap address.

    This keeps existing integration tests working after removing the previous
    DB/Kafka-heavy fixtures.
    """

    external = os.getenv("KAFKA_BOOTSTRAP")
    if external:
        return _normalize_bootstrap(external)

    try:
        from testcontainers.kafka import KafkaContainer
    except Exception as e:
        pytest.skip(f"testcontainers.kafka not available: {e}")
        raise

    # NOTE: avoid pinning too aggressively; choose a common CP Kafka image.
    with KafkaContainer("confluentinc/cp-kafka:7.6.1") as kafka:
        bs = _normalize_bootstrap(kafka.get_bootstrap_server())
        yield bs
