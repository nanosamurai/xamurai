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
    "qwen_rtservice/src",
    "nemotron_rtservice/src",
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
def localstack_s3():
    """Session-scoped LocalStack S3 fixture.

    Requires Docker. Spins up localstack/localstack with S3 enabled,
    creates a fresh bucket, and yields connection info.

    Returns a dict with keys:
      - endpoint_url
      - region
      - access_key
      - secret_key
      - bucket
      - prefix
    """

    try:
        import boto3  # noqa: F401
    except Exception as e:
        pytest.skip(f"boto3 not available (required for LocalStack S3 tests): {e}")
        raise

    try:
        from testcontainers.localstack import LocalStackContainer
    except Exception as e:
        pytest.skip(f"testcontainers.localstack not available: {e}")
        raise

    # LocalStack defaults
    region = os.getenv("TEST_AWS_REGION", "us-east-1")
    bucket = os.getenv("TEST_ENROLL_S3_BUCKET", "xamurai-enrollment-test")
    prefix = os.getenv("TEST_ENROLL_S3_PREFIX", "enrollment")

    # NOTE:
    # Do not use `:latest` here. LocalStack occasionally introduces breaking startup
    # behavior (including account/token requirements) that makes CI flaky.
    #
    # Allow override for experiments, but keep a stable default.
    localstack_image = os.getenv("TEST_LOCALSTACK_IMAGE", "localstack/localstack:3.8.1")

    with (
        LocalStackContainer(localstack_image)
        .with_services("s3")
        # LocalStack may require acknowledging account requirements even for local/CI usage.
        .with_env("LOCALSTACK_ACKNOWLEDGE_ACCOUNT_REQUIREMENT", "1")
    ) as ls:
        endpoint_url = ls.get_url()

        # Create bucket via boto3
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )

        # us-east-1 special-cases CreateBucketConfiguration
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )

        yield {
            "endpoint_url": endpoint_url,
            "region": region,
            "access_key": "test",
            "secret_key": "test",
            "bucket": bucket,
            "prefix": prefix,
        }


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
