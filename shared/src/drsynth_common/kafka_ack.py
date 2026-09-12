"""Bounded publication with delivery acknowledgement."""

import time


def produce_acked(producer, *, topic, key, value, headers=None, timeout=30):
    """Return only after delivery succeeds; an uncertain timeout must be replayed."""
    delivered = []
    deadline = time.monotonic() + timeout
    producer.produce(topic=topic, key=key, value=value, headers=headers,
                     on_delivery=lambda error, _message: delivered.append(error))
    while not delivered and time.monotonic() < deadline:
        producer.poll(0.1)
    if not delivered or delivered[0] is not None:
        raise RuntimeError("kafka_publication_unacknowledged")
