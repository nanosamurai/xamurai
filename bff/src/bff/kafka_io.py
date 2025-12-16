import time
from typing import Optional
from confluent_kafka import Producer
import stream_pb2
from bff.settings import settings

def _on_delivery(err, msg):
    if err:
        # You can swap to structured logging
        print(f"[KAFKA] delivery error: {err}")

def make_producer() -> Producer:
    conf = {
        "bootstrap.servers": settings.kafka_bootstrap,
        "client.id": settings.kafka_client_id,
        "compression.type": settings.kafka_compression,
        "linger.ms": settings.kafka_linger_ms,
        "batch.size": settings.kafka_batch_size,
        "acks": settings.kafka_acks,
        "enable.idempotence": False,  # you can turn on later if needed
        "retries": 3,
    }
    return Producer(conf)

def produce_audio_chunk(
    prod: Producer,
    session_id: str,
    seq: int,
    pcm16_bytes: bytes,
    sample_rate: int,
    t0_ns: Optional[int] = None,
):
    if t0_ns is None:
        t0_ns = time.monotonic_ns()

    msg = stream_pb2.AudioChunk(
        session_id=session_id,
        seq=seq,
        t0_ns=t0_ns,
        sample_rate=sample_rate,
        pcm16_le=pcm16_bytes,
    )
    prod.produce(
        topic=settings.kafka_topic_audio,
        key=session_id.encode("utf-8"),
        value=msg.SerializeToString(),
        on_delivery=_on_delivery,
    )
