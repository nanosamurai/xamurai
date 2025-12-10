#!/usr/bin/env python3
import os
import sys
from confluent_kafka import Consumer, KafkaException

from drsynth_proto import stream_pb2  # generated from proto/stream.proto

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_REFINED = os.getenv("KAFKA_TOPIC_REFINED", "transcripts.refined")
GROUP_ID = os.getenv("KAFKA_TEST_GROUP_ID", "refined-printer")


def make_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        "enable.auto.commit": True,
        "auto.offset.reset": "latest",  # start from "now"
    })


def main():
    print(f"[REFINED-CONSUMER] bootstrap={KAFKA_BOOTSTRAP} topic={TOPIC_REFINED}")

    c = make_consumer()
    c.subscribe([TOPIC_REFINED])

    try:
        while True:
            msg = c.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            key = msg.key().decode("utf-8") if msg.key() else ""
            ev = stream_pb2.RefinedEvent()
            ev.ParseFromString(msg.value())

            # print nicely
            speaker = ev.speaker or "?"
            print(
                f"[REFINED] session={ev.session_id or key} "
                f"{ev.start_s:7.2f}–{ev.end_s:7.2f}s "
                f"speaker={speaker} text={ev.text}"
            )

    except KeyboardInterrupt:
        print("\n[REFINED-CONSUMER] interrupted, exiting…")
    finally:
        try:
            c.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
