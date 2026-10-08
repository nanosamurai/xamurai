"""Poll-owned Kafka commits while a single recording is processed off-thread."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from confluent_kafka import KafkaException, TopicPartition


def run_decoupled(*, consumer, topic, process, cleanup, stop_event=None, on_poll=None,
                  selected=None):
    stop_event = stop_event or Event()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="finalization")
    running = None
    epochs = {}
    paused = set()

    def revoke(_consumer, partitions):
        for p in partitions:
            key = (p.topic, p.partition)
            epochs[key] = epochs.get(key, 0) + 1
            paused.discard(key)

    consumer.subscribe([topic], on_revoke=revoke, on_lost=revoke)
    try:
        while not stop_event.is_set():
            # Poll before handling completion so queued revocations fence commits.
            assigned = {(p.topic, p.partition) for p in consumer.assignment()}
            if running:
                newly_paused = assigned - paused
                if newly_paused:
                    consumer.pause([TopicPartition(*p) for p in newly_paused])
                    paused.update(newly_paused)
            elif paused:
                consumer.resume([TopicPartition(*p) for p in paused & assigned])
                paused.clear()
            msg = consumer.poll(0.1)
            if on_poll:
                on_poll()
            if msg is not None:
                if msg.error():
                    raise KafkaException(msg.error())
                if running:
                    # A freshly assigned partition may deliver once before the
                    # next pause. Rewind it; never skip or commit that record.
                    consumer.seek(TopicPartition(msg.topic(), msg.partition(), msg.offset()))
                else:
                    if selected is not None and not selected(msg):
                        consumer.commit(msg, asynchronous=False)
                        continue
                    key = (msg.topic(), msg.partition())
                    running = (msg, epochs.get(key, 0), executor.submit(process, msg))
            if running and running[2].done():
                original, epoch, future = running
                delete_url = future.result()
                key = (original.topic(), original.partition())
                owned = {(p.topic, p.partition) for p in consumer.assignment()}
                if key in owned and epochs.get(key, 0) == epoch:
                    consumer.commit(original, asynchronous=False)
                    if delete_url:
                        cleanup(delete_url)
                running = None
    finally:
        # No commit during shutdown; unacknowledged work is replayed. The process
        # supervisor bounds draining, including hung native calls.
        executor.shutdown(wait=True, cancel_futures=True)
