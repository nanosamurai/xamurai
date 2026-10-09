"""Bounded concurrent recordings with poll-owned, ordered Kafka commits."""
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

from confluent_kafka import KafkaException, TopicPartition


def run_decoupled(*, consumer, topic, process, cleanup, stop_event=None, on_poll=None,
                  selected=None, max_inflight=1):
    if not 1 <= max_inflight <= 8:
        raise ValueError("Finalization max_inflight must be between 1 and 8")
    stop_event = stop_event or Event()
    executor = ThreadPoolExecutor(max_workers=max_inflight, thread_name_prefix="finalization")
    # Count completed-but-uncommitted work too, bounding the completion backlog.
    running = []
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
            paused.intersection_update(assigned)
            if len(running) >= max_inflight:
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
                if len(running) >= max_inflight:
                    # A freshly assigned partition may deliver once before the
                    # next pause. Rewind it; never skip or commit that record.
                    consumer.seek(TopicPartition(msg.topic(), msg.partition(), msg.offset()))
                else:
                    key = (msg.topic(), msg.partition())
                    if selected is not None and not selected(msg):
                        future = Future()
                        future.set_result(None)
                    else:
                        future = executor.submit(process, msg)
                    running.append((msg, epochs.get(key, 0), future))
            # Observe failures before committing any completion from this poll.
            for _msg, _epoch, future in running:
                if future.done():
                    future.result()
            blocked = set()
            owned = {(p.topic, p.partition) for p in consumer.assignment()}
            for item in list(running):
                if stop_event.is_set():
                    break
                original, epoch, future = item
                key = (original.topic(), original.partition())
                current = key in owned and epochs.get(key, 0) == epoch
                if not future.done():
                    if current:
                        blocked.add(key)
                    continue
                if current and key in blocked:
                    continue
                # The future may have finished since the failure scan above.
                # Retrieve its result again before committing, never afterward.
                delete_url = future.result()
                if current:
                    consumer.commit(original, asynchronous=False)
                    if delete_url:
                        cleanup(delete_url)
                running.remove(item)
    finally:
        # No commit during shutdown; unacknowledged work is replayed. The process
        # supervisor bounds draining, including hung native calls.
        executor.shutdown(wait=True, cancel_futures=True)
