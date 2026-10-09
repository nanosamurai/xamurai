"""Out-of-order completion must never skip unfinished Kafka input."""
from collections import deque
from threading import Event, get_ident
import time

import pytest
from confluent_kafka import TopicPartition
from xamurai_serving.finalization_runtime import run_decoupled


class Message:
    def __init__(self, offset, partition=0):
        self.n, self.p = offset, partition
    def topic(self): return "recordings"
    def partition(self): return self.p
    def offset(self): return self.n
    def error(self): return None


class Consumer:
    def __init__(self, stop, messages):
        self.stop, self.messages = stop, deque(messages)
        self.owner = get_ident()
        self.paused = False
        self.commits = []
        self.hook = lambda: None
        self.deadline = time.monotonic() + 4
    def subscribe(self, _topics, **callbacks): self.callbacks = callbacks
    def assignment(self): return [TopicPartition("recordings", p) for p in (0, 1)]
    def pause(self, _partitions): self.paused = True
    def resume(self, _partitions): self.paused = False
    def seek(self, _partition): pytest.fail("test broker respects pause")
    def poll(self, _timeout):
        assert get_ident() == self.owner
        assert time.monotonic() < self.deadline, "consumer stalled"
        time.sleep(.002)
        self.hook()
        if not self.paused and self.messages:
            return self.messages.popleft()
    def commit(self, msg, **_kwargs):
        assert get_ident() == self.owner
        self.commits.append((msg.partition(), msg.offset()))


def test_failure_between_completion_scan_and_commit_is_not_committed(monkeypatch):
    from xamurai_serving import finalization_runtime
    class LateFailure:
        checks = 0
        def done(self):
            self.checks += 1
            return self.checks > 1
        def result(self):
            raise RuntimeError("publication failed after scan")
    class Executor:
        def __init__(self, **_kwargs): pass
        def submit(self, *_args): return LateFailure()
        def shutdown(self, **_kwargs): pass
    monkeypatch.setattr(finalization_runtime, "ThreadPoolExecutor", Executor)
    stop = Event()
    consumer = Consumer(stop, [Message(4)])
    with pytest.raises(RuntimeError, match="after scan"):
        run_decoupled(consumer=consumer, topic="recordings", process=lambda _msg: None,
                      cleanup=lambda _url: pytest.fail("deletion after failed publication"),
                      stop_event=stop, max_inflight=2)
    assert consumer.commits == []


@pytest.mark.parametrize("second_selected", [True, False])
def test_later_completion_or_skip_cannot_commit_past_slow_recording(second_selected):
    stop, release, second_done = Event(), Event(), Event()
    consumer = Consumer(stop, [Message(4), Message(5)])
    deleted = []
    def process(msg):
        if msg.offset() == 4:
            assert release.wait(3)
        else:
            second_done.set()
        return str(msg.offset())
    def selected(msg):
        if msg.offset() == 5 and not second_selected:
            second_done.set()
            return False
        return True
    def hook():
        if second_done.is_set() and not release.is_set():
            assert consumer.commits == [] and deleted == []
            assert consumer.paused  # completed-but-uncommitted input counts toward cap
            release.set()
        if len(consumer.commits) == 2:
            stop.set()
    consumer.hook = hook
    try:
        run_decoupled(consumer=consumer, topic="recordings", process=process, selected=selected,
                      cleanup=deleted.append, stop_event=stop, max_inflight=2)
    finally:
        release.set()
    assert consumer.commits == [(0, 4), (0, 5)]
    assert deleted == (["4", "5"] if second_selected else ["4"])


def test_other_partition_commits_independently():
    stop, release = Event(), Event()
    consumer = Consumer(stop, [Message(4), Message(9, 1)])
    deleted = []
    def process(msg):
        if msg.partition() == 0:
            assert release.wait(3)
        return str(msg.offset())
    def hook():
        if consumer.commits == [(1, 9)]:
            release.set()
        if len(consumer.commits) == 2:
            stop.set()
    consumer.hook = hook
    try:
        run_decoupled(consumer=consumer, topic="recordings", process=process,
                      cleanup=deleted.append, stop_event=stop, max_inflight=2)
    finally:
        release.set()
    assert consumer.commits == [(1, 9), (0, 4)]
    assert deleted == ["9", "4"]


@pytest.mark.parametrize("mode", ["failure", "revoke", "shutdown"])
def test_failure_revocation_and_shutdown_leave_uncommitted_recordings(mode):
    stop, release, second_done = Event(), Event(), Event()
    consumer = Consumer(stop, [Message(4), Message(5)])
    deleted = []
    def process(msg):
        if msg.offset() == 4:
            assert release.wait(3)
            if mode == "failure":
                raise RuntimeError("publication failed")
        else:
            second_done.set()
        return str(msg.offset())
    def hook():
        if second_done.is_set() and not release.is_set():
            assert consumer.commits == []
            if mode == "revoke":
                # assignment still reports the same partition: epoch fencing matters.
                consumer.callbacks["on_revoke"](consumer, consumer.assignment())
                consumer.paused = False  # a fresh assignment starts unpaused
            if mode == "shutdown":
                stop.set()
            release.set()
        elif release.is_set() and mode == "revoke" and not consumer.paused:
            stop.set()
    consumer.hook = hook
    try:
        kwargs = dict(consumer=consumer, topic="recordings", process=process,
                      cleanup=deleted.append, stop_event=stop, max_inflight=2)
        if mode == "failure":
            with pytest.raises(RuntimeError, match="publication failed"):
                run_decoupled(**kwargs)
        else:
            run_decoupled(**kwargs)
    finally:
        release.set()
    assert consumer.commits == [] and deleted == []
