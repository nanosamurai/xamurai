"""Real thread scheduling with a fake poll-owned Kafka consumer."""
from threading import Event, get_ident
import time

import pytest
from confluent_kafka import TopicPartition
from xamurai_serving.finalization_runtime import run_decoupled


class Message:
    def topic(self): return "recordings"
    def partition(self): return 0
    def offset(self): return 4
    def error(self): return None


class Consumer:
    def __init__(self, stop, *, revoke=False, commit_failure=False):
        self.stop = stop
        self.polls = 0
        self.owner = get_ident()
        self.commits = []
        self.revoked = revoke
        self.commit_failure = commit_failure
        self.paused = False

    def subscribe(self, topics, **callbacks): self.callbacks = callbacks
    def assignment(self): return [TopicPartition("recordings", 0)]
    def pause(self, partitions): self.paused = True
    def resume(self, partitions): self.paused = False
    def poll(self, _timeout):
        assert get_ident() == self.owner
        self.polls += 1
        time.sleep(.002)
        if self.polls == 1: return Message()
        if self.polls == 3 and self.revoked:
            self.callbacks["on_revoke"](self, self.assignment())
        if self.polls > 50: self.stop.set()
    def commit(self, msg, **kwargs):
        assert get_ident() == self.owner
        if self.commit_failure: raise RuntimeError("commit failed")
        self.commits.append(msg.offset())
        self.stop.set()


@pytest.mark.parametrize("revoke", [False, True])
def test_polls_during_inference_and_fences_revoked_completion(revoke):
    stop = Event()
    consumer = Consumer(stop, revoke=revoke)
    deleted = []
    def process(_msg):
        assert get_ident() != consumer.owner
        time.sleep(.03)
        assert consumer.polls > 3
        return "s3://recording"
    run_decoupled(consumer=consumer, topic="recordings", process=process,
                  cleanup=deleted.append, stop_event=stop)
    assert consumer.commits == ([] if revoke else [4])
    assert deleted == ([] if revoke else ["s3://recording"])


@pytest.mark.parametrize("failure", ["publication", "commit"])
def test_failure_never_deletes_recording(failure):
    stop = Event()
    consumer = Consumer(stop, commit_failure=failure == "commit")
    deleted = []
    def process(_msg):
        if failure == "publication": raise RuntimeError("publish failed")
        return "s3://recording"
    with pytest.raises(RuntimeError):
        run_decoupled(consumer=consumer, topic="recordings", process=process,
                      cleanup=deleted.append, stop_event=stop)
    assert consumer.commits == []
    assert deleted == []


def test_unselected_message_skips_processing_without_pausing():
    stop = Event()
    consumer = Consumer(stop)
    run_decoupled(consumer=consumer, topic="recordings", selected=lambda _m: False,
                  process=lambda _m: pytest.fail("unselected inference"),
                  cleanup=lambda _url: pytest.fail("unselected deletion"), stop_event=stop)
    assert consumer.commits == [4]
    assert not consumer.paused
