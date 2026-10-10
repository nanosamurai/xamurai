"""Cross-request batching without model downloads or a GPU."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, get_ident
import time

import pytest

from whisperx_worker.batching import ModelOwner
from whisperx_worker.execution import model_operation, shared_execution, transcribe


@dataclass
class Prepared:
    chunks: list
    key: str

    def remaining(self, chunk):
        return len(self.chunks) - chunk


class Backend:
    def __init__(self):
        self.batches = []
        self.threads = set()

    def prepare(self, audio, language):
        self.threads.add(get_ident())
        return Prepared(list(range(audio)), language)

    def decode(self, batch):
        self.threads.add(get_ident())
        self.batches.append([(r.stage, r.key, chunk) for r, chunk in batch])
        return [(r.stage, chunk) for r, chunk in batch]

    def finish(self, prepared, results):
        return results


def test_requests_share_partial_batch_and_keep_independent_results():
    backend = Backend()
    owner = ModelOwner(backend, batch_size=16, wait_ms=200)
    try:
        with ThreadPoolExecutor(2) as pool:
            r = pool.submit(owner.transcribe, 4, stage="refinement", language="en")
            f = pool.submit(owner.transcribe, 8, stage="finalization", language="en")
            assert r.result(3) == [("refinement", i) for i in range(4)]
            assert f.result(3) == [("finalization", i) for i in range(8)]
        assert len(backend.batches) == 1
        assert len(backend.batches[0]) == 12
        assert owner.mixed_batch_count == 1
        assert len(backend.threads) == 1
    finally:
        owner.close()
        owner.thread.join(2)


def test_long_finalization_yields_between_batches():
    backend = Backend()
    first = Event()
    release = Event()
    original = backend.decode

    def decode(batch):
        if not backend.batches:
            first.set()
            assert release.wait(3)
        return original(batch)

    backend.decode = decode
    owner = ModelOwner(backend, batch_size=16, wait_ms=0)
    try:
        with ThreadPoolExecutor(2) as pool:
            f = pool.submit(owner.transcribe, 60, stage="finalization", language="en")
            assert first.wait(2)
            r = pool.submit(owner.transcribe, 2, stage="refinement", language="en")
            # Ensure refinement preparation is waiting before releasing GPU.
            deadline = time.monotonic() + 2
            while not owner.commands and time.monotonic() < deadline:
                time.sleep(.001)
            release.set()
            assert r.result(3) == [("refinement", 0), ("refinement", 1)]
            assert len(f.result(3)) == 60
        assert all(len(b) <= 16 for b in backend.batches)
        assert any(row[0] == "refinement" for b in backend.batches[1:3] for row in b)
    finally:
        release.set()
        owner.close()
        owner.thread.join(2)


def test_different_languages_never_mix():
    backend = Backend()
    owner = ModelOwner(backend, wait_ms=100)
    try:
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(owner.transcribe, 2, stage=s, language=l)
                    for s, l in [("refinement", "en"), ("finalization", "de")]]
            for job in jobs:
                assert len(job.result(3)) == 2
        assert len(backend.batches) == 2
        assert all(len({row[1] for row in b}) == 1 for b in backend.batches)
    finally:
        owner.close()


def test_empty_audio_and_partial_batch_deadline():
    backend = Backend()
    owner = ModelOwner(backend, wait_ms=20)
    try:
        assert owner.transcribe(0, stage="refinement", language="en") == []
        start = time.monotonic()
        assert owner.transcribe(1, stage="refinement", language="en") == [("refinement", 0)]
        assert .015 <= time.monotonic() - start < 2
    finally:
        owner.close()


def test_failure_wakes_every_waiter_and_rejects_new_work():
    backend = Backend()
    def fail(_batch):
        raise RuntimeError("decoder failed")
    backend.decode = fail
    owner = ModelOwner(backend, wait_ms=100)
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(owner.transcribe, 2, stage=s, language="en")
                for s in ("refinement", "finalization")]
        for job in jobs:
            with pytest.raises(RuntimeError):
                job.result(3)
    assert owner.closed
    with pytest.raises(RuntimeError):
        owner.call(lambda: None)


def test_pipeline_hooks_serialize_nested_operations_and_preserve_standalone():
    owner = ModelOwner(Backend())
    @model_operation
    def nested():
        return get_ident()
    @model_operation
    def operation():
        return nested()
    try:
        assert operation() == get_ident()
        with shared_execution(owner, "refinement"):
            assert operation() == owner.thread.ident
            assert transcribe(None, 1, language="en") == [("refinement", 0)]
        class Standalone:
            def transcribe(self, audio, **kwargs):
                return audio, kwargs
        assert transcribe(Standalone(), "audio", batch_size=16) == ("audio", {"batch_size": 16})
    finally:
        owner.close()


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": 17}, {"wait_ms": -1}])
def test_invalid_batch_limits(kwargs):
    with pytest.raises(ValueError):
        ModelOwner(Backend(), **kwargs)
