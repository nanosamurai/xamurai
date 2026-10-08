"""Bounded cross-request ASR batching and a single owner for all GPU models."""
from collections import deque
from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import dataclass, field
import logging
import threading
import time

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Request:
    prepared: object
    chunks: object
    key: object
    stage: str
    future: Future = field(default_factory=Future)
    results: list = field(default_factory=list)
    next_chunk: object = None
    enqueued: float = field(default_factory=time.monotonic)


class ModelOwner:
    """Two pipeline callers share one owner; only one batch of features is built.

    Commands (VAD, alignment, diarization) alternate with ASR batches. ASR jobs
    rotate by chunk, including between incompatible language groups. A recording
    holds its waveform and segment metadata, never all its mel tensors at once.
    """
    def __init__(self, backend, *, batch_size=16, wait_ms=20, max_pending=8):
        if not 1 <= batch_size <= 16 or not 0 <= wait_ms <= 1000 or max_pending < 2:
            raise ValueError("Invalid shared batch limits")
        self.backend = backend
        self.batch_size = batch_size
        self.wait_s = wait_ms / 1000
        self.max_pending = max_pending
        self.condition = threading.Condition()
        self.commands = deque()
        self.requests = deque()
        self.pending = set()
        self.closed = False
        self.failure = None
        self.busy_since = None
        self.batch_count = 0
        self.mixed_batch_count = 0
        self.thread = threading.Thread(target=self._run, name="whisperx-model", daemon=True)
        self.thread.start()

    def is_owner_thread(self):
        return threading.current_thread() is self.thread

    def _admit(self, future):
        if self.closed:
            raise RuntimeError("Shared model owner stopped") from self.failure
        if len(self.pending) >= self.max_pending:
            raise RuntimeError("Shared model queue is full")
        self.pending.add(future)

    def call(self, fn, *args, **kwargs):
        if self.is_owner_thread():
            return fn(*args, **kwargs)
        future = Future()
        with self.condition:
            self._admit(future)
            self.commands.append((future, copy_context(), fn, args, kwargs))
            self.condition.notify()
        return future.result()

    def transcribe(self, audio, *, stage, language=None):
        def prepare_and_enqueue():
            prepared = self.backend.prepare(audio, language)
            request = Request(prepared, iter(prepared.chunks), prepared.key, stage)
            request.next_chunk = next(request.chunks, None)
            if request.next_chunk is None:
                request.future.set_result(self.backend.finish(prepared, []))
            else:
                with self.condition:
                    self._admit(request.future)
                    self.requests.append(request)
                    self.condition.notify()
            return request
        request = self.call(prepare_and_enqueue)
        return request.future.result()

    def close(self, error=None):
        with self.condition:
            self.closed = True
            self.failure = error
            for future in self.pending:
                if not future.done():
                    future.set_exception(error or RuntimeError("Shared model owner stopped"))
            self.pending.clear()
            self.commands.clear()
            self.requests.clear()
            self.condition.notify_all()

    def _complete(self, future, value):
        with self.condition:
            self.pending.discard(future)
            if not future.done():
                future.set_result(value)

    def _take_batch(self):
        # The first waiting group gets the next batch. Rotate every selected
        # request so a long finalization cannot fill ahead of ready refinement.
        key = self.requests[0].key
        batch = []
        skipped = 0
        while self.requests and len(batch) < self.batch_size:
            request = self.requests.popleft()
            if request.key != key:
                self.requests.append(request)
                skipped += 1
                if skipped >= len(self.requests):
                    break
                continue
            skipped = 0
            batch.append((request, request.next_chunk))
            request.next_chunk = next(request.chunks, None)
            if request.next_chunk is not None:
                self.requests.append(request)
        return batch

    def _run(self):
        last_was_command = False
        try:
            while True:
                with self.condition:
                    while not self.closed and not (self.commands or self.requests):
                        self.condition.wait()
                    if self.closed:
                        return
                    command = None
                    if self.commands and (not self.requests or not last_was_command):
                        command = self.commands.popleft()
                    else:
                        # Wait only for a partial batch, measured from its oldest
                        # ready request. Commands can prepare another request while
                        # the collection window is open.
                        oldest = self.requests[0]
                        available = sum(r.prepared.remaining(r.next_chunk) for r in self.requests
                                        if r.key == oldest.key)
                        delay = oldest.enqueued + self.wait_s - time.monotonic()
                        if available < self.batch_size and delay > 0:
                            if self.commands:
                                command = self.commands.popleft()
                            else:
                                self.condition.wait(delay)
                                continue
                        if command is None:
                            batch = self._take_batch()
                    self.busy_since = time.monotonic()
                if command is not None:
                    future, context, fn, args, kwargs = command
                    self._complete(future, context.run(fn, *args, **kwargs))
                    last_was_command = True
                else:
                    results = self.backend.decode(batch)
                    if len(results) != len(batch):
                        raise RuntimeError("ASR batch result count mismatch")
                    for (request, _chunk), result in zip(batch, results):
                        request.results.append(result)
                    for request in dict.fromkeys(r for r, _ in batch):
                        if request.next_chunk is None:
                            self._complete(request.future, self.backend.finish(request.prepared, request.results))
                    self.batch_count += 1
                    mixed = len({r.stage for r, _ in batch}) > 1
                    self.mixed_batch_count += int(mixed)
                    logger.info("Shared ASR batch: chunks=%d refinement=%d finalization=%d mixed=%s",
                                len(batch), sum(r.stage == "refinement" for r, _ in batch),
                                sum(r.stage == "finalization" for r, _ in batch), mixed)
                    last_was_command = False
                self.busy_since = None
        except BaseException as error:
            logger.error("Shared model owner failed: %s", type(error).__name__)
            self.close(error)
