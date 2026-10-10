"""Resident language models and bounded alignment outside the ASR executor."""
from collections import deque
from concurrent.futures import Future
from contextvars import copy_context
import logging
import os
import re
import threading
import time

logger = logging.getLogger(__name__)


def configured_languages(value):
    languages = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",")))
    if not languages or any(not re.fullmatch(r"[a-z]{2,3}", lang) for lang in languages):
        raise ValueError("Alignment languages must be comma-separated language codes, e.g. en,de,cs")
    return languages


class WhisperXAlignmentBackend:
    def __init__(self, languages, device):
        import nltk
        import torch
        import whisperx
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
        from whisperx.utils import PUNKT_LANGUAGES

        supported = DEFAULT_ALIGN_MODELS_HF.keys() | DEFAULT_ALIGN_MODELS_TORCH.keys()
        if set(languages) - supported:
            raise ValueError("Unsupported alignment languages: " + ",".join(sorted(set(languages) - supported)))
        self.languages = languages
        self.device = device
        self.torch = torch
        self.whisperx = whisperx
        self.models = {}
        self.streams = {}
        for language in languages:
            logger.info("Preloading shared alignment model: language=%s", language)
            model, metadata = whisperx.load_align_model(language_code=language, device=device)
            model.eval()
            self.models[language] = (model, metadata)
            # align() otherwise downloads this tokenizer lazily in a serving thread.
            resource = f"tokenizers/punkt_tab/{PUNKT_LANGUAGES.get(language, 'english')}.pickle"
            try:
                nltk.data.load(resource)
            except LookupError:
                cache_dir = os.environ.get("NLTK_DATA", "").split(os.pathsep)[0] or None
                nltk.download("punkt_tab", download_dir=cache_dir, quiet=True, raise_on_error=True)
                nltk.data.load(resource)
            if str(device).startswith("cuda"):
                self.streams[language] = torch.cuda.Stream(device=device)
        # Model initialization happened on the startup thread's stream. Finish it
        # before the independent language streams read those immutable weights.
        if self.streams:
            torch.cuda.synchronize(device)

    def align(self, segments, audio, language):
        model, metadata = self.models[language]
        def run():
            return self.whisperx.align(segments, model, metadata, audio, self.device,
                                      return_char_alignments=False)
        stream = self.streams.get(language)
        if stream is None:
            return run()
        with self.torch.cuda.stream(stream):
            result = run()
        stream.synchronize()
        return result


class AlignmentExecutor:
    """At most one call per language, with a separate global concurrency cap.

    Waiting calls for a busy language cannot occupy all workers and block another
    language. Each job retains its own inputs, results and tracing context.
    """
    def __init__(self, backend, *, concurrency=2, max_pending=8):
        if not 1 <= concurrency <= 8 or not 1 <= max_pending <= 16:
            raise ValueError("Invalid alignment concurrency/queue limits")
        self.backend = backend
        self.condition = threading.Condition()
        self.queue = deque()
        self.pending = set()
        self.active = {}
        self.closed = False
        self.failure = None
        self.completed = 0
        self.peak_active = 0
        self.threads = [threading.Thread(target=self._run, name=f"whisperx-alignment-{i}", daemon=True)
                        for i in range(min(concurrency, len(backend.languages)))]
        self.max_pending = max_pending
        for thread in self.threads:
            thread.start()

    def align(self, segments, audio, language):
        if language not in self.backend.languages:
            # Preserve the pipeline's unaligned fallback without downloading or
            # evicting models based on a request's detected/supplied language.
            raise ValueError(f"Alignment language is not preloaded: {language}")
        future = Future()
        with self.condition:
            if self.closed:
                raise RuntimeError("Shared alignment stopped") from self.failure
            if len(self.pending) >= self.max_pending:
                raise RuntimeError("Shared alignment queue is full")
            self.pending.add(future)
            self.queue.append((future, copy_context(), segments, audio, language))
            self.condition.notify_all()
        return future.result()

    def snapshot(self):
        with self.condition:
            return {"alignment_active": len(self.active), "alignment_peak_active": self.peak_active,
                    "alignment_completed": self.completed, "alignment_languages": list(self.backend.languages)}

    def healthy(self, timeout):
        with self.condition:
            return (not self.closed and all(t.is_alive() for t in self.threads)
                    and all(time.monotonic() - started <= timeout for started in self.active.values()))

    def close(self, error=None):
        with self.condition:
            self.closed = True
            self.failure = self.failure or error
            for future in self.pending:
                if not future.done():
                    future.set_exception(error or RuntimeError("Shared alignment stopped"))
            self.pending.clear()
            self.queue.clear()
            self.condition.notify_all()

    def _run(self):
        try:
            while True:
                with self.condition:
                    while True:
                        if self.closed:
                            return
                        job = next((job for job in self.queue if job[4] not in self.active), None)
                        if job is not None:
                            self.queue.remove(job)
                            future, context, segments, audio, language = job
                            self.active[language] = time.monotonic()
                            self.peak_active = max(self.peak_active, len(self.active))
                            active_count = len(self.active)
                            break
                        self.condition.wait()
                logger.info("Shared alignment started: language=%s active=%d", language, active_count)
                result = context.run(self.backend.align, segments, audio, language)
                with self.condition:
                    elapsed = time.monotonic() - self.active.pop(language)
                    self.pending.discard(future)
                    self.completed += 1
                    if not future.done():
                        future.set_result(result)
                    self.condition.notify_all()
                logger.info("Shared alignment finished: language=%s seconds=%.3f", language, elapsed)
        except BaseException as error:
            logger.error("Shared alignment failed: %s", type(error).__name__)
            self.close(error)
