"""Independent alignment scheduling, isolation and bounded failure behavior."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from threading import Event
import time
from types import SimpleNamespace

import pytest

from whisperx_worker.alignment import AlignmentExecutor, WhisperXAlignmentBackend, configured_languages
from whisperx_worker.batching import ModelOwner
from whisperx_worker.execution import alignment_operation, shared_execution


def close(executor):
    executor.close()
    for thread in executor.threads:
        thread.join(2)
        assert not thread.is_alive()


@pytest.mark.parametrize("value", ["", "en,", "en,,de", "../../model", "english"])
def test_invalid_language_set(value):
    with pytest.raises(ValueError):
        configured_languages(value)


def test_language_set_normalizes_and_deduplicates():
    assert configured_languages("EN, de,cs,en") == ("en", "de", "cs")


def test_backend_preloads_all_models_and_selects_without_reloading(monkeypatch):
    import sys
    loaded, tokenizers, calls = [], [], []
    def load(*, language_code, device):
        loaded.append(language_code)
        return SimpleNamespace(eval=lambda: None, language=language_code), {"language": language_code}
    def align(segments, model, metadata, audio, device, **_kwargs):
        calls.append((segments, model.language, metadata["language"], audio))
        return calls[-1]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "nltk", SimpleNamespace(data=SimpleNamespace(load=tokenizers.append)))
    monkeypatch.setitem(sys.modules, "whisperx", SimpleNamespace(load_align_model=load, align=align))
    monkeypatch.setitem(sys.modules, "whisperx.alignment", SimpleNamespace(
        DEFAULT_ALIGN_MODELS_TORCH={"en": "english", "de": "german"}, DEFAULT_ALIGN_MODELS_HF={"cs": "czech"}))
    monkeypatch.setitem(sys.modules, "whisperx.utils", SimpleNamespace(PUNKT_LANGUAGES={"de": "german", "cs": "czech"}))
    backend = WhisperXAlignmentBackend(("en", "de", "cs"), "cpu")
    for language in ("en", "de", "cs", "en"):
        assert backend.align([language], language, language) == ([language], language, language, language)
    assert loaded == ["en", "de", "cs"]
    assert len(tokenizers) == 3
    with pytest.raises(ValueError, match="Unsupported"):
        WhisperXAlignmentBackend(("zz",), "cpu")


def test_different_languages_overlap_and_same_language_waits_without_blocking_asr():
    entered = {lang: Event() for lang in ("en", "de")}
    release = Event()
    trace = ContextVar("test_trace", default=None)
    def align(segments, audio, language):
        entered[language].set()
        assert release.wait(3)
        return (segments, audio, language, trace.get())
    executor = AlignmentExecutor(SimpleNamespace(languages=("en", "de"), align=align), concurrency=2)
    owner = ModelOwner(None)
    @alignment_operation
    def route(*_args):
        return "standalone"
    def call(language, identity):
        trace.set(identity)
        with shared_execution(owner, "finalization", executor):
            return route([identity], identity, language)
    try:
        assert route([], None, "en") == "standalone"
        with ThreadPoolExecutor(3) as callers:
            first = callers.submit(call, "en", "first")
            assert entered["en"].wait(2)
            same = callers.submit(call, "en", "second")
            other = callers.submit(call, "de", "third")
            try:
                assert entered["de"].wait(2)
                assert not same.done()
                # Alignment must leave the shared ASR/model thread available.
                assert owner.call(lambda: "ASR available") == "ASR available"
                assert executor.snapshot()["alignment_peak_active"] == 2
            finally:
                release.set()
            assert first.result(2) == (["first"], "first", "en", "first")
            assert same.result(2) == (["second"], "second", "en", "second")
            assert other.result(2) == (["third"], "third", "de", "third")
    finally:
        release.set()
        owner.close()
        close(executor)


def test_capacity_deadline_and_shutdown_wake_callers():
    entered, release = Event(), Event()
    def align(*_args):
        entered.set()
        assert release.wait(3)
    executor = AlignmentExecutor(SimpleNamespace(languages=("en",), align=align), max_pending=1)
    try:
        with ThreadPoolExecutor(1) as caller:
            first = caller.submit(executor.align, [], None, "en")
            try:
                assert entered.wait(2)
                assert not executor.healthy(-1)
                with pytest.raises(RuntimeError, match="queue is full"):
                    executor.align([], None, "en")
                with pytest.raises(ValueError, match="not preloaded"):
                    executor.align([], None, "de")
                executor.close()
                with pytest.raises(RuntimeError, match="stopped"):
                    first.result(1)
            finally:
                release.set()
    finally:
        close(executor)


def test_alignment_failure_closes_executor_and_rejects_later_work():
    def align(*_args):
        raise RuntimeError("GPU failed")
    executor = AlignmentExecutor(SimpleNamespace(languages=("en",), align=align))
    try:
        with pytest.raises(RuntimeError, match="GPU failed"):
            executor.align([], None, "en")
        assert not executor.healthy(10)
        with pytest.raises(RuntimeError, match="stopped"):
            executor.align([], None, "en")
    finally:
        close(executor)


def test_global_concurrency_limit_can_serialize_different_languages():
    entered, release, german = Event(), Event(), Event()
    def align(_segments, _audio, language):
        if language == "en":
            entered.set()
            assert release.wait(3)
        else:
            german.set()
    executor = AlignmentExecutor(SimpleNamespace(languages=("en", "de"), align=align), concurrency=1)
    try:
        with ThreadPoolExecutor(2) as callers:
            first = callers.submit(executor.align, [], None, "en")
            assert entered.wait(2)
            second = callers.submit(executor.align, [], None, "de")
            try:
                assert not german.wait(.05)
            finally:
                release.set()
            first.result(2)
            second.result(2)
        assert executor.snapshot()["alignment_peak_active"] == 1
    finally:
        release.set()
        close(executor)
