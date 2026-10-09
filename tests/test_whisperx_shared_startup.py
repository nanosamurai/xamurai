"""Alignment initialization must finish before consumers and readiness."""
from threading import Event, Thread

import pytest

from whisperx_worker import combined, pipeline, batch_backend
from xamurai_serving import refinement, finalization


class ModelsReady(Exception):
    """End the test at the boundary between model startup and serving."""


@pytest.fixture
def startup(monkeypatch, tmp_path):
    health = tmp_path / "health.json"
    health.write_text('{"updated": 0}')
    monkeypatch.setattr(combined, "HEALTH_PATH", health)
    monkeypatch.setattr(combined, "setup_logging", lambda **_kw: None)
    monkeypatch.setattr(combined, "setup_otel", lambda **_kw: None)
    monkeypatch.setattr(combined.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(pipeline, "_init_whisperx", lambda: None)
    monkeypatch.setattr(pipeline, "_init_diarization_models", lambda: None)
    monkeypatch.setattr(pipeline, "_ENABLE_DIARIZATION", False)
    monkeypatch.delenv("WHISPERX_SHARED_ALIGNMENT_LANGUAGE", raising=False)
    for runtime in (refinement, finalization):
        monkeypatch.setattr(runtime, "make_consumer", lambda: pytest.fail("consumer started before alignment"))
    return health


@pytest.mark.parametrize("configured,expected", [(None, "en"), (" de ", "de")])
def test_alignment_blocks_serving_until_loaded(startup, monkeypatch, configured, expected):
    if configured is not None:
        monkeypatch.setenv("WHISPERX_SHARED_ALIGNMENT_LANGUAGE", configured)
    loading, release, owner_started = Event(), Event(), Event()
    errors = []
    def load(language):
        assert language == expected
        assert not startup.exists()
        loading.set()
        assert release.wait(5)
    def backend(_model):
        owner_started.set()
        raise ModelsReady()
    monkeypatch.setattr(pipeline, "_ensure_align_model", load)
    monkeypatch.setattr(batch_backend, "WhisperXBackend", backend)
    def run():
        try:
            combined.main()
        except BaseException as error:
            errors.append(error)
    thread = Thread(target=run)
    thread.start()
    try:
        assert loading.wait(3)
        assert not startup.exists()
        assert not owner_started.is_set()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ModelsReady)
    assert owner_started.is_set()


def test_alignment_failure_prevents_readiness_and_consumption(startup, monkeypatch):
    def fail(_language):
        raise RuntimeError("alignment download failed")
    monkeypatch.setattr(pipeline, "_ensure_align_model", fail)
    monkeypatch.setattr(batch_backend, "WhisperXBackend", lambda _model: pytest.fail("owner started"))
    with pytest.raises(RuntimeError, match="alignment download failed"):
        combined.main()
    assert not startup.exists()
