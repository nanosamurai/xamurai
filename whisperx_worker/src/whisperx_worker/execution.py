"""Optional per-job execution context; standalone pipelines execute directly."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_execution = ContextVar("whisperx_execution", default=None)


@contextmanager
def shared_execution(owner, stage):
    token = _execution.set((owner, stage))
    try:
        yield
    finally:
        _execution.reset(token)


def model_operation(fn):
    """Serialize model operations while retaining each caller's trace context."""
    @wraps(fn)
    def run(*args, **kwargs):
        execution = _execution.get()
        if execution is None or execution[0].is_owner_thread():
            return fn(*args, **kwargs)
        return execution[0].call(fn, *args, **kwargs)
    return run


def transcribe(model, audio, **kwargs):
    execution = _execution.get()
    if execution is None:
        return model.transcribe(audio, **kwargs)
    owner, stage = execution
    return owner.transcribe(audio, stage=stage, language=kwargs.get("language"))
