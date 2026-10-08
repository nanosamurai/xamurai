"""Run the optional whisperx-shared track without changing standalone workers."""
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time

from drsynth_common.logging_setup import setup_logging
from drsynth_common.otel_setup import setup_otel
from whisperx_worker.batching import ModelOwner
from whisperx_worker.execution import shared_execution

HEALTH_PATH = Path(os.getenv("WHISPERX_SHARED_HEALTH_FILE", "/tmp/whisperx-shared.json"))
logger = logging.getLogger(__name__)


def main():
    # Set defaults before importing runtimes which read their configuration once.
    os.environ.setdefault("REFINEMENT_TRACK_ID", "whisperx-shared")
    os.environ.setdefault("FINALIZER_TRACK_ID", "whisperx-shared")
    from whisperx_worker import pipeline, refinement
    from whisperx_worker.batch_backend import WhisperXBackend
    from xamurai_serving import refinement as refiner, finalization as finalizer

    setup_logging(default_level="INFO")
    setup_otel(service_name=os.getenv("OTEL_SERVICE_NAME", "whisperx-shared"))
    HEALTH_PATH.unlink(missing_ok=True)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    # Warm common models once before either consumer starts.
    pipeline._init_whisperx()
    pipeline._init_diarization_models()
    if pipeline._ENABLE_DIARIZATION and pipeline._DIAR_PIPE is None:
        raise RuntimeError("Shared WhisperX diarization did not initialize")
    owner = ModelOwner(WhisperXBackend(pipeline._WHISPERX_MODEL),
                       batch_size=int(os.getenv("WHISPERX_SHARED_BATCH_SIZE", "16")),
                       wait_ms=float(os.getenv("WHISPERX_SHARED_BATCH_WAIT_MS", "20")))
    model = os.getenv("WHISPERX_MODEL", "medium").strip() or "medium"
    polled = {"refinement": time.monotonic(), "finalization": time.monotonic()}
    failed = threading.Event()

    def infer(stage, path, **kwargs):
        with shared_execution(owner, stage):
            if stage == "refinement":
                result = refinement.transcribe(path, **kwargs)
            else:
                result = pipeline.run_whisperx_diarized_words(path, use_alignment=True, **kwargs)
            if owner.closed:
                raise RuntimeError("Shared model failed during inference") from owner.failure
            return result

    def run(stage, runtime, **kwargs):
        try:
            runtime(lambda path, **kw: infer(stage, path, **kw), model=model,
                    stop_event=stop, on_poll=lambda: polled.update({stage: time.monotonic()}),
                    initialize=False, **kwargs)
        except BaseException:
            logger.exception("Shared %s loop failed", stage)
            failed.set()
        finally:
            if not stop.is_set():
                failed.set()
            stop.set()

    threads = [threading.Thread(target=run, args=("refinement", refiner.main), daemon=True),
               threading.Thread(target=run, args=("finalization", finalizer.main),
                                kwargs={"decoupled": True}, daemon=True)]
    for thread in threads:
        thread.start()
    timeout = float(os.getenv("WHISPERX_SHARED_OPERATION_TIMEOUT_S", "1800"))
    try:
        while not stop.wait(1):
            now = time.monotonic()
            if (not owner.thread.is_alive() or owner.closed
                    or any(now - last > 60 for last in polled.values())
                    or (owner.busy_since is not None and now - owner.busy_since > timeout)):
                failed.set()
                stop.set()
                break
            temporary = HEALTH_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps({"updated": time.time(), "batches": owner.batch_count,
                                             "mixed_batches": owner.mixed_batch_count}))
            temporary.replace(HEALTH_PATH)
    finally:
        HEALTH_PATH.unlink(missing_ok=True)
        stop.set()
        deadline = time.monotonic() + float(os.getenv("WHISPERX_SHARED_DRAIN_SECONDS", "30"))
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
        owner.close()
        if any(thread.is_alive() for thread in threads):
            # ThreadPoolExecutor's exit hook would wait forever for native code.
            os._exit(1)
    if failed.is_set():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
