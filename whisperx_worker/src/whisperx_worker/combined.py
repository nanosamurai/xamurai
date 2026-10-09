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
from whisperx_worker.alignment import AlignmentExecutor, WhisperXAlignmentBackend, configured_languages
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
    languages = configured_languages(os.getenv("WHISPERX_SHARED_ALIGNMENT_LANGUAGES",
                                     os.getenv("WHISPERX_SHARED_ALIGNMENT_LANGUAGE", "en")))
    alignment_concurrency = int(os.getenv("WHISPERX_SHARED_ALIGNMENT_CONCURRENCY", "2"))
    finalization_concurrency = int(os.getenv("WHISPERX_SHARED_FINALIZATION_CONCURRENCY", "2"))
    if not 1 <= alignment_concurrency <= 8 or not 1 <= finalization_concurrency <= 8:
        raise ValueError("Shared alignment and finalization concurrency must be between 1 and 8")
    # Every configured aligner and its tokenizer must be ready before consumption.
    pipeline._init_whisperx()
    pipeline._init_diarization_models()
    if pipeline._ENABLE_DIARIZATION and pipeline._DIAR_PIPE is None:
        raise RuntimeError("Shared WhisperX diarization did not initialize")
    alignment_backend = WhisperXAlignmentBackend(languages, pipeline._WHISPERX_DEVICE)
    logger.info("Shared WhisperX startup models ready: alignment_languages=%s", ",".join(languages))
    owner = ModelOwner(WhisperXBackend(pipeline._WHISPERX_MODEL),
                       batch_size=int(os.getenv("WHISPERX_SHARED_BATCH_SIZE", "16")),
                       wait_ms=float(os.getenv("WHISPERX_SHARED_BATCH_WAIT_MS", "20")),
                       max_pending=2 * (finalization_concurrency + 1))
    alignment = AlignmentExecutor(alignment_backend, concurrency=alignment_concurrency,
                                  max_pending=finalization_concurrency + 1)
    model = os.getenv("WHISPERX_MODEL", "medium").strip() or "medium"
    polled = {"refinement": time.monotonic(), "finalization": time.monotonic()}
    failed = threading.Event()

    def infer(stage, path, **kwargs):
        with shared_execution(owner, stage, alignment):
            if stage == "refinement":
                result = refinement.transcribe(path, **kwargs)
            else:
                result = pipeline.run_whisperx_diarized_words(path, use_alignment=True, **kwargs)
            if owner.closed:
                raise RuntimeError("Shared model failed during inference") from owner.failure
            if alignment.closed:
                raise RuntimeError("Shared alignment failed during inference") from alignment.failure
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
                                kwargs={"decoupled": True, "max_inflight": finalization_concurrency}, daemon=True)]
    for thread in threads:
        thread.start()
    timeout = float(os.getenv("WHISPERX_SHARED_OPERATION_TIMEOUT_S", "1800"))
    try:
        while not stop.wait(1):
            now = time.monotonic()
            if (not owner.thread.is_alive() or owner.closed or not alignment.healthy(timeout)
                    or any(now - last > 60 for last in polled.values())
                    or (owner.busy_since is not None and now - owner.busy_since > timeout)):
                failed.set()
                stop.set()
                break
            temporary = HEALTH_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps({"updated": time.time(), "batches": owner.batch_count,
                                             "mixed_batches": owner.mixed_batch_count, **alignment.snapshot()}))
            temporary.replace(HEALTH_PATH)
    finally:
        HEALTH_PATH.unlink(missing_ok=True)
        stop.set()
        deadline = time.monotonic() + float(os.getenv("WHISPERX_SHARED_DRAIN_SECONDS", "30"))
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
        owner.close()
        alignment.close()
        for thread in alignment.threads:
            thread.join(max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads + alignment.threads):
            # ThreadPoolExecutor's exit hook would wait forever for native code.
            os._exit(1)
    if failed.is_set():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
