"""Finalize complete recordings with the shared WhisperX pipeline."""
import os
from functools import partial

from whisperx_worker.pipeline import _init_whisperx, run_whisperx_diarized_words
from xamurai_serving.finalization import main as run_finalizer


def main():
    _init_whisperx()
    run_finalizer(partial(run_whisperx_diarized_words, use_alignment=True),
                  model=os.getenv("WHISPERX_MODEL", "medium").strip() or "medium")


if __name__ == "__main__":
    main()
