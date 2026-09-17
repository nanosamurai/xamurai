"""Refine audio windows with the shared WhisperX pipeline."""
import os

from drsynth_common.logging_setup import setup_logging
from whisperx_worker import pipeline
from xamurai_serving.refinement import main as run_refinement


def transcribe(path, *, tenant=None, lang=None):
    text, segments = pipeline.run_whisperx_diarized(path, tenant=tenant, lang=lang, use_alignment=False)
    return text, [dict(start_s=start, end_s=end, text=text, speaker=speaker)
                  for start, end, text, speaker in segments]


def main():
    setup_logging(default_level="INFO")
    if pipeline.torch is None:
        raise RuntimeError("WhisperX requires torch")
    pipeline._init_whisperx()
    run_refinement(transcribe, model=os.getenv("WHISPERX_MODEL", "medium"))


if __name__ == "__main__":
    main()
