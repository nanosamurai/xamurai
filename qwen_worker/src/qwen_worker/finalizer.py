"""Finalize recordings with batched Qwen and pyannote."""
from qwen_worker.pipeline import MODEL_ID, Qwen
from xamurai_serving.finalization import main

if __name__ == "__main__":
    main(Qwen(), model=MODEL_ID)
