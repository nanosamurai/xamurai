"""Finalize complete recordings with Parakeet and embedded Sortformer."""
from parakeet_worker.pipeline import MODEL_ID, Parakeet
from xamurai_serving.finalization import main as run_finalizer


def main():
    pipeline = Parakeet()
    try:
        run_finalizer(pipeline, model=MODEL_ID)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
