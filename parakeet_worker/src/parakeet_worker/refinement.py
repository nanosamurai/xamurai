"""Refine audio windows with Parakeet and embedded Sortformer."""
from parakeet_worker.pipeline import MODEL_ID, Parakeet
from xamurai_serving.refinement import main as run_refinement


def main():
    pipeline = Parakeet()
    try:
        run_refinement(pipeline, model=MODEL_ID)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
