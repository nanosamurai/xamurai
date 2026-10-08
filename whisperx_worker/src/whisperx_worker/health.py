"""Container probe for initialized models and supervised consumer loops."""
import json
import os
from pathlib import Path
import time


def main():
    try:
        state = json.loads(Path(os.getenv("WHISPERX_SHARED_HEALTH_FILE", "/tmp/whisperx-shared.json")).read_text())
        return 0 if 0 <= time.time() - state["updated"] < 10 else 1
    except (OSError, ValueError, KeyError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
