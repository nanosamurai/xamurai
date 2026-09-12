"""Explicitly gated deterministic provider for local topology qualification."""

import os
import time
from drsynth_common.final_tracks import TranscriptResult


class TestTrack:
    def __init__(self, profile_id="test-final-r1"):
        self.profile_id = profile_id

    def describe(self):
        return {"profile_id": self.profile_id, "test_only": True}

    def process(self, audio, context):
        time.sleep(min(10, max(0, float(os.getenv("FINAL_TRACK_TEST_DELAY_SECONDS", "0")))))
        if os.getenv("FINAL_TRACK_TEST_FAIL") == "true":
            raise RuntimeError("test_inference_failure")
        return TranscriptResult("fixture", [{"text": "fixture", "start_s": 0.0,
                                              "end_s": min(0.1, audio.source.sample_count / 16000)}],
                                context.language, {"segment_timestamps": True}, self.describe(), [])
