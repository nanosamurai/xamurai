import os
import torch
from pyannote.audio import Pipeline
import soundfile as sf

WAV = "sample.wav"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN environment variable first.")

try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                        token=HF_TOKEN)
except Exception as e:
    print(f"Error loading pipeline with community version: {e}")

# Optional: force GPU
pipeline.to(torch.device("cuda"))

output = pipeline("sample.wav")  # Annotation
print("Diarization segments:")
# Correct way to iterate over diarization results
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start}s and t={turn.end}s")
