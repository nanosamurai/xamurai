import torch
print("torch.cuda.is_available:", torch.cuda.is_available(), "CUDA:", torch.version.cuda)

try:
    from faster_whisper import WhisperModel
    print("faster-whisper import: OK")
except Exception as e:
    print("faster-whisper import ERROR:", e)

try:
    from pyannote.audio import Pipeline
    print("pyannote.audio import: OK")
except Exception as e:
    print("pyannote import ERROR:", e)

