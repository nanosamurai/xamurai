import sounddevice as sd
import soundfile as sf
import numpy as np

SR = 16000
SECONDS = 20
OUT = "sample.wav"

print("Available devices:")
print(sd.query_devices())
print("\nRecording from default input for", SECONDS, "seconds…")
audio = sd.rec(int(SECONDS * SR),device=1, samplerate=SR, channels=1, dtype="float32")
sd.wait()
sf.write(OUT, audio, SR, subtype="PCM_16")
print("Saved:", OUT, "Duration:", len(audio)/SR, "sec")
