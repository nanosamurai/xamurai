import torch, soundfile as sf, numpy as np

SR = 16000
WAV = "sample.wav"

# load silero VAD
vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad",
    force_reload=False, onnx=False, trust_repo=True
)
get_speech_timestamps = vad_utils[0]

wave, sr = sf.read(WAV, dtype="float32")
assert sr == SR, f"Expected {SR} Hz, got {sr}"

def to_int16(x): return (np.clip(x,-1,1)*32767).astype(np.int16)

ts = get_speech_timestamps(to_int16(wave), vad_model,
                           sampling_rate=SR, return_seconds=True,
                           min_speech_duration_ms=300,
                           min_silence_duration_ms=250)

print("Detected speech segments (sec):")
for i, seg in enumerate(ts, 1):
    print(f"{i:02d}: {seg['start']:.2f} → {seg['end']:.2f}  dur={seg['end']-seg['start']:.2f}")
if not ts:
    print("No speech detected—try recording again and speak clearly.")
