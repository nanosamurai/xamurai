from faster_whisper import WhisperModel

AUDIO = "sample.wav"
MODEL_SIZE = "large-v3"   # try "small" if VRAM is tight
COMPUTE = "float16"     # or "int8_float16" for lower VRAM

model = WhisperModel(MODEL_SIZE, device="cuda", compute_type=COMPUTE)
segments, info = model.transcribe(AUDIO, task="transcribe", beam_size=1, vad_filter=False)

print(f"Detected language: {info.language} (prob {info.language_probability:.2f})")
print("Transcript:")
for seg in segments:
    print(f"[{seg.start:.2f}–{seg.end:.2f}] {seg.text.strip()}")
