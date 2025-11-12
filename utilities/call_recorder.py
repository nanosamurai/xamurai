#!/usr/bin/env python3
"""
Record a phone-call style waveform: MIC + WASAPI LOOPBACK → mixed 16 kHz mono WAV.

- Captures mic with sounddevice (callback, 16 kHz, mono)
- Captures system audio via pyaudiowpatch WASAPI loopback (device rate → resampled to 16 kHz)
- Time-aligns by BLOCKs and sums (with clipping) into one mono stream
- Writes to a WAV file (PCM_16) until Ctrl+C

Tested on Windows. For other OSes, replace the loopback capture with an OS-appropriate method.
"""

import time
import queue
import threading
import traceback
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import pyaudiowpatch as pyaudio

# ================== CONFIG ==================
SR = 16000                 # target sample rate
BLOCK = 2048              # processing block at 16 kHz (~128 ms)
OUT_PATH = "call_recording.wav"

# Mic: None = default device; or set an integer index
MIC_DEVICE_INDEX = None

# Loopback device selection: part of device name (case-insensitive) or None for default speakers
LOOPBACK_NAME_CONTAINS = None

# Optional: print a very light meter every N mixed blocks (0 = off)
PRINT_METER_EVERY = 0  # e.g., 20 prints about every ~2.5s
# ============================================

# Queues for each source
q_mic: "queue.Queue[np.ndarray]" = queue.Queue()
q_loop: "queue.Queue[np.ndarray]" = queue.Queue()

def _pick_wasapi_loopback(pa: pyaudio.PyAudio, name_contains: str | None = None):
    """Pick a WASAPI loopback device for system audio."""
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    def_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])

    if name_contains:
        name_l = name_contains.lower()
        for info in pa.get_loopback_device_info_generator():
            if name_l in info["name"].lower():
                return info

    if def_out.get("isLoopbackDevice"):
        return def_out

    for info in pa.get_loopback_device_info_generator():
        if def_out["name"] in info["name"]:
            return info

    # fallback: first loopback device
    for info in pa.get_loopback_device_info_generator():
        return info

    raise RuntimeError("No WASAPI loopback device found. Run `python -m pyaudiowpatch` to inspect devices.")

# -------- Mic capture (sounddevice @ 16k mono) --------
def mic_callback(indata, frames, t, status):
    if status:
        print(status)
    # indata: float32 (frames, 1)
    q_mic.put(indata.copy().reshape(-1))

def mic_thread(device_index=None):
    kwargs = dict(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK, callback=mic_callback)
    if device_index is not None:
        kwargs["device"] = device_index
    try:
        with sd.InputStream(**kwargs):
            print(f"🎙️ Mic listening (device={device_index})… Ctrl+C to stop")
            while True:
                time.sleep(0.1)
    except Exception:
        print("❌ Mic thread crashed:")
        traceback.print_exc()

# -------- Loopback capture (pyaudiowpatch → resample to 16k) --------
def loopback_thread(name_contains: str | None = None):
    print("🔁 Loopback thread starting…")
    pa = pyaudio.PyAudio()
    stream = None
    try:
        dev = _pick_wasapi_loopback(pa, name_contains)
        in_rate = int(dev["defaultSampleRate"])
        in_ch = min(2, dev["maxInputChannels"]) or 2
        in_block = 4096  # device chunk; will be reblocked to 2048 @ 16k

        print(f"🔁 Opening WASAPI loopback: ({dev['index']}) {dev['name']} @ {in_rate} Hz")
        stream = pa.open(format=pyaudio.paFloat32,
                         channels=in_ch,
                         rate=in_rate,
                         input=True,
                         input_device_index=dev["index"],
                         frames_per_buffer=in_block)

        resampler = torchaudio.transforms.Resample(orig_freq=in_rate, new_freq=SR)
        print(f"🔁 Loopback running: {in_rate} Hz → {SR} Hz")

        rem = np.zeros(0, dtype=np.float32)

        while True:
            data = stream.read(in_block, exception_on_overflow=False)
            x = np.frombuffer(data, dtype=np.float32)
            if x.size == 0:
                continue

            # mono mix of interleaved channels
            if in_ch > 1:
                x = x.reshape(-1, in_ch)
                L = x[:, 0]
                R = x[:, 1]
                rmsL = float(np.sqrt(np.mean(L * L)) + 1e-12)
                rmsR = float(np.sqrt(np.mean(R * R)) + 1e-12)
                corr = float(np.sum(L * R) / (len(L) * rmsL * rmsR))
                if corr < 0.2:
                    mono = (L if rmsL >= rmsR else R).astype(np.float32)
                else:
                    mono = (0.7 * L + 0.7 * R).astype(np.float32)
            else:
                mono = x.astype(np.float32)

            # resample to 16k
            y = resampler(torch.from_numpy(mono).unsqueeze(0)).squeeze(0).numpy()

            # accumulate and emit fixed 2048-sample blocks
            if rem.size:
                y = np.concatenate([rem, y])
            n_full = (y.size // BLOCK) * BLOCK
            if n_full:
                for ch in y[:n_full].reshape(-1, BLOCK):
                    ch = np.clip(np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                    q_loop.put(ch.copy())
                rem = y[n_full:]
            else:
                rem = y
    except Exception:
        print("❌ Loopback thread crashed:")
        traceback.print_exc()
    finally:
        if stream is not None:
            try:
                stream.stop_stream(); stream.close()
            except Exception:
                pass
        pa.terminate()
        print("🔁 Loopback thread terminated.")

# -------- Mixer: align by time, sum, write to WAV --------
def record_call(out_path: str):
    # Local source ring buffers
    mic_buf = np.zeros(0, dtype=np.float32)
    loop_buf = np.zeros(0, dtype=np.float32)

    # Create/overwrite output file
    out_path = str(Path(out_path).resolve())
    print(f"💾 Writing to: {out_path}")
    wav = sf.SoundFile(out_path, mode="w", samplerate=SR, channels=1, subtype="PCM_16")

    written_blocks = 0
    try:
        while True:
            # Drain queues
            drained = False
            while True:
                try:
                    block = q_mic.get_nowait()
                    mic_buf = np.concatenate([mic_buf, block])
                    drained = True
                except queue.Empty:
                    break
            while True:
                try:
                    block = q_loop.get_nowait()
                    loop_buf = np.concatenate([loop_buf, block])
                    drained = True
                except queue.Empty:
                    break

            # If nothing new arrived, idle briefly
            if not drained:
                time.sleep(0.005)

            # Mix aligned BLOCKs and write
            while len(mic_buf) >= BLOCK or len(loop_buf) >= BLOCK:
                m = np.zeros(BLOCK, dtype=np.float32)
                l = np.zeros(BLOCK, dtype=np.float32)

                if len(mic_buf) >= BLOCK:
                    m = mic_buf[:BLOCK]
                    mic_buf = mic_buf[BLOCK:]
                elif len(mic_buf) > 0:
                    m[:len(mic_buf)] = mic_buf
                    mic_buf = mic_buf[0:0]

                if len(loop_buf) >= BLOCK:
                    l = loop_buf[:BLOCK]
                    loop_buf = loop_buf[BLOCK:]
                elif len(loop_buf) > 0:
                    l[:len(loop_buf)] = loop_buf
                    loop_buf = loop_buf[0:0]

                mixed = np.clip(m + l, -1.0, 1.0)
                wav.write(mixed)

                written_blocks += 1
                if PRINT_METER_EVERY and (written_blocks % PRINT_METER_EVERY == 0):
                    rms = float(np.sqrt(np.mean(mixed**2)) + 1e-12)
                    dbfs = 20.0 * np.log10(rms) if rms > 0 else -120.0
                    print(f"\r🔊 level ~ {dbfs:6.1f} dBFS", end="", flush=True)

    except KeyboardInterrupt:
        print("\n🛑 Stopping…")
    finally:
        wav.close()
        print("✅ Saved.")

def main():
    # Start threads
    t_mic = threading.Thread(target=mic_thread, kwargs={"device_index": MIC_DEVICE_INDEX}, daemon=True)
    t_loop = threading.Thread(target=loopback_thread, kwargs={"name_contains": LOOPBACK_NAME_CONTAINS}, daemon=True)
    t_mic.start()
    t_loop.start()

    print("🎛️ Recording MIC + LOOPBACK → WAV. Press Ctrl+C to stop.")
    record_call(OUT_PATH)

if __name__ == "__main__":
    main()
