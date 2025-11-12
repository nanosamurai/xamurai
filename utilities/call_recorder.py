#!/usr/bin/env python3
# MIC (sounddevice 16k mono) + LOOPBACK (PyAudioWPatch) -> 16k WAV
# - No torchaudio
# - Common-block writer (no zero padding)
# - Stable decimate-by-3 fallback if loopback can’t open at 16k

import queue, threading, time, traceback
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import pyaudiowpatch as pyaudio

# ================== CONFIG ==================
SR = 16000
BLOCK = 2048                  # 128 ms @ 16k
OUT_PATH = "call_recording.wav"
MIC_DEVICE_INDEX = 9       # None = default mic for sounddevice
LOOPBACK_NAME_CONTAINS: Optional[str] = None  # substring to pick loopback device, or None
WRITE_STEREO = True          # True: ch0=loopback, ch1=mic ; False: mono mix
PRINT_METER_EVERY = 0         # 0 = off
# ============================================

q_mic:  "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)   # float32 (BLOCK,)
q_loop: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)   # float32 (BLOCK,)

# ---------- utils ----------
def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x*x)) + 1e-12)
    return 20*np.log10(rms) if rms > 0 else -120.0

def pick_wasapi_loopback(pa: pyaudio.PyAudio, name_contains: Optional[str]) -> dict:
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    def_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])

    if name_contains:
        key = name_contains.lower()
        for info in pa.get_loopback_device_info_generator():
            if key in info["name"].lower():
                return info

    if def_out.get("isLoopbackDevice"):
        return def_out

    for info in pa.get_loopback_device_info_generator():
        if def_out["name"] in info["name"]:
            return info

    for info in pa.get_loopback_device_info_generator():
        return info

    raise RuntimeError("No WASAPI loopback device found. Run `python -m pyaudiowpatch` to inspect devices.")

# ---------- MIC (sounddevice @ 16k mono) ----------
def mic_callback(indata, frames, t, status):
    if status:
        print("[Mic] Status:", status)
    # indata: float32 shape (frames, 1)
    x = indata.reshape(-1).astype(np.float32)
    try:
        q_mic.put(x, timeout=0.5)
    except queue.Full:
        pass

def mic_thread(device_index=None, stop_evt: threading.Event = None):
    kwargs = dict(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK, callback=mic_callback)
    if device_index is not None:
        kwargs["device"] = device_index
    try:
        with sd.InputStream(**kwargs):
            print(f"[Mic     ] sounddevice  @ {SR} Hz, ch=1  (device={device_index})")
            while not (stop_evt and stop_evt.is_set()):
                time.sleep(0.05)
    except Exception:
        print("❌ Mic thread crashed:")
        traceback.print_exc()

# ---------- LOOPBACK (PyAudioWPatch) ----------
def loopback_thread(name_contains: Optional[str], stop_evt: threading.Event):
    pa = pyaudio.PyAudio()
    stream = None
    try:
        dev = pick_wasapi_loopback(pa, name_contains)
        idx = dev["index"]

        # Try 16 kHz int16 first (simplest path)
        try_sr = SR
        try:
            stream = pa.open(format=pyaudio.paInt16,
                             channels=min(2, dev["maxInputChannels"]) or 2,
                             rate=try_sr,
                             input=True,
                             input_device_index=idx,
                             frames_per_buffer=BLOCK)
            mode = "16k"
        except Exception:
            # Fallback: 48 kHz, larger block (3x samples per 128ms)
            if stream:
                stream.close()
            try_sr = 48000
            stream = pa.open(format=pyaudio.paInt16,
                             channels=min(2, dev["maxInputChannels"]) or 2,
                             rate=try_sr,
                             input=True,
                             input_device_index=idx,
                             frames_per_buffer=BLOCK * (try_sr // SR))
            mode = "48k"

        ch = min(2, dev["maxInputChannels"]) or 2
        print(f"[Loopback] ({idx}) {dev['name']}  @ {try_sr} Hz, ch={ch}  mode={mode}")

        # If 48k, we’ll decimate by 3 (boxcar average) to 16k
        deci = (try_sr // SR) if mode == "48k" else 1
        assert (mode == "16k") or (try_sr == 48000 and deci == 3)

        frames_per_read = BLOCK if mode == "16k" else BLOCK * deci

        rem = np.zeros(0, dtype=np.float32)
        while not stop_evt.is_set():
            try:
                data = stream.read(frames_per_read, exception_on_overflow=False)
            except Exception as e:
                print("[Loopback] stream.read failed:", repr(e))
                break

            xi = np.frombuffer(data, dtype=np.int16)
            if xi.size == 0:
                continue

            if ch > 1:
                xi = xi.reshape(-1, ch)
                L = xi[:, 0].astype(np.float32)
                R = xi[:, 1].astype(np.float32)
                # phase-safe mono (on int16 domain, fine)
                rmsL = float(np.sqrt(np.mean(L*L)) + 1e-6)
                rmsR = float(np.sqrt(np.mean(R*R)) + 1e-6)
                corr = float(np.sum(L*R) / (len(L)*rmsL*rmsR))
                mono = (L if corr < 0.2 and rmsL >= rmsR else R if corr < 0.2 else 0.7*L + 0.7*R)
            else:
                mono = xi.astype(np.float32)

            # int16 -> float32 [-1,1]
            mono = np.clip(mono / 32768.0, -1.0, 1.0)

            # If 48k, cheap decimate-by-3 (boxcar)
            if deci == 3:
                # pad to multiple of 3
                r = mono.size % 3
                if r:
                    mono = np.pad(mono, (0, 3 - r), mode="constant")
                mono = mono.reshape(-1, 3).mean(axis=1).astype(np.float32)

            # Accumulate to 2048-sample @ 16k
            if rem.size:
                mono = np.concatenate([rem, mono])
            n_full = (mono.size // BLOCK) * BLOCK
            if n_full:
                for b in mono[:n_full].reshape(-1, BLOCK):
                    b = np.clip(np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                    try:
                        q_loop.put(b, timeout=0.5)
                    except queue.Full:
                        pass
                rem = mono[n_full:]
            else:
                rem = mono
    except Exception:
        print("❌ Loopback thread crashed:")
        traceback.print_exc()
    finally:
        try:
            if stream: stream.stop_stream(); stream.close()
        except Exception:
            pass
        pa.terminate()
        print("[Loopback] Stopped.")

# ---------- Writer (only when both have data) ----------
def record_call(out_path: str, stop_evt: threading.Event):
    """
    Clocked writer (one BLOCK @ SR per tick) with *conditional* draining:
    - If only one source has data, take EXACTLY one block from that source; do NOT drain.
    - If BOTH sources have data and either queue is backlogged, drain that queue to the latest.
    This preserves continuous mic when loopback is idle (no “pre-roll chop”), while still
    preventing latency build-up when both are flowing.
    """
    out_path = str(Path(out_path).resolve())
    chans = 2 if WRITE_STEREO else 1
    wav = sf.SoundFile(out_path, mode="w", samplerate=SR, channels=chans, subtype="PCM_16")
    print(f"[Writer ] → {out_path}  (channels={chans}, sr={SR})")

    # Warm up: ensure we have some MIC data so pre-roll is smooth
    warm_deadline = time.time() + 1.5
    while time.time() < warm_deadline and not stop_evt.is_set():
        if q_mic.qsize() >= 2:   # wait for mic to have at least two blocks
            break
        time.sleep(0.01)

    block_dur = BLOCK / float(SR)
    next_t = time.perf_counter()
    written_blocks = 0

    # Threshold above which we allow draining to "latest" to avoid lag
    BACKLOG_HIGH = 8  # ~1s worth at 128 ms/block

    try:
        while not stop_evt.is_set():
            mic_q = q_mic.qsize()
            loop_q = q_loop.qsize()

            # ---- MIC: fetch ONE block, but drain only if both sides are active and mic is backlogged
            try:
                if mic_q > 0:
                    m = q_mic.get_nowait()
                    if loop_q > 0 and mic_q > BACKLOG_HIGH:
                        # drain extras to keep latency small only when both are active
                        while q_mic.qsize() > 1:
                            m = q_mic.get_nowait()
                else:
                    m = np.zeros(BLOCK, dtype=np.float32)
            except queue.Empty:
                m = np.zeros(BLOCK, dtype=np.float32)

            # ---- LOOPBACK: same policy
            try:
                if loop_q > 0:
                    l = q_loop.get_nowait()
                    if mic_q > 0 and loop_q > BACKLOG_HIGH:
                        while q_loop.qsize() > 1:
                            l = q_loop.get_nowait()
                else:
                    l = np.zeros(BLOCK, dtype=np.float32)
            except queue.Empty:
                l = np.zeros(BLOCK, dtype=np.float32)

            if WRITE_STEREO:
                stereo = np.column_stack((l, m)).astype(np.float32)
                wav.write(np.clip(stereo, -1.0, 1.0))
                written_blocks += 1
                if PRINT_METER_EVERY and (written_blocks % PRINT_METER_EVERY == 0):
                    s = stereo.mean(axis=1)
                    rms = float(np.sqrt(np.mean(s*s)) + 1e-12)
                    print(f"\r🔊 level ~ {20*np.log10(rms):6.1f} dBFS", end="", flush=True)
            else:
                mixed = np.clip(l + m, -1.0, 1.0)
                wav.write(mixed)
                written_blocks += 1
                if PRINT_METER_EVERY and (written_blocks % PRINT_METER_EVERY == 0):
                    rms = float(np.sqrt(np.mean(mixed*mixed)) + 1e-12)
                    print(f"\r🔊 level ~ {20*np.log10(rms):6.1f} dBFS", end="", flush=True)

            # pace to real time
            next_t += block_dur
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()
    finally:
        wav.close()
        print("\n[Writer ] Saved & closed.")

# ---------- main ----------
def main():
    stop_evt = threading.Event()
    t_mic  = threading.Thread(target=mic_thread,      kwargs={"device_index": MIC_DEVICE_INDEX, "stop_evt": stop_evt}, daemon=True)
    t_loop = threading.Thread(target=loopback_thread, kwargs={"name_contains": LOOPBACK_NAME_CONTAINS, "stop_evt": stop_evt}, daemon=True)
    t_mic.start(); t_loop.start()

    print("🎛️ Recording (Loopback + Mic) → WAV. Press Ctrl+C to stop.")
    try:
        record_call(OUT_PATH, stop_evt)
    except KeyboardInterrupt:
        print("\n🛑 Stopping…")
    finally:
        stop_evt.set()
        t_mic.join(timeout=1.0)
        t_loop.join(timeout=1.0)

if __name__ == "__main__":
    main()
