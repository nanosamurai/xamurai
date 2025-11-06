import time
import queue
import threading
import tempfile
import os
import numpy as np
import sounddevice as sd
import pyaudiowpatch as pyaudio
import soundfile as sf
import torch, torchaudio
from faster_whisper import WhisperModel

SR = 16000
TARGET_SR = 16000
OUT_BLOCK = 2048
BLOCK = 2048            # ~128 ms
WINDOW_SEC = 5.0        # window length
OVERLAP_SEC = 0.5       # overlap between consecutive windows
HOP_SIZE = WINDOW_SEC - OVERLAP_SEC
LANG = None             # or set e.g. "cs"

# ---- audio ring buffer ----
buf = np.zeros(0, dtype=np.float32)
q = queue.Queue()

def audio_cb(indata, frames, t, status):
    if status:
        print(status)
    q.put(indata.copy())

def audio_thread(device_index=None):
    kwargs = dict(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK, callback=audio_cb)
    if device_index is not None:
        kwargs["device"] = device_index
    with sd.InputStream(**kwargs):
        print("🎙️ listening … Ctrl+C to stop")
        while True:
            time.sleep(0.1)

def _pick_wasapi_loopback(p: pyaudio.PyAudio, name_contains: str | None = None):
    """Return device info for the default speakers' loopback (or first matching by substring)."""
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    def_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])

    # Prefer explicit match if requested
    if name_contains:
        name_contains = name_contains.lower()
        for info in p.get_loopback_device_info_generator():
            if name_contains in info["name"].lower():
                return info

    # Otherwise try to find the loopback that matches the default speakers' name
    if def_out.get("isLoopbackDevice"):
        return def_out
    for info in p.get_loopback_device_info_generator():
        if def_out["name"] in info["name"]:
            return info

    # Fallback: first available loopback
    for info in p.get_loopback_device_info_generator():
        return info

    raise RuntimeError("No WASAPI loopback device found. Run `python -m pyaudiowpatch` to inspect devices.")

def audio_loopback_thread_pyaudio(name_contains: str | None = None):
    """Capture system audio via WASAPI loopback → mono 16 kHz → enqueue 2048-sample chunks to q."""
    pa = pyaudio.PyAudio()
    try:
        dev = _pick_wasapi_loopback(pa, name_contains=name_contains)
        in_rate = int(dev["defaultSampleRate"])
        in_ch   = min(2, dev["maxInputChannels"]) or 2
        in_block = 4096  # device frames per read; not critical, we reblock to 2048 @ 16 k

        # Open float32 input stream
        stream = pa.open(format=pyaudio.paFloat32,
                         channels=in_ch,
                         rate=in_rate,
                         input=True,
                         input_device_index=dev["index"],
                         frames_per_buffer=in_block)

        # Streaming resampler to 16k
        resampler = torchaudio.transforms.Resample(orig_freq=in_rate, new_freq=SR, dtype=torch.float32)

        print(f"🔁 Loopback: ({dev['index']}) {dev['name']}  @ {in_rate} Hz → {SR} Hz")
        out_rem = np.zeros(0, dtype=np.float32)

        while True:
            # Read interleaved float32
            data = stream.read(in_block, exception_on_overflow=False)
            x = np.frombuffer(data, dtype=np.float32)
            if in_ch > 1:
                x = x.reshape(-1, in_ch)
                L = x[:, 0]
                R = x[:, 1]
                # phase-safe mono mix
                rmsL = float(np.sqrt(np.mean(L*L)) + 1e-12)
                rmsR = float(np.sqrt(np.mean(R*R)) + 1e-12)
                corr = float(np.sum(L*R) / (len(L)*rmsL*rmsR))
                mono = (L if corr < 0.2 and rmsL >= rmsR else R if corr < 0.2 else 0.7*L + 0.7*R).astype(np.float32)
            else:
                mono = x.astype(np.float32)

            # Resample → 16k mono
            y = resampler(torch.from_numpy(mono).unsqueeze(0)).squeeze(0).numpy()

            # Accumulate to exact 2048-sample chunks and enqueue
            if out_rem.size:
                y = np.concatenate([out_rem, y])
            n_full = (y.size // BLOCK) * BLOCK
            if n_full:
                for ch16 in y[:n_full].reshape(-1, BLOCK):
                    ch16 = np.clip(np.nan_to_num(ch16, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                    # one-line live meter to check dB levels of the sound:
                    # print(f"\r{20*np.log10(np.sqrt(np.mean(ch16**2))+1e-12):6.1f} dBFS", end="")
                    q.put(ch16.copy())
                out_rem = y[n_full:]
            else:
                out_rem = y
    finally:
        try:
            stream.stop_stream(); stream.close()
        except Exception:
            pass
        pa.terminate()

# ---- ASR model ----
model = WhisperModel("medium", device="cuda", compute_type="float16")

last_printed = 0.0          # last absolute time we printed up to (seconds)
base_offset_sec = 0.0       # absolute time of the LEFT edge of `buf` (monotonic)

WINDOW_SAMPLES = int(WINDOW_SEC * SR)
HOP_SAMPLES = int(HOP_SIZE * SR)

def transcribe_chunk(x: np.ndarray, offset_sec: float):
    """
    Transcribe x (float32, 16k mono) whose logical time span is
    [offset_sec, offset_sec + len(x)/SR]. Print only NEW words whose
    start time is after `last_printed` (prevents drops across overlaps).
    """
    global last_printed
    if x.size < int(0.5 * SR):
        return

    # OPTIONAL: quick silence gate to skip empty windows
    if np.sqrt(np.mean(x * x)) < 1.0e-4:
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, x, SR, subtype="PCM_16")

    try:
        # word-level timestamps are the key to not missing anything
        segs, _info = model.transcribe(
            tmp_path,
            language=LANG,            # e.g., "cs" recommended for stability
            task="transcribe",
            beam_size=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,   # strict incremental
            vad_filter=True,                   # we manage windows ourselves
            word_timestamps=True
        )

        NEW_EPS = 1e-3
        printed_any = False
        newest_time = last_printed

        # Collect only words that are truly new in absolute time
        new_words = []
        for seg in segs:
            if not hasattr(seg, "words") or not seg.words:
                # Fallback: if no word timings, use segment timing conservatively
                st_abs = offset_sec + seg.start
                et_abs = offset_sec + seg.end
                if et_abs > last_printed + NEW_EPS:
                    new_words.append(seg.text.strip())
                    newest_time = max(newest_time, et_abs)
                continue

            for w in seg.words:
                w_start = offset_sec + (w.start or seg.start)
                w_end   = offset_sec + (w.end   or seg.end)
                if w_start > last_printed + NEW_EPS:
                    new_words.append(w.word)
                    newest_time = max(newest_time, w_end)

        if new_words:
            print(" ".join(new_words).strip(), flush=True)
            last_printed = newest_time
            printed_any = True

    except Exception as e:
        print(f"Error during transcription: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def main():
    global buf, base_offset_sec

    #t_rec = threading.Thread(target=audio_thread, kwargs={"device_index": 1}, daemon=True)
    #'Speakers (Realtek(R) Audio)' or "Headphones (soundcore Liberty 4 NC)"
    t_rec = threading.Thread(target=audio_loopback_thread_pyaudio, kwargs={"name_contains": None}, daemon=True)
    t_rec.start()

    try:
        while True:
            # Drain audio queue into buf
            while not q.empty():
                block = q.get_nowait()
                buf = np.concatenate([buf, block.reshape(-1)])

            # Emit as many windows as we have, advancing by HOP each time
            while len(buf) >= WINDOW_SAMPLES:
                window = buf[:WINDOW_SAMPLES]
                window_offset = base_offset_sec               # absolute start time of this window
                # simple RMS gate
                if np.sqrt(np.mean(window * window)) < 1.0e-4:
                    pass
                else:
                    transcribe_chunk(window, window_offset)

                # advance buffer by hop; keep overlap in the remaining buffer
                buf = buf[HOP_SAMPLES:]
                base_offset_sec += HOP_SAMPLES / SR           # move the left-edge time forward

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("bye.")

if __name__ == "__main__":
    main()
