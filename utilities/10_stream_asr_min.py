import time
import queue
import threading
import tempfile
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

SR = 16000
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

    t_rec = threading.Thread(target=audio_thread, kwargs={"device_index": 1}, daemon=True)
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
