import time, queue, threading, tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

SR = 16000
BLOCK = 2048           # ~128 ms
WINDOW_SEC = 5.0       # analyze this much each tick
TICK_SEC = 2.0         # how often to analyze
OVERLAP_SEC = 0.5      # small context overlap
LANG = None            # set your language (None = autodetect)

# ---- audio ring buffer ----
buf = np.zeros(0, dtype=np.float32)
q = queue.Queue()

def audio_cb(indata, frames, t, status):
    if status: print(status)
    q.put(indata.copy())

def audio_thread(device_index=None):
    kwargs = dict(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK, callback=audio_cb)
    if device_index is not None: kwargs["device"] = device_index
    with sd.InputStream(**kwargs):
        print("🎙️ listening … Ctrl+C to stop")
        while True: time.sleep(0.1)

# ---- ASR model ----
model = WhisperModel("medium", device="cuda", compute_type="float16")

last_printed = 0.0  # global timeline in seconds of the stream

def transcribe_chunk(x: np.ndarray, offset_sec: float):
    """Transcribe x (float32, 16k mono), whose logical time span is [offset_sec, offset_sec+len(x)/SR].
       Print only segments with start > last_printed (global)."""
    global last_printed
    if x.size < int(0.5*SR): return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, x, SR, subtype="PCM_16")
    segs, info = model.transcribe(
        tmp_path,
        language=LANG,
        task="transcribe",
        beam_size=5,
        temperature=[0.0, 0.2, 0.4],
        condition_on_previous_text=False,   # keep strictly incremental
        vad_filter=True,
    )
    for s in segs:
        st = offset_sec + s.start
        et = offset_sec + s.end
        if st > last_printed + 1e-3:
            print(s.text.strip(), flush=True)
            last_printed = et
    try: import os; os.unlink(tmp_path)
    except: pass

def main():
    global buf
    t_rec = threading.Thread(target=audio_thread, kwargs={"device_index": 34}, daemon=True)
    t_rec.start()
    t0 = time.time()
    try:
        while True:
            # drain audio queue
            while True:
                try:
                    block = q.get_nowait()
                except queue.Empty:
                    break
                buf = np.concatenate([buf, block.reshape(-1)])
                # bound buffer to 60 s
                if len(buf) > 60*SR: buf = buf[-60*SR:]

            now = time.time()
            # tick every TICK_SEC
            if now - t0 >= TICK_SEC:
                t0 = now
                # take last WINDOW_SEC (+ small overlap to stabilize)
                need = int(WINDOW_SEC*SR)
                if len(buf) >= int(1.0*SR):  # wait for at least 1s of audio
                    x = buf[-need:] if len(buf) >= need else buf
                    # compute the logical offset of x in stream time
                    total_sec = len(buf)/SR
                    x_offset = max(0.0, total_sec - len(x)/SR)
                    # prepend small overlap for acoustics (we won’t reprint due to last_printed gate)
                    ov = int(OVERLAP_SEC*SR)
                    if len(buf) > len(x) and ov > 0:
                        x = np.concatenate([buf[-len(x)-ov:-len(x)], x])
                        x_offset -= OVERLAP_SEC
                        if x_offset < 0: x_offset = 0.0
                    transcribe_chunk(x, x_offset)

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("bye.")

if __name__ == "__main__":
    main()
