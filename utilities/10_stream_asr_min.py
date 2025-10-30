import time
import queue
import threading
import tempfile
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

    # Write to a temporary file for transcription
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, x, SR, subtype="PCM_16")

    try:
        # Transcribe the audio chunk with VAD filter enabled to handle silence better
        segs, info = model.transcribe(
            tmp_path,
            language=LANG,
            task="transcribe",
            beam_size=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,   # keep strictly incremental
            vad_filter=True,                    # Use VAD filter to avoid hallucinations during silence
        )

        for s in segs:
            st = offset_sec + s.start
            et = offset_sec + s.end
            if st > last_printed + 1e-3:  # Avoid overlapping outputs
                print(s.text.strip(), flush=True)
                last_printed = et

    except Exception as e:
        print(f"Error during transcription: {str(e)}")

    finally:
        try:
            import os
            os.unlink(tmp_path)  # Clean up the temporary file
        except Exception as e:
            print(f"Failed to remove temporary file: {str(e)}")

def main():
    global buf

    t_rec = threading.Thread(target=audio_thread, kwargs={"device_index": 34}, daemon=True)
    t_rec.start()

    t0 = time.time()
    try:
        while True:
            # Drain audio queue
            while not q.empty():
                block = q.get_nowait()
                buf = np.concatenate([buf, block.reshape(-1)])
                print(f"Buffer length after adding block: {len(buf)}")

                # Process the buffer in smaller chunks to avoid memory issues and ensure continuous transcription
                chunk_size = int(WINDOW_SEC * SR)
                while len(buf) >= chunk_size:
                    chunk = buf[:chunk_size]
                    buf = buf[chunk_size:]
                    total_sec = len(chunk) / SR

                    # Add overlap from previous window if needed
                    ov = int(OVERLAP_SEC * SR)
                    if len(buf) > 0 and ov > 0:
                        chunk_with_overlap = np.concatenate([buf[-ov:], chunk])
                    else:
                        chunk_with_overlap = chunk

                    transcribe_chunk(chunk_with_overlap, total_sec - WINDOW_SEC)

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("bye.")

if __name__ == "__main__":
    main()