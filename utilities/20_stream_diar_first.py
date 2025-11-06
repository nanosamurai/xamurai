#!/usr/bin/env python3
"""
Streaming diarization-first pipeline with WASAPI loopback or mic.

Core behavior:
- Capture audio (loopback or mic) -> 16 kHz mono
- Rolling windows (WINDOW_SEC, HOP_SEC) over stream
- For each window:
    - (optional) VAD gate to skip silence-heavy windows
    - pyannote/speaker-diarization-community-1 on in-memory waveform
    - convert segments to ABSOLUTE times
    - finalize segments whose end is safely behind the live audio ("now")
    - run faster-whisper on each finalized segment (vad_filter=False)
    - print exactly once (dedupe across overlapping windows)
- Optional speaker enrollment:
    - load short wavs from ENROLL_DIR to build name -> embedding
    - map diarization speakers to names by cosine similarity
"""

import os
import time
import queue
import threading
import tempfile
import traceback
from collections import defaultdict

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Model
from pyannote.audio import Inference as EmbeddingInference

# ===================== CONFIG =====================

SR = 16000
BLOCK = 2048                  # internal processing block (samples)
WINDOW_SEC = 5.0              # diarization/ASR analysis window
OVERLAP_SEC = 0.5             # overlap; HOP = WINDOW - OVERLAP
HOP_SEC = WINDOW_SEC - OVERLAP_SEC

LANG = None                   # ASR language (set None for autodetect if you like)

FINALIZE_TAIL_SEC = 0.30      # segment must end this far before "now" to be emitted
FINALIZE_MIN_DUR_SEC = 0.25   # skip tiny segments

# VAD over whole window (for efficiency, optional)
USE_SILERO_VAD = False        # keep False until recall is verified
MIN_SPEECH_IN_WINDOW_SEC = 0.35
VAD_MIN_SPEECH_MS = 350
VAD_MIN_SILENCE_MS = 250

# Speaker enrollment
USE_SPEAKER_ENROLLMENT = True
ENROLL_DIR = "enrolled_speakers"   # put Jason.wav, Alex_1.wav etc here
ENROLL_SIM_THRESHOLD = 0.6         # cosine similarity cutoff

# Debug toggles (set True if you want to inspect behavior)
DEBUG_SEGMENTS = True
DEBUG_SPEAKER_MATCH = True

# ===================== GLOBAL STATE =====================

buf = np.zeros(0, dtype=np.float32)
q: "queue.Queue[np.ndarray]" = queue.Queue()

WINDOW_SAMPLES = int(WINDOW_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)

# Absolute time (sec) of LEFT edge of buf
base_offset_sec = 0.0

# Track which segments we have already printed:
# key = (round(start, 2), round(end, 2), label)
emitted_keys: set[tuple[float, float, str]] = set()

# VAD globals
vad_model = None
get_speech_timestamps = None

# Enrollment globals
embedding_infer: EmbeddingInference | None = None
enrolled_speakers: dict[str, np.ndarray] = {}  # name -> L2-normalized embedding

# ===================== AUDIO INPUT =====================

def audio_cb(indata, frames, t, status):
    if status:
        print(status)
    q.put(indata.copy())

def audio_thread(device_index=None):
    kwargs = dict(
        samplerate=SR,
        channels=1,
        dtype="float32",
        blocksize=BLOCK,
        callback=audio_cb,
    )
    if device_index is not None:
        kwargs["device"] = device_index
    with sd.InputStream(**kwargs):
        print("🎙️ Mic listening… Ctrl+C to stop")
        while True:
            time.sleep(0.1)

def _pick_wasapi_loopback(p: pyaudio.PyAudio, name_contains: str | None = None):
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    def_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])

    if name_contains:
        name_l = name_contains.lower()
        for info in p.get_loopback_device_info_generator():
            if name_l in info["name"].lower():
                return info

    if def_out.get("isLoopbackDevice"):
        return def_out

    for info in p.get_loopback_device_info_generator():
        if def_out["name"] in info["name"]:
            return info

    for info in p.get_loopback_device_info_generator():
        return info

    raise RuntimeError("No WASAPI loopback device found for system audio.")

def audio_loopback_thread_pyaudio(name_contains: str | None = None):
    """Capture system audio via WASAPI loopback → 16k mono → BLOCK chunks into q."""
    print("🔁 Loopback thread starting…")
    pa = pyaudio.PyAudio()
    stream = None
    try:
        dev = _pick_wasapi_loopback(pa, name_contains)
        in_rate = int(dev["defaultSampleRate"])
        in_ch = min(2, dev["maxInputChannels"]) or 2
        in_block = 4096

        print(f"🔁 Opening WASAPI loopback: ({dev['index']}) {dev['name']} @ {in_rate} Hz")

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=in_ch,
            rate=in_rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=in_block,
        )

        resampler = torchaudio.transforms.Resample(orig_freq=in_rate, new_freq=SR)

        print(f"🔁 Loopback running: {in_rate} Hz → {SR} Hz (BLOCK={BLOCK})")
        out_rem = np.zeros(0, dtype=np.float32)

        while True:
            data = stream.read(in_block, exception_on_overflow=False)
            x = np.frombuffer(data, dtype=np.float32)
            if x.size == 0:
                continue

            # Mono
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

            y = resampler(torch.from_numpy(mono).unsqueeze(0)).squeeze(0).numpy()

            if out_rem.size:
                y = np.concatenate([out_rem, y])

            n_full = (y.size // BLOCK) * BLOCK
            if n_full:
                for ch in y[:n_full].reshape(-1, BLOCK):
                    ch = np.clip(
                        np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0),
                        -1.0, 1.0,
                    ).astype(np.float32)
                    q.put(ch.copy())
                out_rem = y[n_full:]
            else:
                out_rem = y

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

# ===================== MODELS =====================

HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Set HF_TOKEN environment variable."

# ASR
asr = WhisperModel(
    "medium",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8_float32",
)

# Diarization
pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN,
)
pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Silero VAD (optional)
if USE_SILERO_VAD:
    _vad_model, _vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = _vad_utils
    vad_model = _vad_model

# Speaker embeddings (optional)
if USE_SPEAKER_ENROLLMENT:
    try:
        emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
        embedding_infer = EmbeddingInference(emb_model, window="whole")
    except Exception as e:
        print(f"⚠️ Failed to load embedding model: {e}")
        embedding_infer = None
        USE_SPEAKER_ENROLLMENT = False

# ===================== HELPERS =====================

def round_time(t: float, resolution: float = 0.02) -> float:
    return round(t / resolution) * resolution

def diarize_window_in_memory(wave: np.ndarray):
    """Run Community-1 on in-memory waveform (float32, (T,))."""
    if wave.size == 0:
        return []
    w = torch.from_numpy(wave.copy()).unsqueeze(0)  # (1, T)
    out = pipe({"waveform": w, "sample_rate": SR})
    segs = []
    for turn, spk in out.speaker_diarization:
        segs.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(spk),
        })
    if DEBUG_SEGMENTS:
        print(f"[DEBUG] diar segments in window: {segs}")
    return segs

def asr_text_on_chunk(wave: np.ndarray) -> str:
    """ASR on a speech chunk."""
    if wave.size < int(FINALIZE_MIN_DUR_SEC * SR):
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    sf.write(path, wave, SR, subtype="PCM_16")
    try:
        segs, _ = asr.transcribe(
            path,
            language=LANG,
            task="transcribe",
            beam_size=5,
            temperature=[0.0],
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=True,
        )
        words = []
        for s in segs:
            if getattr(s, "words", None):
                for w in s.words:
                    if w.word.strip():
                        words.append(w.word)
            elif s.text.strip():
                words.append(s.text.strip())
        return " ".join(words).strip()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

def maybe_gate_window_with_vad(window: np.ndarray) -> bool:
    """Skip obviously silent windows (to save compute)."""
    # gentle RMS gate
    if np.sqrt(np.mean(window * window)) < 3e-5:
        return False
    if not USE_SILERO_VAD or vad_model is None:
        return True

    wav16 = (np.clip(window, -1.0, 1.0) * 32767).astype(np.int16)
    ts = get_speech_timestamps(
        wav16,
        vad_model,
        sampling_rate=SR,
        return_seconds=True,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
    )
    if not ts:
        return False
    speech_sec = sum(max(0.0, t["end"] - t["start"]) for t in ts)
    return speech_sec >= MIN_SPEECH_IN_WINDOW_SEC

# -------- Speaker enrollment --------

def load_enrolled_speakers():
    """Build name -> embedding from wavs in ENROLL_DIR."""
    global enrolled_speakers
    if not USE_SPEAKER_ENROLLMENT or embedding_infer is None:
        print("👤 Speaker enrollment disabled or unavailable.")
        return

    if not os.path.isdir(ENROLL_DIR):
        print(f"👤 Enrollment dir '{ENROLL_DIR}' not found; skipping.")
        return

    per_name = defaultdict(list)
    for fname in os.listdir(ENROLL_DIR):
        if not fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
            continue
        path = os.path.join(ENROLL_DIR, fname)
        base = os.path.splitext(fname)[0]
        name = base.split("_")[0]

        try:
            # Let EmbeddingInference handle decoding from file path
            emb = embedding_infer(path)
            emb = np.array(emb, dtype=np.float32)
            if emb.ndim > 1:
                emb = emb.mean(axis=0)
            per_name[name].append(emb)
            print(f"👤 Enrolled segment for '{name}' from {fname}")
        except Exception as e:
            print(f"⚠️ Failed to enroll from {fname}: {e}")

    for name, embs in per_name.items():
        if not embs:
            continue
        arr = np.stack(embs, axis=0)
        mean_emb = arr.mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb) + 1e-12
        enrolled_speakers[name] = mean_emb

    if enrolled_speakers:
        print("👤 Enrollment complete. Known speakers:", ", ".join(enrolled_speakers.keys()))
    else:
        print("👤 No valid enrollment samples found.")

def embed_wave(wave: np.ndarray) -> np.ndarray | None:
    """Get L2-normalized embedding for in-memory wave segment."""
    if embedding_infer is None or wave.size < int(0.25 * SR):
        return None
    try:
        audio = {"waveform": torch.from_numpy(wave).unsqueeze(0), "sample_rate": SR}
        emb = embedding_infer(audio)
        emb = np.array(emb, dtype=np.float32)
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        emb /= np.linalg.norm(emb) + 1e-12
        return emb
    except Exception as e:
        print(f"⚠️ Failed to embed segment: {e}")
        return None

def map_speaker_label(diar_label: str, wave_chunk: np.ndarray) -> str:
    """Map pyannote cluster label to enrolled name if similarity is high enough."""
    if not USE_SPEAKER_ENROLLMENT or not enrolled_speakers:
        return diar_label

    emb = embed_wave(wave_chunk)
    if emb is None:
        return diar_label

    best_name = None
    best_sim = -1.0
    for name, ref in enrolled_speakers.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_sim = sim
            best_name = name

    if DEBUG_SPEAKER_MATCH:
        print(f"[DEBUG] speaker {diar_label}, best={best_name}, sim={best_sim:.3f}")

    if best_name is not None and best_sim >= ENROLL_SIM_THRESHOLD:
        return best_name
    return diar_label

# ===================== MAIN LOOP =====================

def main():
    global buf, base_offset_sec

    if USE_SPEAKER_ENROLLMENT:
        load_enrolled_speakers()

    # Choose ONE source:

    # 1) Mic:
    # t_rec = threading.Thread(target=audio_thread, kwargs={"device_index": None}, daemon=True)

    # 2) System audio via WASAPI loopback:
    t_rec = threading.Thread(
        target=audio_loopback_thread_pyaudio,
        kwargs={"name_contains": None},  # or partial name of your output device
        daemon=True,
    )
    t_rec.start()

    try:
        while True:
            # Pull new audio into buffer
            while True:
                try:
                    block = q.get_nowait()
                except queue.Empty:
                    break
                buf = np.concatenate([buf, block.reshape(-1)])

            # Compute "now" in absolute seconds based on current buffer end
            stream_now = base_offset_sec + len(buf) / SR

            # Process windows
            while len(buf) >= WINDOW_SAMPLES:
                window = buf[:WINDOW_SAMPLES]
                window_offset = base_offset_sec      # abs start of this window

                if maybe_gate_window_with_vad(window):
                    diar_segs = diarize_window_in_memory(window)

                    for d in diar_segs:
                        seg_start_rel = d["start"]
                        seg_end_rel = d["end"]
                        dur = seg_end_rel - seg_start_rel
                        if dur < FINALIZE_MIN_DUR_SEC:
                            continue

                        # Absolute times
                        s_abs = window_offset + seg_start_rel
                        e_abs = window_offset + seg_end_rel

                        # Finalize rule:
                        # segment ended at least FINALIZE_TAIL_SEC before current stream_now
                        if e_abs > stream_now - FINALIZE_TAIL_SEC:
                            continue

                        key = (round_time(s_abs), round_time(e_abs), d["speaker"])
                        if key in emitted_keys:
                            continue

                        # Cut from this window
                        s_idx = int(seg_start_rel * SR)
                        e_idx = int(seg_end_rel * SR)
                        s_idx = max(0, min(len(window), s_idx))
                        e_idx = max(0, min(len(window), e_idx))
                        if e_idx - s_idx < int(FINALIZE_MIN_DUR_SEC * SR):
                            continue

                        wave_chunk = window[s_idx:e_idx]

                        label = map_speaker_label(d["speaker"], wave_chunk)
                        text = asr_text_on_chunk(wave_chunk)
                        if text:
                            print(f"{label}: {text}", flush=True)
                            emitted_keys.add(key)

                # Slide buffer by HOP; overlapping audio stays
                buf = buf[HOP_SAMPLES:]
                base_offset_sec += HOP_SAMPLES / SR

                # Update stream_now after sliding
                stream_now = base_offset_sec + len(buf) / SR

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("bye.")

if __name__ == "__main__":
    main()
