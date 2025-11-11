#!/usr/bin/env python3
"""
Streaming diarization-first pipeline with WASAPI loopback, mic, or both.

Key points:
- In "loopback" or "mic" mode:
    - single source → 16k mono → rolling windows → diarization → ASR.
- In "both" mode:
    - mic and loopback are captured separately,
    - resampled to 16k mono,
    - time-aligned and MIXED (summed) into one mono stream,
    - then same diarization+ASR pipeline.
- Center-based "owner window" logic prevents dupes across overlaps.
- Optional speaker enrollment to label diar clusters.
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
BLOCK = 2048                  # processing block (samples @16k)
WINDOW_SEC = 5.0              # diarization/ASR window size
OVERLAP_SEC = 0.5             # overlap between windows
HOP_SEC = WINDOW_SEC - OVERLAP_SEC

LANG = "en"                   # "cs", "en", or None

FINALIZE_MIN_DUR_SEC = 0.25   # ignore tiny segments

# Capture mode: "loopback", "mic", or "both"
CAPTURE_MODE = "both"

# Mic device selection (None = default). Use sd.query_devices() if needed.
MIC_DEVICE_INDEX = None

# Loopback selection by substring in device name, e.g. "Headphones" or None for default.
LOOPBACK_NAME_CONTAINS = None

# Optional coarse VAD on each window
USE_SILERO_VAD = False
MIN_SPEECH_IN_WINDOW_SEC = 0.35
VAD_MIN_SPEECH_MS = 350
VAD_MIN_SILENCE_MS = 250

# Speaker enrollment
USE_SPEAKER_ENROLLMENT = True
ENROLL_DIR = "enrolled_speakers"
ENROLL_SIM_THRESHOLD = 0.35 #0.5

# Debug toggles
DEBUG_SEGMENTS = False
DEBUG_SPEAKER_MATCH = False

# ===================== GLOBAL STATE =====================

# For single-source modes
q = queue.Queue()

# For "both" mode: separate queues
q_mic = queue.Queue()
q_loop = queue.Queue()

# Rolling analysis buffer (mixed mono stream)
buf = np.zeros(0, dtype=np.float32)

WINDOW_SAMPLES = int(WINDOW_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)

# Time of LEFT edge of `buf` in seconds
base_offset_sec = 0.0

# Window index for center-ownership
window_index = 0

# Dedup across overlapping windows: coarse (start,end,label)
KEY_RES = 0.25
emitted_keys: set[tuple[float, float, str]] = set()

# VAD globals
vad_model = None
get_speech_timestamps = None

# Enrollment globals
embedding_infer: EmbeddingInference | None = None
enrolled_speakers: dict[str, np.ndarray] = {}


# ===================== AUDIO INPUT HELPERS =====================

def _pick_wasapi_loopback(p: pyaudio.PyAudio, name_contains: str | None = None):
    """Pick WASAPI loopback device."""
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

    raise RuntimeError("No WASAPI loopback device found.")


# ===================== CAPTURE THREADS =====================

def mic_callback(indata, frames, t, status):
    if status:
        print(status)
    if CAPTURE_MODE == "both":
        q_mic.put(indata.copy())
    else:
        q.put(indata.copy())

def mic_capture_thread(device_index=None):
    """Mic → 16k mono → BLOCK chunks via callback."""
    kwargs = dict(
        samplerate=SR,
        channels=1,
        dtype="float32",
        blocksize=BLOCK,
        callback=mic_callback,
    )
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


def loopback_capture_thread(name_contains: str | None = None):
    """System audio via WASAPI loopback → 16k mono → BLOCK chunks."""
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

            # Mono mix of device channels
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

            # Resample to 16k
            y = resampler(torch.from_numpy(mono).unsqueeze(0)).squeeze(0).numpy()
            if out_rem.size:
                y = np.concatenate([out_rem, y])

            n_full = (y.size // BLOCK) * BLOCK
            if n_full:
                for ch in y[:n_full].reshape(-1, BLOCK):
                    ch = np.clip(
                        np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0),
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                    if CAPTURE_MODE == "both":
                        q_loop.put(ch.copy())
                    else:
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
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        pa.terminate()
        print("🔁 Loopback thread terminated.")


# ===================== MODELS =====================

HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Set HF_TOKEN environment variable."

asr = WhisperModel(
    "medium",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8_float32",
)

pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN,
)
pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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

if USE_SPEAKER_ENROLLMENT:
    try:
        emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
        embedding_infer = EmbeddingInference(emb_model, window="whole")
    except Exception as e:
        print(f"⚠️ Failed to load embedding model: {e}")
        embedding_infer = None
        USE_SPEAKER_ENROLLMENT = False


# ===================== HELPERS =====================

def round_time(t: float, resolution: float) -> float:
    return round(t / resolution) * resolution

def diarize_window_in_memory(wave: np.ndarray):
    if wave.size == 0:
        return []
    w = torch.from_numpy(wave.copy()).unsqueeze(0)
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


# ====== Enrollment ======

def load_enrolled_speakers():
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


# ===================== MAIN MIX + PIPELINE =====================

def mix_two_sources_into_buf():
    """For CAPTURE_MODE='both': mix q_mic + q_loop into global buf."""
    global buf

    # Local ring buffers for each source
    if not hasattr(mix_two_sources_into_buf, "mic_buf"):
        mix_two_sources_into_buf.mic_buf = np.zeros(0, dtype=np.float32)
        mix_two_sources_into_buf.loop_buf = np.zeros(0, dtype=np.float32)

    mic_buf = mix_two_sources_into_buf.mic_buf
    loop_buf = mix_two_sources_into_buf.loop_buf

    # Drain queues
    while True:
        try:
            block = q_mic.get_nowait()
            mic_buf = np.concatenate([mic_buf, block.reshape(-1)])
        except queue.Empty:
            break

    while True:
        try:
            block = q_loop.get_nowait()
            loop_buf = np.concatenate([loop_buf, block.reshape(-1)])
        except queue.Empty:
            break

    # Produce mixed BLOCKs
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
            l[:len(loop_buf)] += loop_buf
            loop_buf = loop_buf[0:0]

        mixed = np.clip(m + l, -1.0, 1.0)
        buf = np.concatenate([buf, mixed])

    mix_two_sources_into_buf.mic_buf = mic_buf
    mix_two_sources_into_buf.loop_buf = loop_buf


def main():
    global buf, base_offset_sec, window_index

    if USE_SPEAKER_ENROLLMENT:
        load_enrolled_speakers()

    # Start capture threads
    if CAPTURE_MODE == "mic":
        print("🎛️ Mode: MIC only")
        threading.Thread(
            target=mic_capture_thread,
            kwargs={"device_index": MIC_DEVICE_INDEX},
            daemon=True,
        ).start()

    elif CAPTURE_MODE == "loopback":
        print("🎛️ Mode: LOOPBACK only")
        threading.Thread(
            target=loopback_capture_thread,
            kwargs={"name_contains": LOOPBACK_NAME_CONTAINS},
            daemon=True,
        ).start()

    elif CAPTURE_MODE == "both":
        print("🎛️ Mode: BOTH (mic + loopback)")
        threading.Thread(
            target=mic_capture_thread,
            kwargs={"device_index": MIC_DEVICE_INDEX},
            daemon=True,
        ).start()
        threading.Thread(
            target=loopback_capture_thread,
            kwargs={"name_contains": LOOPBACK_NAME_CONTAINS},
            daemon=True,
        ).start()
    else:
        raise ValueError(f"Unknown CAPTURE_MODE={CAPTURE_MODE!r}")

    try:
        while True:
            # 1) Ingest audio into buf
            if CAPTURE_MODE == "both":
                mix_two_sources_into_buf()
            else:
                # single-source: drain q into buf directly
                while True:
                    try:
                        block = q.get_nowait()
                    except queue.Empty:
                        break
                    buf = np.concatenate([buf, block.reshape(-1)])

            # 2) Process complete windows
            while len(buf) >= WINDOW_SAMPLES:
                window = buf[:WINDOW_SAMPLES]
                window_offset = base_offset_sec
                my_idx = window_index

                if maybe_gate_window_with_vad(window):
                    diar_segs = diarize_window_in_memory(window)

                    for d in diar_segs:
                        seg_start = d["start"]
                        seg_end = d["end"]
                        dur = seg_end - seg_start
                        if dur < FINALIZE_MIN_DUR_SEC:
                            continue

                        # Absolute times
                        s_abs = window_offset + seg_start
                        e_abs = window_offset + seg_end
                        center_abs = 0.5 * (s_abs + e_abs)

                        # Unique owner window via segment center
                        owner_idx = int(center_abs / HOP_SEC + 1e-6)
                        if owner_idx != my_idx:
                            continue

                        key = (
                            round_time(s_abs, KEY_RES),
                            round_time(e_abs, KEY_RES),
                            d["speaker"],
                        )
                        if key in emitted_keys:
                            continue

                        # Cut from this window
                        s_idx = int(seg_start * SR)
                        e_idx = int(seg_end * SR)
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

                # Slide window
                buf = buf[HOP_SAMPLES:]
                base_offset_sec += HOP_SAMPLES / SR
                window_index += 1

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("bye.")

if __name__ == "__main__":
    main()
