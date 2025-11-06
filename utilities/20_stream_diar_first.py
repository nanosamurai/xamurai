#!/usr/bin/env python3
# Streaming diarization-first pipeline with WASAPI loopback:
# - rolling window (WINDOW/HOP) from mic OR loopback
# - optional Silero VAD gate per window
# - pyannote Community-1 diarization on in-memory waveform
# - finalize speech turns (end before "now - FINALIZE_TAIL")
# - ASR (faster-whisper) per finalized turn (vad_filter=False), word-level
# - dedupe so finalized turns are transcribed exactly once

import os
import time
import math
import queue
import threading
import tempfile
import traceback

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# ---------------- Config ----------------
SR = 16000
BLOCK = 2048                 # internal processing block (samples)
WINDOW_SEC = 5.0             # diarization/ASR window length
OVERLAP_SEC = 0.5            # overlap between consecutive windows
HOP_SEC = WINDOW_SEC - OVERLAP_SEC
LANG = None                  # set language explicitly ("cs", "en", ...)

FINALIZE_TAIL_SEC = 0.30     # only emit turns that ended at least this long ago

# VAD gate (skip windows with too little speech before running pyannote)
USE_SILERO_VAD = True
MIN_SPEECH_IN_WINDOW_SEC = 0.35
VAD_MIN_SPEECH_MS = 350
VAD_MIN_SILENCE_MS = 250

# ---------------- State ----------------
buf = np.zeros(0, dtype=np.float32)
q: "queue.Queue[np.ndarray]" = queue.Queue()

WINDOW_SAMPLES = int(WINDOW_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)

# Absolute time (seconds) of the LEFT edge of `buf`
base_offset_sec = 0.0

# Set of finalized spans we've already transcribed:
# key = (round(start*50)/50, round(end*50)/50, speaker)  ~20ms resolution
emitted_keys = set()


# ---------------- Mic Input (optional) ----------------
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


# ---------------- WASAPI Loopback Input ----------------
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

    raise RuntimeError(
        "No WASAPI loopback device found. Run `python -m pyaudiowpatch` to inspect devices."
    )


def audio_loopback_thread_pyaudio(name_contains: str | None = None):
    """Capture system audio via WASAPI loopback → mono 16 kHz → enqueue BLOCK-sized chunks to q."""
    print("🔁 Loopback thread starting…")
    pa = pyaudio.PyAudio()
    stream = None
    try:
        dev = _pick_wasapi_loopback(pa, name_contains=name_contains)
        in_rate = int(dev["defaultSampleRate"])
        in_ch = min(2, dev["maxInputChannels"]) or 2
        in_block = 4096  # frames per read

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

            # To mono
            if in_ch > 1:
                x = x.reshape(-1, in_ch)
                L = x[:, 0]
                R = x[:, 1]
                rmsL = float(np.sqrt(np.mean(L * L)) + 1e-12)
                rmsR = float(np.sqrt(np.mean(R * R)) + 1e-12)
                corr = float(np.sum(L * R) / (len(L) * rmsL * rmsR))
                if corr < 0.2:  # low correlation: pick stronger channel
                    mono = (L if rmsL >= rmsR else R).astype(np.float32)
                else:           # high correlation: mix
                    mono = (0.7 * L + 0.7 * R).astype(np.float32)
            else:
                mono = x.astype(np.float32)

            # Resample -> 16k
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


# ---------------- Models -----------------
# ASR (faster-whisper)
asr = WhisperModel(
    "medium",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="float16" if torch.cuda.is_available() else "int8_float32",
)

# pyannote Community-1 diarization
HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Set HF_TOKEN environment variable."
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Optional Silero VAD
if USE_SILERO_VAD:
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks,
    ) = vad_utils


# ---------------- Helpers ----------------
def round_time(t: float, resolution: float = 0.02) -> float:
    return round(t / resolution) * resolution


def diarize_window_in_memory(wave: np.ndarray):
    """Run Community-1 on in-memory waveform (float32, shape [T])."""
    import torch as th

    if wave.size == 0:
        return []

    w = th.from_numpy(wave.copy()).view(1, -1)  # (1, T)
    out = pipe({"waveform": w, "sample_rate": SR})
    segs = []
    for turn, spk in out.speaker_diarization:
        segs.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(spk),
            }
        )
    return segs


def asr_text_on_chunk(wave: np.ndarray, abs_offset_sec: float) -> str:
    """Run ASR on a single speech turn (wave in float32 mono)."""
    if wave.size < int(0.25 * SR):
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    sf.write(path, wave, SR, subtype="PCM_16")
    try:
        segments, _ = asr.transcribe(
            path,
            language=LANG,
            task="transcribe",
            beam_size=5,
            temperature=[0.0],
            condition_on_previous_text=False,
            vad_filter=False,           # diarization already gated speech
            word_timestamps=True,
        )
        words = []
        for s in segments:
            if getattr(s, "words", None):
                for w in s.words:
                    if w.word.strip():
                        words.append(w.word)
            else:
                if s.text.strip():
                    words.append(s.text.strip())
        return " ".join(words).strip()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def maybe_gate_window_with_vad(window: np.ndarray) -> bool:
    """Return True if we should process this window (speech present)."""
    # quick RMS prefilter
    if np.sqrt(np.mean(window * window)) < 8e-5:
        return False
    if not USE_SILERO_VAD:
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


# ---------------- Main -------------------
def main():
    global buf, base_offset_sec

    # Choose ONE input:
    # 1) Mic:
    # t_rec = threading.Thread(target=audio_thread, kwargs={"device_index": None}, daemon=True)

    # 2) WASAPI loopback (system audio):
    t_rec = threading.Thread(
        target=audio_loopback_thread_pyaudio,
        kwargs={"name_contains": None},  # or partial device name string
        daemon=True,
    )

    t_rec.start()

    try:
        while True:
            # Drain new audio into buffer
            while True:
                try:
                    block = q.get_nowait()
                except queue.Empty:
                    break
                buf = np.concatenate([buf, block.reshape(-1)])

            # Process as many windows as available
            while len(buf) >= WINDOW_SAMPLES:
                window = buf[:WINDOW_SAMPLES]
                window_offset = base_offset_sec                   # abs start of this window
                now_abs = base_offset_sec + len(buf) / SR         # current abs time based on buffer

                if maybe_gate_window_with_vad(window):
                    # ---- Diarize this window ----
                    diar = diarize_window_in_memory(window)

                    # Convert diar segments to absolute times
                    abs_spans = []
                    for d in diar:
                        s_abs = window_offset + d["start"]
                        e_abs = window_offset + d["end"]
                        abs_spans.append(
                            {"start": s_abs, "end": e_abs, "speaker": d["speaker"]}
                        )

                    # Finalize & ASR only on turns that are safely in the past
                    for span in abs_spans:
                        if span["end"] <= (now_abs - FINALIZE_TAIL_SEC):
                            k = (
                                round_time(span["start"]),
                                round_time(span["end"]),
                                span["speaker"],
                            )
                            if k in emitted_keys:
                                continue

                            s_idx = max(
                                0, int((span["start"] - window_offset) * SR)
                            )
                            e_idx = min(
                                len(window),
                                int((span["end"] - window_offset) * SR),
                            )
                            if e_idx - s_idx <= int(0.25 * SR):
                                continue

                            wave_chunk = window[s_idx:e_idx]
                            text = asr_text_on_chunk(wave_chunk, span["start"])
                            if text:
                                print(f"{span['speaker']}: {text}", flush=True)
                                emitted_keys.add(k)

                # Slide window by HOP; overlap stays in buf
                buf = buf[HOP_SAMPLES:]
                base_offset_sec += HOP_SAMPLES / SR

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("bye.")


if __name__ == "__main__":
    main()
