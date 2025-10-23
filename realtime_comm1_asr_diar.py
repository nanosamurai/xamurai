#!/usr/bin/env python3
import argparse
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import warnings

# ASR: faster-whisper (CTranslate2)
from faster_whisper import WhisperModel
# Diarization: pyannote v4 Community-1
from pyannote.audio import Pipeline

SAMPLE_RATE = 16000
BLOCK_SIZE = 2048  # ~64 ms @16k
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

@dataclass
class BufferState:
    buf: np.ndarray

def to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16)

def load_vad():
    # Silero-VAD via torch.hub (kept simple)
    vad_model, vad_utils = torch.hub.load(
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
     collect_chunks) = vad_utils
    return vad_model, get_speech_timestamps

def align_speakers_to_asr(asr_segments: List[Dict[str, Any]],
                          diar_segments: List[Dict[str, Any]]) -> List[str]:
    labels = []
    for s in asr_segments:
        s0, s1 = s["start"], s["end"]
        best, ovmax = "SPEAKER_00", 0.0
        for d in diar_segments:
            ov = max(0.0, min(s1, d["end"]) - max(s0, d["start"]))
            if ov > ovmax:
                ovmax = ov
                best = d["speaker"]
        labels.append(best)
    return labels

def run(args):
    prev_text_ctx = ""  # rolling context of last N chars (for initial_prompt)
    CTX_MAX_CHARS = 200
    OVERLAP_SEC = 0.5  # prepend this much audio from previous chunk for context
    PAD_SEC = 0.3  # pad left/right around detected speech
    prev_tail = np.zeros(0, dtype=np.float32)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"Device for diarization: {device}")

    # Load Community-1 (pyannote v4) — uses token= (not use_auth_token=)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set.")
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )
    pipe.to(torch.device(device))

    # Load faster-whisper (GPU if available)
    print(f"Loading faster-whisper ({args.whisper_size}, {args.compute_type})…")
    asr = WhisperModel(args.whisper_size,
                       device=("cuda" if not args.cpu_asr and torch.cuda.is_available() else "cpu"),
                       compute_type=args.compute_type,
                       cpu_threads=args.cpu_threads)

    # VAD
    print("Loading Silero VAD…")
    vad_model, get_speech_timestamps = load_vad()

    # Audio stream
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    state = BufferState(buf=np.zeros(0, dtype=np.float32))

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_q.put(indata.copy())

    sd_kwargs = dict(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    )
    if args.device_index is not None:
        sd_kwargs["device"] = args.device_index

    def record_thread():
        with sd.InputStream(**sd_kwargs):
            dev = sd.default.device
            print(f"🎙️  Listening (device={dev})… Ctrl+C to stop.")
            while True:
                time.sleep(0.1)

    def diarize_waveform_dict(wave_f32: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        Run Community-1 on in-memory waveform to avoid torchcodec/ffmpeg.
        Returns list of {start,end,speaker}.
        """
        import torch as th
        w = th.from_numpy(wave_f32.copy()).view(-1, 1)  # (T,1)
        w = w.transpose(0, 1).contiguous()  # (1, T)
        out = pipe({"waveform": w, "sample_rate": sr})

        # Build flat diar segments
        diar_segments = []
        # out.speaker_diarization is an iterable of (Segment, speaker)
        for turn, spk in out.speaker_diarization:
            diar_segments.append({"start": float(turn.start),
                                  "end": float(turn.end),
                                  "speaker": str(spk)})
        return diar_segments

    def cut_chunks_from_buffer(buf_f32: np.ndarray) -> (list, np.ndarray):
        """
        Return list of (start_idx, end_idx) to emit now, and leftover buffer.
        We merge close VAD segments and add padding.
        """
        wav16 = to_int16(buf_f32)
        stamps = get_speech_timestamps(
            wav16, vad_model, sampling_rate=SAMPLE_RATE, return_seconds=True,
            min_speech_duration_ms=args.vad_min_chunk_ms,  # Consider reducing this
            min_silence_duration_ms=args.vad_max_silence_ms  # Consider reducing this
        )
        if not stamps:
            # trim runaway buffer
            if len(buf_f32) > SAMPLE_RATE * 30:
                buf_f32 = buf_f32[-SAMPLE_RATE * 30:]
            return [], buf_f32

        print(f"DEBUG - VAD Stamps: {stamps}")

        # Merge stamps that are close (< 400 ms gap) into windows
        merged = []
        GAP = 0.4
        cur = [stamps[0]["start"], stamps[0]["end"]]
        for s in stamps[1:]:
            if s["start"] - cur[1] <= GAP:
                cur[1] = max(cur[1], s["end"])
            else:
                merged.append(tuple(cur))
                cur = [s["start"], s["end"]]
        merged.append(tuple(cur))

        print(f"DEBUG - Merged Windows: {merged}")

        # Emit all windows that are complete
        emit = []
        for window in merged:
            if (time.time() - start_time) > 1.0 and (window[1] - window[0]) >= 0.6:  # Relaxed criteria
                emit.append(window)

        print(f"DEBUG - Windows to Emit: {emit}")

        # Convert to samples and add padding
        out = []
        for (s, e) in emit:
            s_idx = max(0, int((s - PAD_SEC) * SAMPLE_RATE))
            e_idx = min(len(buf_f32), int((e + PAD_SEC) * SAMPLE_RATE))
            if e_idx - s_idx >= int(0.4 * SAMPLE_RATE):  # Relaxed from 0.6 to 0.4
                out.append((s_idx, e_idx))

        print(f"DEBUG - Final Emitting Chunks: {out}")

        # Keep leftover buffer for the next round of processing
        if emit:
            keep_from = max(0, int((emit[-1][1] + PAD_SEC) * SAMPLE_RATE))
            leftover = buf_f32[keep_from:]
        else:
            leftover = buf_f32

        return out, leftover

    start_time = time.time()

    def process_chunk(wave_f32: np.ndarray):
        nonlocal prev_tail, prev_text_ctx

        # Gate silence & ensure length
        if wave_f32.size < int(0.6 * SAMPLE_RATE):
            return
        # Prepend overlap tail for acoustic context (don’t print overlapped text later)
        prepend = prev_tail
        combo = np.concatenate([prepend, wave_f32]) if prepend.size else wave_f32

        # Save temp wav for ASR
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, combo, SAMPLE_RATE, subtype="PCM_16")

        # Decode with beam + temperature fallback & context
        segments, info = asr.transcribe(
            tmp_path,
            language=args.language,  # None = autodetect; set "cs"/"en"/...
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4],
            patience=1.0,
            condition_on_previous_text=True,
            initial_prompt=prev_text_ctx[-CTX_MAX_CHARS:] if prev_text_ctx else None,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            vad_filter=False
        )

        # Convert and drop anything that falls entirely in the prepended overlap
        overlap_sec = prepend.size / SAMPLE_RATE
        asr_segments = []
        for s in segments:
            start = float(s.start)
            end = float(s.end)
            text = s.text.strip()
            if not text:
                continue
            # Keep only the part after the overlap
            if end <= overlap_sec:
                continue
            if start < overlap_sec:
                start = overlap_sec
            asr_segments.append({"start": start - overlap_sec,
                                 "end": end - overlap_sec,
                                 "text": text})

        # Update rolling context
        for seg in asr_segments:
            prev_text_ctx = (prev_text_ctx + " " + seg["text"]).strip()
            if len(prev_text_ctx) > 1000:
                prev_text_ctx = prev_text_ctx[-1000:]

        if not asr_segments:
            os.unlink(tmp_path)
            # Update tail for next chunk
            prev_tail = wave_f32[-int(OVERLAP_SEC * SAMPLE_RATE):]
            return

        # Diarize in-memory (unchanged)
        diar_segments = diarize_waveform_dict(wave_f32, SAMPLE_RATE)

        # Align & print
        spk_labels = align_speakers_to_asr(asr_segments, diar_segments)
        for lab, seg in zip(spk_labels, asr_segments):
            print(f"{lab}: {seg['text']}", flush=True)

        # Update tail for next chunk
        prev_tail = wave_f32[-int(OVERLAP_SEC * SAMPLE_RATE):]

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    def worker_thread():
        nonlocal state, prev_tail
        max_buffer_sec = 60
        while True:
            block = audio_q.get()
            if block is None:
                break

            # Debug print to see buffer size before adding new block
            print(f"Buffer length before concat: {len(state.buf)}")

            state.buf = np.concatenate([state.buf, block.reshape(-1)])

            # Debug print to see buffer size after adding new block
            print(f"Buffer length after concat: {len(state.buf)}")

            # Cut stable chunks
            spans, leftover = cut_chunks_from_buffer(state.buf)
            print(f"Emitting {len(spans)} chunks")
            for (s_idx, e_idx) in spans:
                chunk = state.buf[s_idx:e_idx]
                process_chunk(chunk)

            state.buf = leftover
            # keep buffer bounded
            if len(state.buf) > SAMPLE_RATE * max_buffer_sec:
                state.buf = state.buf[-SAMPLE_RATE * max_buffer_sec:]

            # Debug print to see remaining buffer size
            print(f"Remaining buffer length: {len(state.buf)}")

    t_rec = threading.Thread(target=record_thread, daemon=True)
    t_proc = threading.Thread(target=worker_thread, daemon=True)
    t_rec.start()
    t_proc.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping…")
        audio_q.put(None)
        t_proc.join(timeout=2)

def main():
    p = argparse.ArgumentParser("Real-time ASR (faster-whisper) + Community-1 diarization (pyannote v4)")
    p.add_argument("--cpu", action="store_true", help="Force CPU for diarization.")
    p.add_argument("--cpu-asr", action="store_true", help="Force CPU for faster-whisper too.")
    p.add_argument("--cpu-threads", type=int, default=8, help="CPU threads for faster-whisper (when --cpu-asr).")
    p.add_argument("--device-index", type=int, default=34, help="Input device index for sounddevice.")
    p.add_argument("--whisper-size", type=str, default="medium",
                   choices=["tiny", "base", "small", "medium", "large-v3"])
    p.add_argument("--compute-type", type=str, default="float16",
                   help="faster-whisper compute type (float16, int8_float16, int8, etc.)")
    p.add_argument("--vad-min-chunk-ms", type=int, default=1200)
    p.add_argument("--vad-max-silence-ms", type=int, default=600)
    p.add_argument("--language", default="cs", help="Force language code like 'en', 'cs', 'de' …")
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()
