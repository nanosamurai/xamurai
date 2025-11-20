import os
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import torch
import soundfile as sf
from pyannote.audio import Pipeline, Model
from pyannote.audio import Inference as EmbeddingInference
from faster_whisper import WhisperModel

from bff.settings import settings
import stream_pb2  # only used by .to_asr_events()

# ============== Public result DTO used by app.py ==============

@dataclass
class AsrResult:
    start_s: float
    end_s: float
    text: str
    is_final: bool

# ============== Realtime diar-first engine =====================

class RealtimeEngine:
    """
    Diarization-first rolling pipeline:
      - Maintain a rolling 16 kHz mono buffer
      - For each WINDOW/HOP:
          * optional VAD gate
          * pyannote/speaker-diarization-community-1 on in-memory waveform
          * convert to absolute times
          * finalize spans (ended >= FINALIZE_TAIL_SEC before 'now')
          * center-ownership to dedupe across overlaps
          * optional speaker enrollment mapping
          * ASR (faster-whisper) with vad_filter=False
          * Emit AsrResult(is_final=True)
    Notes:
      - feed(pcm16_bytes) may be invoked with any frame size; we append and process.
      - All heavy models are created once in __init__.
    """

    # --- Tunables (you can externalize to settings if you like) ---
    WINDOW_SEC = 5.0
    OVERLAP_SEC = 0.5
    HOP_SEC = WINDOW_SEC - OVERLAP_SEC

    FINALIZE_TAIL_SEC = 0.30
    FINALIZE_MIN_DUR_SEC = 0.25  # ignore tiny diar segments

    # Speaker enrollment
    USE_SPEAKER_ENROLLMENT = True
    ENROLL_DIR = os.getenv("ENROLL_DIR", "enrolled_speakers")
    ENROLL_SIM_THRESHOLD = 0.35  # a bit lower for mixed/call audio

    # VAD (coarse gate per window) — lightweight, optional
    USE_SILERO_VAD = True
    MIN_SPEECH_IN_WINDOW_SEC = 0.35
    VAD_MIN_SPEECH_MS = 350
    VAD_MIN_SILENCE_MS = 250

    # Dedupe granularity (for emitted span key)
    KEY_RES = 0.25  # seconds

    # Faster-Whisper options
    FW_MODEL_NAME = os.getenv("FW_MODEL_NAME", "medium")  # or "large-v3"
    FW_LANG = os.getenv("FW_LANG", "")  # e.g., "cs" or "" for autodetect

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FW_COMPUTE = "float16" if torch.cuda.is_available() else "int8_float32"

    def __init__(self):
        self.sr = settings.sample_rate  # expect 16000
        assert self.sr == 16000, "Engine currently assumes 16 kHz mono input."

        # Rolling buffer & time bookkeeping
        self._buf = np.zeros(0, dtype=np.float32)
        self._base_offset_sec = 0.0  # absolute start time of _buf's left edge
        self._window_index = 0

        # Dedupe set across overlapping windows
        self._emitted_keys: Set[Tuple[float, float, str]] = set()

        # Concurrency
        self._lock = threading.Lock()

        # ----- Models -----
        # pyannote diar
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for pyannote Community-1 pipeline.")
        self._diar = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
        self._diar.to(torch.device(self.DEVICE))

        # Speaker embedding (for enrollment mapping)
        self._embedding_infer: Optional[EmbeddingInference] = None
        self._enrolled: Dict[str, np.ndarray] = {}
        if self.USE_SPEAKER_ENROLLMENT:
            try:
                emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
                self._embedding_infer = EmbeddingInference(emb_model, window="whole")
            except Exception as e:
                print(f"⚠️  Failed to init embedding model, disabling enrollment: {e}")
                self.USE_SPEAKER_ENROLLMENT = False

        if self.USE_SPEAKER_ENROLLMENT:
            self._load_enrollment()

        # Faster-Whisper ASR
        self._asr = WhisperModel(self.FW_MODEL_NAME, device=self.DEVICE, compute_type=self.FW_COMPUTE)

        # Optional Silero VAD
        self._vad_model = None
        self._get_speech_timestamps = None
        if self.USE_SILERO_VAD:
            try:
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
                self._vad_model = _vad_model
                self._get_speech_timestamps = get_speech_timestamps
            except Exception as e:
                print(f"⚠️  Failed to init Silero VAD, proceeding without: {e}")
                self.USE_SILERO_VAD = False

        # Precompute samples
        self.WINDOW_SAMPLES = int(self.WINDOW_SEC * self.sr)
        self.HOP_SAMPLES = int(self.HOP_SEC * self.sr)

    # ----------------- Public API -----------------

    def feed(self, session_id: str, pcm16: bytes) -> List[AsrResult]:
        """
        Append PCM16 mono audio to the rolling buffer, run as many windows as available,
        and return any finalized ASR results.
        """
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        out: List[AsrResult] = []

        with self._lock:
            # Append to rolling buffer
            self._buf = np.concatenate([self._buf, x])

            # Process complete windows (advancing by HOP)
            while self._buf.size >= self.WINDOW_SAMPLES:
                window = self._buf[:self.WINDOW_SAMPLES]
                window_offset = self._base_offset_sec
                now_abs = self._base_offset_sec + len(self._buf) / self.sr
                my_idx = self._window_index

                if self._maybe_gate_window_with_vad(window):
                    # Diarize this window in-memory
                    diar_segs = self._diarize_window(window)

                    # For each diar segment, finalize only if ended far enough in the past
                    for d in diar_segs:
                        seg_start = d["start"]
                        seg_end = d["end"]
                        dur = seg_end - seg_start
                        if dur < self.FINALIZE_MIN_DUR_SEC:
                            continue

                        s_abs = window_offset + seg_start
                        e_abs = window_offset + seg_end

                        if e_abs > (now_abs - self.FINALIZE_TAIL_SEC):
                            # not finalized yet
                            continue

                        # Ownership: segment center belongs to exactly one window
                        center_abs = 0.5 * (s_abs + e_abs)
                        owner_idx = int(center_abs / self.HOP_SEC + 1e-6)
                        if owner_idx != my_idx:
                            continue

                        # Dedup key
                        key = (self._round_time(s_abs, self.KEY_RES),
                               self._round_time(e_abs, self.KEY_RES),
                               d["speaker"])
                        if key in self._emitted_keys:
                            continue

                        # Cut audio for the segment from 'window'
                        s_idx = int(seg_start * self.sr)
                        e_idx = int(seg_end * self.sr)
                        s_idx = max(0, min(len(window), s_idx))
                        e_idx = max(0, min(len(window), e_idx))
                        if e_idx - s_idx < int(self.FINALIZE_MIN_DUR_SEC * self.sr):
                            continue

                        wav = window[s_idx:e_idx]
                        label = self._map_speaker_label(d["speaker"], wav)
                        text = self._asr_text_on_chunk(wav)

                        if text:
                            self._emitted_keys.add(key)
                            out.append(AsrResult(
                                start_s=s_abs,
                                end_s=e_abs,
                                text=f"{label}: {text}",
                                is_final=True
                            ))

                # Slide window
                self._buf = self._buf[self.HOP_SAMPLES:]
                self._base_offset_sec += self.HOP_SAMPLES / self.sr
                self._window_index += 1

        return out

    def to_asr_events(self, session_id: str, r: AsrResult) -> stream_pb2.AsrEvent:
        # WS layer already uses this
        return stream_pb2.AsrEvent(
            session_id=session_id,
            start_s=r.start_s,
            end_s=r.end_s,
            text=r.text,
            type=stream_pb2.FINAL if r.is_final else stream_pb2.PARTIAL,
            lang=self.FW_LANG or "",
        )

    # ----------------- Internals -----------------

    def _diarize_window(self, wave: np.ndarray):
        if wave.size == 0:
            return []
        w = torch.from_numpy(wave.copy()).unsqueeze(0)  # (1, T)
        out = self._diar({"waveform": w, "sample_rate": self.sr})
        segs = []
        for turn, spk in out.speaker_diarization:
            segs.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)})
        return segs

    def _asr_text_on_chunk(self, wave: np.ndarray) -> str:
        if wave.size < int(self.FINALIZE_MIN_DUR_SEC * self.sr):
            return ""
        # Write temp WAV (faster_whisper works on path)
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            p = tmp.name
        # soundfile expects float in [-1,1]
        sf.write(p, np.clip(wave, -1.0, 1.0).astype(np.float32), self.sr, subtype="PCM_16")
        try:
            segs, _ = self._asr.transcribe(
                p,
                language=(self.FW_LANG or None),
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
                _os.unlink(p)
            except Exception:
                pass

    def _maybe_gate_window_with_vad(self, window: np.ndarray) -> bool:
        # quick RMS prefilter
        if float(np.sqrt(np.mean(window * window)) + 1e-12) < 3e-5:
            return False
        if not self.USE_SILERO_VAD or self._vad_model is None:
            return True

        wav16 = (np.clip(window, -1.0, 1.0) * 32767).astype(np.int16)
        ts = self._get_speech_timestamps(
            wav16,
            self._vad_model,
            sampling_rate=self.sr,
            return_seconds=True,
            min_speech_duration_ms=self.VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=self.VAD_MIN_SILENCE_MS,
        )
        if not ts:
            return False
        speech_sec = sum(max(0.0, t["end"] - t["start"]) for t in ts)
        return speech_sec >= self.MIN_SPEECH_IN_WINDOW_SEC

    def _round_time(self, t: float, res: float) -> float:
        return round(t / res) * res

    # -------- Enrollment mapping --------

    def _load_enrollment(self):
        if not self._embedding_infer:
            return
        if not os.path.isdir(self.ENROLL_DIR):
            print(f"👤 Enrollment dir '{self.ENROLL_DIR}' not found; skipping.")
            return
        per_name: Dict[str, List[np.ndarray]] = {}
        for fname in os.listdir(self.ENROLL_DIR):
            if not fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                continue
            path = os.path.join(self.ENROLL_DIR, fname)
            name = os.path.splitext(fname)[0].split("_")[0]
            try:
                emb = self._embedding_infer(path)
                emb = np.array(emb, dtype=np.float32)
                if emb.ndim > 1:
                    emb = emb.mean(axis=0)
                emb /= np.linalg.norm(emb) + 1e-12
                per_name.setdefault(name, []).append(emb)
                print(f"👤 Enrolled '{name}' from {fname}")
            except Exception as e:
                print(f"⚠️ Failed enrollment for {fname}: {e}")

        for name, embs in per_name.items():
            if not embs:
                continue
            mean_emb = np.stack(embs, axis=0).mean(axis=0)
            mean_emb /= np.linalg.norm(mean_emb) + 1e-12
            self._enrolled[name] = mean_emb

        if self._enrolled:
            print("👤 Enrollment ready:", ", ".join(sorted(self._enrolled.keys())))
        else:
            print("👤 No valid enrollment samples found.")

    def _embed_wave(self, wave: np.ndarray) -> Optional[np.ndarray]:
        if self._embedding_infer is None:
            return None
        if wave.size < int(0.4 * self.sr):
            return None
        rms = float(np.sqrt(np.mean(wave**2)) + 1e-12)
        if rms < 1e-4:
            return None
        x = np.clip(wave / (rms * 3.0), -1.0, 1.0).astype(np.float32)
        try:
            audio = {"waveform": torch.from_numpy(x).unsqueeze(0), "sample_rate": self.sr}
            emb = self._embedding_infer(audio)
            emb = np.array(emb, dtype=np.float32)
            if emb.ndim > 1:
                emb = emb.mean(axis=0)
            emb /= np.linalg.norm(emb) + 1e-12
            return emb
        except Exception as e:
            print(f"⚠️ Failed to embed segment: {e}")
            return None

    def _map_speaker_label(self, diar_label: str, wave_chunk: np.ndarray) -> str:
        if not self.USE_SPEAKER_ENROLLMENT or not self._enrolled:
            return diar_label
        emb = self._embed_wave(wave_chunk)
        if emb is None:
            return diar_label
        best_name, best_sim = None, -1.0
        for name, ref in self._enrolled.items():
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_name is not None and best_sim >= self.ENROLL_SIM_THRESHOLD:
            return best_name
        return diar_label
