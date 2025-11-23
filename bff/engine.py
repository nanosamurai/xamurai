# bff/engine.py

import os
import threading
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import torch
import soundfile as sf
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Model
from pyannote.audio import Inference as EmbeddingInference

from bff.settings import settings
import stream_pb2

logger = logging.getLogger(__name__)

SR = 16000
BLOCK = 2048                  # processing block (samples @16k)
WINDOW_SEC = 5.0              # diarization/ASR window size
OVERLAP_SEC = 0.5             # overlap between windows
HOP_SEC = WINDOW_SEC - OVERLAP_SEC

# default language; can be overridden per-session
DEFAULT_LANG = os.getenv("FW_LANG_DEFAULT", None)   # "cs", "en", or None

FINALIZE_MIN_DUR_SEC = 0.25   # ignore tiny segments

# VAD (window-level)
USE_SILERO_VAD = False
VAD_MIN_SPEECH_MS = 150      # was 350
VAD_MIN_SILENCE_MS = 400     # was 250
# MIN_SPEECH_IN_WINDOW_SEC = 0.35

# Speaker enrollment
USE_SPEAKER_ENROLLMENT = True
ENROLL_DIR = os.getenv("ENROLL_DIR", "enrolled_speakers")
ENROLL_SIM_THRESHOLD = 0.30

# Dedupe key resolution
KEY_RES = 0.25  # seconds (250 ms)


@dataclass
class AsrResult:
    start_s: float
    end_s: float
    text: str
    is_final: bool
    lang: Optional[str] = None  # NEW


class RealtimeEngine:
    """
    Realtime diarization+ASR engine, ported directly from 20_stream_diar_first.py:

    - Input: 16kHz mono PCM16 bytes (BLOCK=2048) from BFF WS.
    - Internal:
        * rolling buffer `buf` of float32 samples
        * WINDOWS of length 5s, hop 4.5s
        * per-window:
            - optional VAD gate
            - pyannote/speaker-diarization-community-1
            - center-based window "ownership" to avoid double ASR across overlaps
            - coarse dedupe key (round start/end, label)
            - optional speaker enrollment mapping
            - faster-whisper ASR (vad_filter=False, language=per-session)
    - Output: list[AsrResult] for each feed() call.
    """

    def __init__(self) -> None:
        if settings.sample_rate != SR:
            raise RuntimeError(f"Engine assumes {SR} Hz, got {settings.sample_rate}")

        self.sr = SR
        self.default_lang = DEFAULT_LANG

        # Rolling analysis buffer (mixed mono stream)
        self._buf = np.zeros(0, dtype=np.float32)

        # Time of LEFT edge of `buf` in seconds
        self._base_offset_sec = 0.0

        # Window index for center-ownership
        self._window_index = 0

        # Dedup across overlapping windows: coarse (start,end,label)
        self._emitted_keys: Set[Tuple[float, float, str]] = set()

        # VAD globals
        self._vad_model = None
        self._get_speech_timestamps = None

        # Enrollment globals (shared within this engine)
        self._embedding_infer: Optional[EmbeddingInference] = None
        self._enrolled_speakers: Dict[str, np.ndarray] = {}

        # Thread safety for feed()
        self._lock = threading.Lock()

        # ---- HF token ----
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("Set HF_TOKEN environment variable for pyannote.")

        # ---- Models (same as script) ----
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Initializing Faster-Whisper 'medium' on %s", device)
        self._asr = WhisperModel(
            "medium",
            device=device,
            compute_type="float16" if torch.cuda.is_available() else "int8_float32",
        )

        logger.info("Initializing pyannote Community-1 diarization on %s", device)
        self._pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
        self._pipe.to(torch.device(device))

        if USE_SILERO_VAD:
            try:
                logger.info("Initializing Silero VAD…")
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
                logger.warning("Failed to init Silero VAD, disabling: %s", e)

        if USE_SPEAKER_ENROLLMENT:
            try:
                logger.info("Initializing pyannote/embedding for enrollment on %s", device)
                emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
                self._embedding_infer = EmbeddingInference(emb_model, window="whole")
            except Exception as e:
                logger.warning("Failed to load embedding model: %s", e)
                self._embedding_infer = None

        if USE_SPEAKER_ENROLLMENT and self._embedding_infer is not None:
            self._load_enrolled_speakers()
        else:
            logger.info("Speaker enrollment disabled or unavailable.")

        # Precompute sample counts
        self.WINDOW_SAMPLES = int(WINDOW_SEC * SR)
        self.HOP_SAMPLES = int(HOP_SEC * SR)

        logger.info(
            "RealtimeEngine ready: SR=%d WINDOW=%.2fs HOP=%.2fs USE_VAD=%s ENROLL=%s default_lang=%s",
            SR,
            WINDOW_SEC,
            HOP_SEC,
            USE_SILERO_VAD,
            USE_SPEAKER_ENROLLMENT,
            self.default_lang,
        )

    # ===================== PUBLIC API =====================

    def feed(self, session_id: str, pcm16: bytes, lang: Optional[str] = None) -> List[AsrResult]:
        """
        Append PCM16 mono audio to the rolling buffer, run as many 5s windows
        as available, and return finalized ASR results.

        lang: optional ISO code ("cs", "en", etc.). If None, uses self.default_lang.
        """
        effective_lang = lang or self.default_lang

        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        out: List[AsrResult] = []

        with self._lock:
            # 1) Ingest audio into buf
            self._buf = np.concatenate([self._buf, x])
            logger.debug(
                "feed(session=%s, lang=%s): appended %d samples, buf_len=%d",
                session_id, effective_lang, x.size, self._buf.size
            )

            # 2) Process complete windows
            while len(self._buf) >= self.WINDOW_SAMPLES:
                window = self._buf[:self.WINDOW_SAMPLES]
                window_offset = self._base_offset_sec
                my_idx = self._window_index

                if self._maybe_gate_window_with_vad(window):
                    diar_segs = self._diarize_window_in_memory(window)

                    logger.debug(
                        "Window idx=%d offset=[%.3f, %.3f] diar_segs=%d",
                        my_idx,
                        window_offset,
                        window_offset + WINDOW_SEC,
                        len(diar_segs),
                    )

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
                            self._round_time(s_abs, KEY_RES),
                            self._round_time(e_abs, KEY_RES),
                            d["speaker"],
                        )
                        if key in self._emitted_keys:
                            continue

                        # Cut from this window
                        s_idx = int(seg_start * SR)
                        e_idx = int(seg_end * SR)
                        s_idx = max(0, min(len(window), s_idx))
                        e_idx = max(0, min(len(window), e_idx))
                        if e_idx - s_idx < int(FINALIZE_MIN_DUR_SEC * SR):
                            continue

                        wave_chunk = window[s_idx:e_idx]
                        label = self._map_speaker_label(d["speaker"], wave_chunk)
                        text = self._asr_text_on_chunk(wave_chunk, effective_lang)

                        if text:
                            self._emitted_keys.add(key)
                            logger.debug(
                                "Emitting session=%s speaker=%s [%.3f, %.3f] lang=%s: %s",
                                session_id, label, s_abs, e_abs, effective_lang, text
                            )
                            out.append(AsrResult(
                                start_s=s_abs,
                                end_s=e_abs,
                                text=f"{label}: {text}",
                                is_final=True,
                                lang=effective_lang,
                            ))

                # Slide window
                self._buf = self._buf[self.HOP_SAMPLES:]
                self._base_offset_sec += self.HOP_SAMPLES / SR
                self._window_index += 1

        return out

    def to_asr_events(self, session_id: str, r: AsrResult) -> stream_pb2.AsrEvent:
        return stream_pb2.AsrEvent(
            session_id=session_id,
            start_s=r.start_s,
            end_s=r.end_s,
            text=r.text,
            type=stream_pb2.FINAL if r.is_final else stream_pb2.PARTIAL,
            lang=(r.lang or self.default_lang or ""),
        )

    # ===================== INTERNALS (1:1 with script, + lang param) =====================

    def _round_time(self, t: float, resolution: float) -> float:
        return round(t / resolution) * resolution

    def _diarize_window_in_memory(self, wave: np.ndarray):
        if wave.size == 0:
            return []
        w = torch.from_numpy(wave.copy()).unsqueeze(0)
        out = self._pipe({"waveform": w, "sample_rate": SR})
        segs = []
        for turn, spk in out.speaker_diarization:
            segs.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(spk),
            })
        return segs

    def _asr_text_on_chunk(self, wave: np.ndarray, lang: Optional[str]) -> str:
        if wave.size < int(FINALIZE_MIN_DUR_SEC * SR):
            return ""
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        sf.write(path, wave, SR, subtype="PCM_16")
        try:
            segs, _ = self._asr.transcribe(
                path,
                language=(lang or None),
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
                _os.unlink(path)
            except Exception:
                pass

    def _maybe_gate_window_with_vad(self, window: np.ndarray) -> bool:
        # 1) RMS pre-gate: kill *really* quiet stuff
        rms = float(np.sqrt(np.mean(window * window)) + 1e-12)
        if rms < 3e-5:
            return False

        # 2) If Silero is disabled or unavailable → always keep
        if not USE_SILERO_VAD or self._vad_model is None:
            return True

        # 3) Silero: we only care if there is *any* speech at all
        wav16 = (np.clip(window, -1.0, 1.0) * 32767).astype(np.int16)
        ts = self._get_speech_timestamps(
            wav16,
            self._vad_model,
            sampling_rate=SR,
            return_seconds=True,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        )

        if not ts:
            # pure silence: drop window, prevents hallucinations
            return False

        # some speech → keep the window and let diarization+ASR decide
        return True

    # ====== Enrollment ======

    def _load_enrolled_speakers(self) -> None:
        if not USE_SPEAKER_ENROLLMENT or self._embedding_infer is None:
            logger.info("Speaker enrollment disabled or no embedding model.")
            return

        if not os.path.isdir(ENROLL_DIR):
            logger.info("Enrollment dir '%s' not found; skipping.", ENROLL_DIR)
            return

        per_name: Dict[str, List[np.ndarray]] = defaultdict(list)
        for fname in os.listdir(ENROLL_DIR):
            if not fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                continue
            path = os.path.join(ENROLL_DIR, fname)
            base = os.path.splitext(fname)[0]
            name = base.split("_")[0]

            try:
                emb = self._embedding_infer(path)
                emb = np.array(emb, dtype=np.float32)
                if emb.ndim > 1:
                    emb = emb.mean(axis=0)
                per_name[name].append(emb)
                logger.info("Enrolled segment for '%s' from %s", name, fname)
            except Exception as e:
                logger.warning("Failed to enroll from %s: %s", fname, e)

        for name, embs in per_name.items():
            if not embs:
                continue
            arr = np.stack(embs, axis=0)
            mean_emb = arr.mean(axis=0)
            mean_emb /= np.linalg.norm(mean_emb) + 1e-12
            self._enrolled_speakers[name] = mean_emb

        if self._enrolled_speakers:
            logger.info(
                "Enrollment complete. Known speakers: %s",
                ", ".join(self._enrolled_speakers.keys()),
            )
        else:
            logger.info("No valid enrollment samples found.")

    def _embed_wave(self, wave: np.ndarray) -> Optional[np.ndarray]:
        if self._embedding_infer is None or wave.size < int(0.25 * SR):
            return None
        try:
            audio = {"waveform": torch.from_numpy(wave).unsqueeze(0), "sample_rate": SR}
            emb = self._embedding_infer(audio)
            emb = np.array(emb, dtype=np.float32)
            if emb.ndim > 1:
                emb = emb.mean(axis=0)
            emb /= np.linalg.norm(emb) + 1e-12
            return emb
        except Exception as e:
            logger.warning("Failed to embed segment: %s", e)
            return None

    def _map_speaker_label(self, diar_label: str, wave_chunk: np.ndarray) -> str:
        if not USE_SPEAKER_ENROLLMENT or not self._enrolled_speakers:
            return diar_label

        emb = self._embed_wave(wave_chunk)
        if emb is None:
            return diar_label

        best_name = None
        best_sim = -1.0
        for name, ref in self._enrolled_speakers.items():
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name is not None and best_sim >= ENROLL_SIM_THRESHOLD:
            logger.debug(
                "speaker %s → %s (sim=%.3f)", diar_label, best_name, best_sim
            )
            return best_name
        return diar_label
