import io
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

from proto_gen import stream_pb2

# -------------------- PyTorch checkpoint loading compatibility --------------------
#
# Recent PyTorch versions (2.6+) introduced safer defaults for torch.load() that can
# break loading some third-party checkpoints (notably pyannote/lightning models)
# due to `weights_only=True` restrictions.
#
# In our stack we trust upstream model checkpoints (HF/pyannote), so we:
# 1) allowlist OmegaConf classes for weights_only=True code paths
# 2) monkeypatch torch.load to default to weights_only=False (more robust)
#
# NOTE: weights_only=False can load arbitrary pickled code. Do not use this with
# untrusted checkpoints.

try:
    import torch

    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata

        try:
            from omegaconf.nodes import (
                AnyNode,
                BooleanNode,
                BytesNode,
                EnumNode,
                FloatNode,
                IntegerNode,
                PathNode,
                StringNode,
            )
        except Exception:
            AnyNode = BooleanNode = BytesNode = EnumNode = FloatNode = IntegerNode = PathNode = StringNode = None

        safe = [
            DictConfig,
            ListConfig,
            ContainerMetadata,
            AnyNode,
            BooleanNode,
            BytesNode,
            EnumNode,
            FloatNode,
            IntegerNode,
            PathNode,
            StringNode,
        ]
        torch.serialization.add_safe_globals([c for c in safe if c is not None])
    except Exception:
        pass

    try:
        if not hasattr(torch, "_drsynth_orig_load"):
            torch._drsynth_orig_load = torch.load  # type: ignore[attr-defined]

        if getattr(torch.load, "__name__", "") != "_torch_load_weights_only_false_default":
            _orig_torch_load = torch._drsynth_orig_load  # type: ignore[attr-defined]

            def _torch_load_weights_only_false_default(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)

            torch.load = _torch_load_weights_only_false_default
    except Exception:
        pass
except Exception:
    # torch not installed in lightweight unit test envs
    pass

logger = logging.getLogger(__name__)


def _bool_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

SR = 16000

# Windowing defaults are configurable via env for deployment tuning.
WINDOW_SEC = float(os.getenv("RT_WINDOW_SEC", "5.0"))
OVERLAP_SEC = float(os.getenv("RT_OVERLAP_SEC", "0.5"))

# Cumulative refinement: emit partial hypotheses on shorter cadence.
PARTIAL_ENABLE = _bool_env("RT_PARTIAL_ENABLE") if os.getenv("RT_PARTIAL_ENABLE") is not None else True
EMIT_EVERY_SEC = float(os.getenv("RT_EMIT_EVERY_SEC", "0.7"))

# Partial stability: require the same hypothesis to appear N times before emitting.
# Set to 1 to disable.
PARTIAL_STABILITY_REPEATS = int(os.getenv("RT_PARTIAL_STABILITY_REPEATS", "1"))

# Minimum audio duration before we emit any PARTIAL. This avoids very-early
# hallucinated partials.
PARTIAL_MIN_BUFFER_SEC = float(os.getenv("RT_PARTIAL_MIN_BUFFER_SEC", "1.5"))

# Kept for reference (most code should use RealtimeConfig.hop_sec)
HOP_SEC = WINDOW_SEC - OVERLAP_SEC

DEFAULT_LANG = os.getenv("FW_LANG_DEFAULT", None)  # per-session override still possible

FINALIZE_MIN_DUR_SEC = 0.25
KEY_RES = 0.25

# VAD (window-level)
USE_SILERO_VAD = True
VAD_MIN_SPEECH_MS = 150
VAD_MIN_SILENCE_MS = 400

# Speaker enrollment
USE_SPEAKER_ENROLLMENT = True
ENROLL_SIM_THRESHOLD = float(os.getenv("ENROLL_SIM_THRESHOLD", "0.30"))

# Session eviction (rtservice only)
RT_SESSION_IDLE_SECONDS = float(os.getenv("RT_SESSION_IDLE_SECONDS", "120.0"))
RT_SESSION_EVICT_SCAN_SECONDS = float(os.getenv("RT_SESSION_EVICT_SCAN_SECONDS", "10.0"))


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    n_out = int(round(float(x.size) * float(sr_out) / float(sr_in)))
    if n_out <= 1:
        return np.zeros((0,), dtype=np.float32)
    t_old = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    y = np.interp(t_new, t_old, x).astype(np.float32)
    return y


@dataclass
class AsrResult:
    start_s: float
    end_s: float
    text: str
    is_final: bool
    lang: Optional[str] = None
    speaker: Optional[str] = None


@dataclass
class RealtimeConfig:
    """Runtime configuration for windowing and heuristics.

    NOTE: This is separated so unit tests can instantiate a lightweight engine
    without pulling in torch/pyannote.
    """

    sr: int = SR
    window_sec: float = WINDOW_SEC
    overlap_sec: float = OVERLAP_SEC
    partial_enable: bool = PARTIAL_ENABLE
    emit_every_sec: float = EMIT_EVERY_SEC
    partial_stability_repeats: int = PARTIAL_STABILITY_REPEATS
    partial_min_buffer_sec: float = PARTIAL_MIN_BUFFER_SEC
    finalize_min_dur_sec: float = FINALIZE_MIN_DUR_SEC
    key_resolution_sec: float = KEY_RES

    @property
    def hop_sec(self) -> float:
        return float(self.window_sec - self.overlap_sec)

    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.sr))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_sec * self.sr))

    @property
    def emit_every_samples(self) -> int:
        # Avoid pathological configs.
        sec = max(0.05, float(self.emit_every_sec))
        return int(round(sec * self.sr))

    @property
    def partial_min_buffer_samples(self) -> int:
        sec = max(0.0, float(self.partial_min_buffer_sec))
        return int(round(sec * self.sr))


@dataclass
class SessionState:
    """Per-(tenant,session) realtime rolling window state."""

    buf: np.ndarray
    base_offset_sec: float
    window_index: int
    emitted_keys: Set[Tuple[float, float, str]]
    last_activity_s: float
    lock: threading.Lock
    # Cumulative refinement tracking (monotonic per session).
    total_samples_ingested: int
    last_partial_emit_at_samples: int
    last_partial_text: str
    pending_partial_text: str
    pending_partial_repeats: int

    # Signature of the realtime config used for this session. If a client changes
    # per-session overrides mid-stream, we reset the session state.
    cfg_key: Optional[Tuple[object, ...]]

    @staticmethod
    def new(now_s: float) -> "SessionState":
        return SessionState(
            buf=np.zeros(0, dtype=np.float32),
            base_offset_sec=0.0,
            window_index=0,
            emitted_keys=set(),
            last_activity_s=now_s,
            lock=threading.Lock(),
            total_samples_ingested=0,
            last_partial_emit_at_samples=0,
            last_partial_text="",
            pending_partial_text="",
            pending_partial_repeats=0,
            cfg_key=None,
        )


# ---------------------------------------------------------------------------
# Session processor (injectable for unit tests)
# ---------------------------------------------------------------------------

DiarSeg = Dict[str, object]  # {start: float, end: float, speaker: str}

# Gate functions for realtime processing.
# - partial_gate_fn: typically RMS + VAD (avoid hallucinated partials)
# - window_gate_fn: typically RMS-only (avoid dropping content due to VAD misses)
GateFn = Callable[[np.ndarray], bool]
DiarizeFn = Callable[[np.ndarray], List[DiarSeg]]
AsrFn = Callable[[np.ndarray, Optional[str]], str]
MapSpeakerFn = Callable[[str, str, np.ndarray], str]  # (tenant_id, diar_label, wave_chunk) -> label


class RealtimeSessionProcessor:
    def __init__(
        self,
        *,
        cfg: RealtimeConfig,
        partial_gate_fn: GateFn,
        window_gate_fn: Optional[GateFn] = None,
        diarize_fn: DiarizeFn,
        asr_fn: AsrFn,
        map_speaker_fn: MapSpeakerFn,
    ) -> None:
        self._cfg = cfg
        self._partial_gate_fn = partial_gate_fn
        self._window_gate_fn = window_gate_fn or partial_gate_fn
        self._diarize_fn = diarize_fn
        self._asr_fn = asr_fn
        self._map_speaker_fn = map_speaker_fn

    def process(
        self,
        *,
        tenant_id: str,
        session_id: str,
        state: SessionState,
        pcm16: bytes,
        lang: Optional[str],
    ) -> List[AsrResult]:
        cfg = self._cfg
        effective_lang = lang

        is_flush = (pcm16 is None) or (len(pcm16) == 0)

        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        out: List[AsrResult] = []

        # ingest
        state.buf = np.concatenate([state.buf, x])
        state.total_samples_ingested += int(x.size)

        # Emit low-latency partial updates as audio accumulates, before we have a full window.
        # This uses ASR-only (no diarization) in phase 1 to keep it cheaper.
        #
        # IMPORTANT: do not emit partials when the caller is flushing (empty chunk)
        # because that would produce confusing late PARTIALs.
        if cfg.partial_enable and not is_flush:
            out.extend(
                self._maybe_emit_partial(
                    tenant_id=tenant_id,
                    session_id=session_id,
                    state=state,
                    lang=effective_lang,
                )
            )

        # process
        while state.buf.size >= cfg.window_samples:
            window = state.buf[: cfg.window_samples]
            window_offset = state.base_offset_sec
            my_idx = state.window_index

            emitted_any_final_for_window = False

            # We keep two separate gates:
            # - window gate: avoid dropping windows entirely (RMS-only in prod)
            # - partial/speech gate: uses VAD to decide whether diarization is worth running
            if self._window_gate_fn(window):
                speech_ok = True
                try:
                    speech_ok = bool(self._partial_gate_fn(window))
                except Exception:
                    speech_ok = True

                diar_segs = self._diarize_fn(window) if speech_ok else []

                for d in diar_segs:
                    seg_start = float(d.get("start") or 0.0)
                    seg_end = float(d.get("end") or 0.0)
                    diar_label = str(d.get("speaker") or "")

                    dur = seg_end - seg_start
                    if dur < cfg.finalize_min_dur_sec:
                        continue

                    s_abs = window_offset + seg_start
                    e_abs = window_offset + seg_end
                    center_abs = 0.5 * (s_abs + e_abs)

                    owner_idx = int(center_abs / cfg.hop_sec + 1e-6)
                    if owner_idx != my_idx:
                        continue

                    key = (
                        _round_time(s_abs, cfg.key_resolution_sec),
                        _round_time(e_abs, cfg.key_resolution_sec),
                        diar_label,
                    )
                    if key in state.emitted_keys:
                        continue

                    s_idx = int(seg_start * cfg.sr)
                    e_idx = int(seg_end * cfg.sr)
                    s_idx = max(0, min(window.size, s_idx))
                    e_idx = max(0, min(window.size, e_idx))
                    if e_idx - s_idx < int(cfg.finalize_min_dur_sec * cfg.sr):
                        continue

                    wave_chunk = window[s_idx:e_idx]
                    label = self._map_speaker_fn(tenant_id, diar_label, wave_chunk)
                    text = self._asr_fn(wave_chunk, effective_lang)

                    if text:
                        state.emitted_keys.add(key)
                        out.append(
                            AsrResult(
                                start_s=s_abs,
                                end_s=e_abs,
                                text=text,
                                is_final=True,
                                lang=effective_lang,
                                speaker=label,
                            )
                        )
                        emitted_any_final_for_window = True

                # Fallback: if diarization returns no usable FINAL segments (either because
                # diarization yielded nothing, or because all segments were filtered out by
                # duration/ownership/deduping), still emit a single FINAL for the full
                # window. This avoids the confusing behavior where clients see PARTIALs
                # but never get a FINAL for that window.
                if not emitted_any_final_for_window:
                    text = self._asr_fn(window, effective_lang)
                    text = (text or "").strip()
                    if text:
                        out.append(
                            AsrResult(
                                start_s=float(window_offset),
                                end_s=float(window_offset + cfg.window_sec),
                                text=text,
                                is_final=True,
                                lang=effective_lang,
                                speaker=None,
                            )
                        )

            # slide
            state.buf = state.buf[cfg.hop_samples :]
            state.base_offset_sec += cfg.hop_samples / cfg.sr
            state.window_index += 1

            # Partial emission should start fresh per window.
            state.last_partial_text = ""
            state.pending_partial_text = ""
            state.pending_partial_repeats = 0

            # After sliding, we may again be in an "incomplete window" state;
            # allow partial emission for the new window as it accumulates.
            if cfg.partial_enable and not is_flush:
                out.extend(
                    self._maybe_emit_partial(
                        tenant_id=tenant_id,
                        session_id=session_id,
                        state=state,
                        lang=effective_lang,
                    )
                )

        # End-of-stream flush: when the client sends an empty chunk (or we get an empty
        # buffer), finalize the remaining tail (shorter than window_sec) as a FINAL.
        # This ensures we don't drop the last ~overlap chunk of the stream.
        if is_flush and state.buf.size >= int(cfg.finalize_min_dur_sec * cfg.sr):
            try:
                if self._window_gate_fn(state.buf):
                    buf = state.buf
                    # Cap to window_samples in case callers flush while having >1 window.
                    if buf.size > cfg.window_samples:
                        buf = buf[: cfg.window_samples]
                    text = self._asr_fn(buf, effective_lang)
                    text = (text or "").strip()
                    if text:
                        s_abs = float(state.base_offset_sec)
                        e_abs = float(state.base_offset_sec + (buf.size / cfg.sr))
                        out.append(
                            AsrResult(
                                start_s=s_abs,
                                end_s=e_abs,
                                text=text,
                                is_final=True,
                                lang=effective_lang,
                                speaker=None,
                            )
                        )
            except Exception:
                logger.warning("rtservice: tail flush failed", exc_info=True)

            # Prevent emitting the tail repeatedly if multiple flush chunks arrive.
            state.buf = np.zeros(0, dtype=np.float32)
            state.pending_partial_text = ""
            state.pending_partial_repeats = 0

        return out

    def _maybe_emit_partial(
        self,
        *,
        tenant_id: str,
        session_id: str,
        state: SessionState,
        lang: Optional[str],
    ) -> List[AsrResult]:
        cfg = self._cfg

        # Need a bit of audio before we can emit anything meaningful.
        if state.buf.size < int(cfg.finalize_min_dur_sec * cfg.sr):
            return []

        # Avoid very-early hallucinated partials.
        if state.buf.size < cfg.partial_min_buffer_samples:
            return []

        # If we already have a full window available, we'll emit finals via the
        # normal path. Avoid producing redundant PARTIALs here.
        if state.buf.size >= cfg.window_samples:
            return []

        # Rate-limit partials by an ingestion-based monotonic clock.
        # This remains stable even if the rolling buffer slides.
        if (state.total_samples_ingested - state.last_partial_emit_at_samples) < cfg.emit_every_samples:
            return []

        # Only emit partials when there's likely speech.
        # gate_fn uses RMS + optional VAD; it's fine to call it on partial windows.
        try:
            if not self._partial_gate_fn(state.buf):
                state.last_partial_emit_at_samples = state.total_samples_ingested
                return []
        except Exception:
            # Never let partial gating break the stream.
            pass

        # ASR-only partial: transcribe the current incomplete window (cap to window).
        # NOTE: we do not include diarization/speaker mapping in partials for phase 1.
        buf = state.buf
        if buf.size > cfg.window_samples:
            buf = buf[: cfg.window_samples]

        text = self._asr_fn(buf, lang)
        state.last_partial_emit_at_samples = state.total_samples_ingested

        text = (text or "").strip()
        if not text:
            return []

        # De-spam identical repeats.
        if text == state.last_partial_text:
            return []

        # Stability filter: only emit when we see the same hypothesis repeatedly.
        # This helps avoid very-early hallucinations on short buffers.
        repeats = int(cfg.partial_stability_repeats)
        if repeats > 1:
            if text == state.pending_partial_text:
                state.pending_partial_repeats += 1
            else:
                state.pending_partial_text = text
                state.pending_partial_repeats = 1

            if state.pending_partial_repeats < repeats:
                return []

        state.last_partial_text = text

        s_abs = float(state.base_offset_sec)
        e_abs = float(state.base_offset_sec + (buf.size / cfg.sr))

        return [
            AsrResult(
                start_s=s_abs,
                end_s=e_abs,
                text=text,
                is_final=False,
                lang=lang,
                speaker=None,
            )
        ]


def _round_time(t: float, resolution: float) -> float:
    return round(t / resolution) * resolution


# ---------------------------------------------------------------------------
# Production model bundle (heavy deps are imported lazily here)
# ---------------------------------------------------------------------------


class RealtimeModelBundle:
    """Holds heavy ML models + enrollment resolver.

    The goal is:
    - models load once per process
    - per-session state stays separate

    Heavy imports (torch/pyannote/faster_whisper) are intentionally inside
    this class so importing `rtservice.engine` stays lightweight for unit tests.
    """

    def __init__(self, *, cfg: RealtimeConfig) -> None:
        self._cfg = cfg

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("Set HF_TOKEN environment variable for pyannote.")

        import torch
        from faster_whisper import WhisperModel
        from pyannote.audio import Pipeline, Model
        from pyannote.audio import Inference as EmbeddingInference

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not torch.cuda.is_available():
            logger.warning(
                "rtservice: CUDA not available; running on CPU (torch=%s torch.version.cuda=%s)",
                getattr(torch, "__version__", "unknown"),
                getattr(getattr(torch, "version", None), "cuda", None),
            )
        else:
            logger.info(
                "rtservice: CUDA available; will use GPU (torch=%s torch.version.cuda=%s)",
                getattr(torch, "__version__", "unknown"),
                getattr(getattr(torch, "version", None), "cuda", None),
            )

        asr_model = os.getenv("RT_ASR_MODEL", "medium").strip() or "medium"
        logger.info("Initializing Faster-Whisper '%s' on %s", asr_model, device)
        self._asr = WhisperModel(
            asr_model,
            device=device,
            compute_type="float16" if torch.cuda.is_available() else "int8_float32",
        )

        diar_model_id = os.getenv(
            "RT_DIAR_MODEL",
            # pyannote.audio 3.x compatible pipeline id
            "pyannote/speaker-diarization-3.1",
        ).strip()

        def _load_pipe(mid: str):
            logger.info("Initializing diarization pipeline (%s) on %s", mid, device)
            # pyannote.audio 4.x uses `token=` (not `use_auth_token=`)
            pipe = Pipeline.from_pretrained(mid, token=hf_token)
            pipe.to(torch.device(device))
            return pipe

        try:
            self._pipe = _load_pipe(diar_model_id)
        except Exception:
            fallback = "pyannote/speaker-diarization-3.1"
            if diar_model_id != fallback:
                logger.warning(
                    "Failed to init diarization pipeline (%s); trying fallback %s",
                    diar_model_id,
                    fallback,
                    exc_info=True,
                )
                self._pipe = _load_pipe(fallback)
            else:
                raise

        # VAD
        self._vad_model = None
        self._get_speech_timestamps = None
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
                (get_speech_timestamps, _save_audio, _read_audio, _VADIterator, _collect_chunks) = _vad_utils
                self._vad_model = _vad_model
                self._get_speech_timestamps = get_speech_timestamps
            except Exception as e:
                logger.warning("Failed to init Silero VAD, disabling: %s", e)

        # Embedding
        self._embedding_infer = None
        self._embed_lock = threading.Lock()
        if USE_SPEAKER_ENROLLMENT:
            try:
                logger.info("Initializing pyannote/embedding for enrollment on %s", device)
                emb_model = Model.from_pretrained("pyannote/embedding", token=hf_token)
                self._embedding_infer = EmbeddingInference(emb_model, window="whole")
            except Exception as e:
                logger.warning("Failed to load embedding model: %s", e)
                self._embedding_infer = None

        # Enrollment: new per-tenant cache (optional)
        self._enrollment_cache = None
        self._init_enrollment_cache_from_env()

        # Enrollment: legacy flat dir (dev compatibility)
        self._legacy_enrolled_speakers: Dict[str, np.ndarray] = {}
        self._load_legacy_enrolled_speakers_flat_dir()

        # Startup diagnostics (make it obvious what is enabled and how enrollment is configured)
        logger.info(
            "RealtimeModelBundle ready: SR=%d WINDOW=%.2fs HOP=%.2fs USE_VAD=%s ENROLL=%s",
            cfg.sr,
            cfg.window_sec,
            cfg.hop_sec,
            USE_SILERO_VAD,
            USE_SPEAKER_ENROLLMENT,
        )

        try:
            import importlib.metadata as _md

            logger.info(
                "rtservice versions: torch=%s pyannote-audio=%s faster-whisper=%s",
                getattr(torch, "__version__", "unknown"),
                _md.version("pyannote-audio"),
                _md.version("faster-whisper"),
            )
        except Exception:
            logger.debug("rtservice versions: torch=%s", getattr(torch, "__version__", "unknown"))

        logger.info(
            "rtservice enrollment config: backend=%s cache_ttl_s=%s max_tenants=%s sim_threshold=%.3f",
            os.getenv("ENROLL_BACKEND", "legacy_dir"),
            os.getenv("ENROLL_CACHE_TTL_S", "300"),
            os.getenv("ENROLL_CACHE_MAX_TENANTS", "128"),
            ENROLL_SIM_THRESHOLD,
        )

        if os.getenv("ENROLL_BACKEND", "legacy_dir").strip().lower() == "s3_manifest":
            logger.info(
                "rtservice enrollment s3: endpoint=%s bucket=%s prefix=%s force_path_style=%s",
                os.getenv("ENROLL_S3_ENDPOINT", ""),
                os.getenv("ENROLL_S3_BUCKET", ""),
                os.getenv("ENROLL_S3_PREFIX", "enrollment"),
                os.getenv("ENROLL_S3_FORCE_PATH_STYLE", os.getenv("S3_FORCE_PATH_STYLE", "true")),
            )

    # ---------------- VAD / diarization / ASR ----------------

    def gate_window(self, window: np.ndarray) -> bool:
        # RMS pre-gate
        rms = float(np.sqrt(np.mean(window * window)) + 1e-12)
        if rms < 3e-5:
            return False
        return True

    def gate_speech(self, window: np.ndarray) -> bool:
        """Speech gate for PARTIALs / diarization.

        This is intentionally stricter than `gate_window` because we use VAD to
        avoid hallucinated partials and to avoid doing expensive diarization
        work on windows that look like silence.
        """

        if not self.gate_window(window):
            return False

        if not USE_SILERO_VAD or self._vad_model is None:
            return True

        wav16 = (np.clip(window, -1.0, 1.0) * 32767).astype(np.int16)
        ts = self._get_speech_timestamps(
            wav16,
            self._vad_model,
            sampling_rate=self._cfg.sr,
            return_seconds=True,
            min_speech_duration_ms=VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        )
        return bool(ts)

    def diarize_window(self, wave: np.ndarray) -> List[DiarSeg]:
        if wave.size == 0:
            return []
        import torch

        w = torch.from_numpy(wave.copy()).unsqueeze(0)
        out = self._pipe({"waveform": w, "sample_rate": self._cfg.sr})

        # pyannote.audio 4.x returns DiarizeOutput with an Annotation in `.speaker_diarization`
        diar = getattr(out, "speaker_diarization", None)
        segs: List[DiarSeg] = []
        if diar is None:
            return segs

        for turn, _, spk in diar.itertracks(yield_label=True):
            segs.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)})
        return segs

    def asr_text(self, wave: np.ndarray, lang: Optional[str]) -> str:
        if wave.size < int(self._cfg.finalize_min_dur_sec * self._cfg.sr):
            return ""
        import tempfile
        import os as _os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name

        sf.write(path, wave, self._cfg.sr, subtype="PCM_16")
        try:
            segs, _info = self._asr.transcribe(
                path,
                language=(lang or None),
                task="transcribe",
                beam_size=5,
                temperature=[0.0],
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=True,
            )
            words: list[str] = []
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

    # ---------------- Enrollment mapping ----------------

    def _init_enrollment_cache_from_env(self) -> None:
        if not USE_SPEAKER_ENROLLMENT:
            return
        if self._embedding_infer is None:
            return

        backend = os.getenv("ENROLL_BACKEND", "legacy_dir").strip().lower()
        if backend in ("none", "disabled"):
            return
        if backend == "legacy_dir":
            return

        from drsynth_common.enrollment import EnrollmentCache, LocalEnrollmentProvider, S3EnrollmentProvider

        ttl_s = float(os.getenv("ENROLL_CACHE_TTL_S", "300"))
        max_tenants = int(os.getenv("ENROLL_CACHE_MAX_TENANTS", "128"))

        if backend == "local_manifest":
            enroll_dir = os.getenv("ENROLL_DIR", "enrolled_speakers")
            provider = LocalEnrollmentProvider(root_dir=enroll_dir)
        elif backend == "s3_manifest":
            bucket = os.getenv("ENROLL_S3_BUCKET", os.getenv("S3_BUCKET", "")).strip()
            if not bucket:
                raise RuntimeError("ENROLL_BACKEND=s3_manifest but ENROLL_S3_BUCKET is not set")

            prefix = os.getenv("ENROLL_S3_PREFIX", "enrollment").strip()

            provider = S3EnrollmentProvider(
                bucket=bucket,
                prefix=prefix,
                endpoint=os.getenv("ENROLL_S3_ENDPOINT", os.getenv("S3_ENDPOINT", "")).strip(),
                region=os.getenv("ENROLL_S3_REGION", os.getenv("S3_REGION", "")).strip(),
                access_key=os.getenv("ENROLL_S3_ACCESS_KEY", os.getenv("S3_ACCESS_KEY", "")).strip(),
                secret_key=os.getenv("ENROLL_S3_SECRET_KEY", os.getenv("S3_SECRET_KEY", "")).strip(),
                force_path_style=(
                    os.getenv("ENROLL_S3_FORCE_PATH_STYLE", os.getenv("S3_FORCE_PATH_STYLE", "true"))
                    .strip()
                    .lower()
                    in ("1", "true", "yes", "y")
                ),
            )
        else:
            raise RuntimeError(f"Unknown ENROLL_BACKEND={backend}")

        def _embed_bytes(sample_bytes: bytes) -> np.ndarray:
            # decode audio bytes to mono float32
            b = io.BytesIO(sample_bytes)
            x, sr = sf.read(b, dtype="float32")
            if isinstance(x, np.ndarray) and x.ndim > 1:
                x = x.mean(axis=1)
            x = np.asarray(x, dtype=np.float32).reshape(-1)
            if int(sr) != self._cfg.sr:
                x = _resample_linear(x, int(sr), self._cfg.sr)
            emb = self._embed_wave(x)
            if emb is None:
                raise RuntimeError("Failed to embed enrollment sample")
            return emb

        self._enrollment_cache = EnrollmentCache(
            provider=provider,
            embed_fn=_embed_bytes,
            ttl_s=ttl_s,
            max_tenants=max_tenants,
        )
        logger.info("Enrollment cache enabled: backend=%s ttl_s=%.1f max_tenants=%d", backend, ttl_s, max_tenants)

    def _load_legacy_enrolled_speakers_flat_dir(self) -> None:
        if not USE_SPEAKER_ENROLLMENT or self._embedding_infer is None:
            return

        # legacy enrollment is kept for local runs.
        enroll_dir = os.getenv("ENROLL_DIR", "enrolled_speakers")
        if not os.path.isdir(enroll_dir):
            return

        # If the directory contains per-tenant subdirs, we don't do legacy.
        # (tenant-aware should be done via ENROLL_BACKEND=local_manifest)
        try:
            has_tenant_layout = any((os.path.isdir(os.path.join(enroll_dir, d)) for d in os.listdir(enroll_dir)))
            if has_tenant_layout:
                return
        except Exception:
            pass

        per_name: Dict[str, List[np.ndarray]] = defaultdict(list)
        for fname in os.listdir(enroll_dir):
            if not fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                continue
            path = os.path.join(enroll_dir, fname)
            base = os.path.splitext(fname)[0]
            name = base.split("_")[0]
            try:
                emb = self._embedding_infer(path)
                emb = np.array(emb, dtype=np.float32)
                if emb.ndim > 1:
                    emb = emb.mean(axis=0)
                per_name[name].append(emb)
                logger.info("[legacy] Enrolled segment for '%s' from %s", name, fname)
            except Exception:
                logger.warning("[legacy] Failed to enroll from %s", fname, exc_info=True)

        for name, embs in per_name.items():
            if not embs:
                continue
            arr = np.stack(embs, axis=0)
            mean = arr.mean(axis=0)
            mean /= np.linalg.norm(mean) + 1e-12
            self._legacy_enrolled_speakers[name] = mean

        if self._legacy_enrolled_speakers:
            logger.info("[legacy] Enrollment complete. Known speakers: %s", ", ".join(self._legacy_enrolled_speakers.keys()))

    def _embed_wave(self, wave: np.ndarray) -> Optional[np.ndarray]:
        if self._embedding_infer is None or wave.size < int(0.25 * self._cfg.sr):
            return None

        import torch

        try:
            with self._embed_lock:
                audio = {"waveform": torch.from_numpy(wave).unsqueeze(0), "sample_rate": self._cfg.sr}
                emb = self._embedding_infer(audio)
            emb = np.array(emb, dtype=np.float32)
            if emb.ndim > 1:
                emb = emb.mean(axis=0)
            emb /= np.linalg.norm(emb) + 1e-12
            return emb
        except Exception:
            logger.warning("Failed to embed wave", exc_info=True)
            return None

    def map_speaker_label(self, tenant_id: str, diar_label: str, wave_chunk: np.ndarray) -> str:
        if not USE_SPEAKER_ENROLLMENT:
            return diar_label

        # Prefer tenant-aware enrollment cache when enabled.
        if self._enrollment_cache is not None and tenant_id:
            try:
                snap = self._enrollment_cache.get(tenant_id)
                enrolled = snap.embeddings_by_label
            except Exception:
                logger.warning("Failed to load tenant enrollment tenant=%s", tenant_id, exc_info=True)
                enrolled = {}
        else:
            enrolled = {}

        # Legacy fallback only for default tenant.
        if not enrolled and (not tenant_id or tenant_id == "default"):
            enrolled = self._legacy_enrolled_speakers

        if not enrolled:
            logger.debug(
                "rtservice speaker mapping skipped: no enrolled speakers tenant=%s diar_label=%s",
                tenant_id,
                diar_label,
            )
            return diar_label

        emb = self._embed_wave(wave_chunk)
        if emb is None:
            return diar_label

        best_name = None
        best_sim = -1.0
        for name, ref in enrolled.items():
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        logger.debug(
            "rtservice enrollment mapping: tenant=%s diar_label=%s best=%s sim=%.3f threshold=%.3f candidates=%d",
            tenant_id,
            diar_label,
            best_name,
            best_sim,
            ENROLL_SIM_THRESHOLD,
            len(enrolled),
        )

        if best_name is not None and best_sim >= ENROLL_SIM_THRESHOLD:
            return best_name
        return diar_label


# ---------------------------------------------------------------------------
# Public rtservice engine
# ---------------------------------------------------------------------------


class RealtimeEngine:
    """Realtime diarization+ASR engine.

    Key properties:
    - heavy models are loaded once (in RealtimeModelBundle)
    - session state is per (tenant_id, session_id)
    - safe under concurrent sessions
    """

    def __init__(
        self,
        *,
        cfg: Optional[RealtimeConfig] = None,
        model_bundle: Optional[RealtimeModelBundle] = None,
        processor: Optional[RealtimeSessionProcessor] = None,
    ) -> None:
        self.cfg = cfg or RealtimeConfig()
        self.default_lang = DEFAULT_LANG

        self._sessions: Dict[Tuple[str, str], SessionState] = {}
        self._sessions_lock = threading.Lock()

        self._last_evict_scan_s = 0.0

        self._model_bundle = model_bundle

        # In unit tests we often inject a lightweight processor; in that case we
        # keep behavior fixed and ignore per-session config overrides.
        self._fixed_processor: Optional[RealtimeSessionProcessor] = processor

        if self._fixed_processor is None:
            if self._model_bundle is None:
                self._model_bundle = RealtimeModelBundle(cfg=self.cfg)

            self._processors_lock = threading.Lock()
            self._processors_by_key: Dict[Tuple[object, ...], RealtimeSessionProcessor] = {}
            # Ensure default processor is created.
            _p = RealtimeSessionProcessor(
                cfg=self.cfg,
                partial_gate_fn=self._model_bundle.gate_speech,
                window_gate_fn=self._model_bundle.gate_window,
                diarize_fn=self._model_bundle.diarize_window,
                asr_fn=self._model_bundle.asr_text,
                map_speaker_fn=self._model_bundle.map_speaker_label,
            )
            self._processors_by_key[self._cfg_key(self.cfg)] = _p

        logger.info(
            "RealtimeEngine ready: SR=%d WINDOW=%.2fs HOP=%.2fs default_lang=%s",
            self.cfg.sr,
            self.cfg.window_sec,
            self.cfg.hop_sec,
            self.default_lang,
        )

    def feed(
        self,
        session_id: str,
        pcm16: bytes,
        *,
        lang: Optional[str] = None,
        tenant_id: Optional[str] = None,
        rt_window_sec: Optional[float] = None,
        rt_overlap_sec: Optional[float] = None,
        rt_emit_every_sec: Optional[float] = None,
    ) -> List[AsrResult]:
        """Feed PCM16 audio bytes for a session.

        tenant_id:
        - used only for enrolled speaker resolution
        - defaults to "default" (dev) when not present
        """

        effective_lang = lang or self.default_lang
        tid = str(tenant_id or "default")
        sid = str(session_id or "unknown")

        now = time.time()
        self._maybe_evict_idle_sessions(now)

        key = (tid, sid)
        with self._sessions_lock:
            state = self._sessions.get(key)
            if state is None:
                state = SessionState.new(now)
                self._sessions[key] = state

        # Resolve processor/config (default or per-session overrides).
        proc = self._fixed_processor
        cfg_key = None
        if proc is None:
            cfg = self._effective_cfg(
                base=self.cfg,
                rt_window_sec=rt_window_sec,
                rt_overlap_sec=rt_overlap_sec,
                rt_emit_every_sec=rt_emit_every_sec,
            )
            cfg_key = self._cfg_key(cfg)
            proc = self._get_or_create_processor(cfg, cfg_key)

        # Per-session lock for correctness under concurrent sessions.
        with state.lock:
            state.last_activity_s = now

            # If per-session overrides changed mid-stream, reset session state.
            if cfg_key is not None and state.cfg_key is not None and state.cfg_key != cfg_key:
                logger.warning(
                    "rtservice: session config changed mid-stream; resetting session state tenant=%s session=%s",
                    tid,
                    sid,
                )
                st2 = SessionState.new(now)
                st2.cfg_key = cfg_key
                self._sessions[key] = st2
                state = st2
            elif cfg_key is not None and state.cfg_key is None:
                state.cfg_key = cfg_key

            return proc.process(
                tenant_id=tid,
                session_id=sid,
                state=state,
                pcm16=pcm16,
                lang=effective_lang,
            )

    @staticmethod
    def _cfg_key(cfg: RealtimeConfig) -> Tuple[object, ...]:
        # Float stability: round to millisecond-ish precision.
        def r(x: float) -> float:
            return round(float(x), 6)

        return (
            int(cfg.sr),
            r(cfg.window_sec),
            r(cfg.overlap_sec),
            bool(cfg.partial_enable),
            r(cfg.emit_every_sec),
            r(cfg.finalize_min_dur_sec),
            r(cfg.key_resolution_sec),
        )

    def _effective_cfg(
        self,
        *,
        base: RealtimeConfig,
        rt_window_sec: Optional[float],
        rt_overlap_sec: Optional[float],
        rt_emit_every_sec: Optional[float],
    ) -> RealtimeConfig:
        # Validate and apply per-session overrides.
        win = float(rt_window_sec) if rt_window_sec is not None else base.window_sec
        ov = float(rt_overlap_sec) if rt_overlap_sec is not None else base.overlap_sec
        emit = float(rt_emit_every_sec) if rt_emit_every_sec is not None else base.emit_every_sec

        # Basic sanity. We keep ranges permissive and let operators enforce tighter
        # policies in the BFF.
        if not np.isfinite(win) or win <= 0.1:
            win = base.window_sec
        if not np.isfinite(ov) or ov < 0.0 or ov >= win:
            ov = base.overlap_sec
        if not np.isfinite(emit) or emit <= 0.0:
            emit = base.emit_every_sec
        if emit > win:
            emit = win

        return RealtimeConfig(
            sr=base.sr,
            window_sec=win,
            overlap_sec=ov,
            partial_enable=base.partial_enable,
            emit_every_sec=emit,
            partial_stability_repeats=base.partial_stability_repeats,
            partial_min_buffer_sec=base.partial_min_buffer_sec,
            finalize_min_dur_sec=base.finalize_min_dur_sec,
            key_resolution_sec=base.key_resolution_sec,
        )

    def _get_or_create_processor(
        self,
        cfg: RealtimeConfig,
        cfg_key: Tuple[object, ...],
    ) -> RealtimeSessionProcessor:
        with self._processors_lock:
            p = self._processors_by_key.get(cfg_key)
            if p is not None:
                return p

            assert self._model_bundle is not None
            p = RealtimeSessionProcessor(
                cfg=cfg,
                partial_gate_fn=self._model_bundle.gate_speech,
                window_gate_fn=self._model_bundle.gate_window,
                diarize_fn=self._model_bundle.diarize_window,
                asr_fn=self._model_bundle.asr_text,
                map_speaker_fn=self._model_bundle.map_speaker_label,
            )
            self._processors_by_key[cfg_key] = p
            logger.info(
                "rtservice: created new processor for per-session cfg window=%.3fs overlap=%.3fs emit=%.3fs",
                cfg.window_sec,
                cfg.overlap_sec,
                cfg.emit_every_sec,
            )
            return p

    def to_asr_events(self, session_id: str, r: AsrResult) -> stream_pb2.AsrEvent:
        return stream_pb2.AsrEvent(
            session_id=session_id,
            start_s=r.start_s,
            end_s=r.end_s,
            text=r.text,
            type=stream_pb2.FINAL if r.is_final else stream_pb2.PARTIAL,
            lang=(r.lang or self.default_lang or ""),
            speaker=(r.speaker or ""),
        )

    def _maybe_evict_idle_sessions(self, now_s: float) -> None:
        if (now_s - self._last_evict_scan_s) < RT_SESSION_EVICT_SCAN_SECONDS:
            return
        self._last_evict_scan_s = now_s

        idle_before = now_s - RT_SESSION_IDLE_SECONDS

        with self._sessions_lock:
            to_delete = [k for k, st in self._sessions.items() if st.last_activity_s < idle_before]
            for k in to_delete:
                self._sessions.pop(k, None)

        if to_delete:
            logger.info("Evicted %d idle realtime sessions", len(to_delete))
