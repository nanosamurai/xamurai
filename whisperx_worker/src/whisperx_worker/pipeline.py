import logging
import os
import threading
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, TypedDict

import numpy as np
import soundfile as sf


# NOTE: torch is not installed in lightweight unit-test environments.
# Keep this import optional so `import whisperx_worker.pipeline` works
# even without torch.
try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from drsynth_common.pyannote_telemetry import disable_pyannote_telemetry


# Security: pyannote.audio may try to export telemetry to a remote OTLP endpoint.
# Force-disable it as early as possible in the process.
disable_pyannote_telemetry()


# -------------------- PyTorch checkpoint loading compatibility --------------------
#
# Recent PyTorch versions (2.6+) introduced safer defaults for torch.load() that can
# break loading some third-party checkpoints (including WhisperX -> pyannote VAD).
#
# In our dev stack we trust the upstream model checkpoints (HF/pyannote), so we:
# 1) allowlist OmegaConf classes for weights_only=True code paths
# 2) also monkeypatch torch.load to default to weights_only=False (more robust)
#
# NOTE: weights_only=False can load arbitrary pickled code. Do not use this with
# untrusted checkpoints.

if torch is not None:
    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata

        # Nodes show up in some serialized OmegaConf objects.
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

if torch is not None:
    try:
        # Make this idempotent across module reloads (tests do importlib.reload).
        # Store the original unwrapped torch.load on the torch module.
        if not hasattr(torch, "_drsynth_orig_load"):
            torch._drsynth_orig_load = torch.load  # type: ignore[attr-defined]

        if getattr(torch.load, "__name__", "") != "_torch_load_weights_only_false_default":
            _orig_torch_load = torch._drsynth_orig_load  # type: ignore[attr-defined]

            def _torch_load_weights_only_false_default(*args, **kwargs):
                # Force weights_only=False even if a caller explicitly passes True.
                # This is required for some Lightning/pyannote checkpoints containing
                # objects outside the default safe allowlist.
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)

            torch.load = _torch_load_weights_only_false_default
    except Exception:
        pass

# NOTE: whisperx is heavy and is not installed in every service environment.
# Import lazily inside functions so unit tests can still import this module.
whisperx = None

# diarization + enrollment mapping
# NOTE: we import pyannote lazily so this worker can still run in environments
# where WhisperX is available but diarization deps are not.
from drsynth_common.diarization_assign import (
    DiarizationSegment,
    TimeSegment,
    assign_speaker_by_overlap,
)

if TYPE_CHECKING:
    from drsynth_common.enrollment import EnrollmentCache

SR = 16000

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# WhisperX globals (initialized at startup)
# --------------------------------------------------------------------------- #
_WHISPERX_MODEL = None
_ALIGN_MODEL = None
_ALIGN_METADATA = None
_WHISPERX_DEVICE = None

# --------------------------------------------------------------------------- #
# Diarization + enrollment globals
# --------------------------------------------------------------------------- #

_ENABLE_DIARIZATION = os.getenv("WHISPERX_ENABLE_DIARIZATION", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
)

_DIAR_PIPE = None
_DIAR_DEVICE = None

_EMBED_INFER: Optional[object] = None
_EMBED_LOCK = threading.Lock()

# Legacy enrollment (flat dir)
_LEGACY_ENROLLED: Dict[str, np.ndarray] = {}

# Per-tenant enrollment cache (optional)
_ENROLL_CACHE: Optional["EnrollmentCache"] = None

ENROLL_SIM_THRESHOLD = float(os.getenv("ENROLL_SIM_THRESHOLD", "0.30"))

# Refined diarization quality fix:
#
# When enabled, use diarization segments to drive speaker turns (re-segmentation)
# instead of relying on WhisperX ASR segment boundaries.
#
# Why:
# - For long refinement windows, WhisperX often emits 1-2 coarse ASR segments.
# - Our default overlap-based assignment picks one speaker per ASR segment,
#   collapsing back-and-forth turns into a single speaker label.
#
# With split mode ON, we:
# - run pyannote diarization on the slice
# - merge turns (short gaps, drop tiny turns)
# - run WhisperX transcription per turn
# - emit segments with speaker = diarization speaker (or enrolled mapped label)

_DIAR_SPLIT_MODE = os.getenv("WHISPERX_DIAR_SPLIT_MODE", "off").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)

_DIAR_SPLIT_MIN_TURN_SEC = float(os.getenv("WHISPERX_DIAR_SPLIT_MIN_TURN_SEC", "0.7"))
_DIAR_SPLIT_MERGE_GAP_SEC = float(os.getenv("WHISPERX_DIAR_SPLIT_MERGE_GAP_SEC", "0.15"))
_DIAR_SPLIT_MAX_TURNS = int(os.getenv("WHISPERX_DIAR_SPLIT_MAX_TURNS", "40"))
_DIAR_SPLIT_MAX_AUDIO_SEC = float(os.getenv("WHISPERX_DIAR_SPLIT_MAX_AUDIO_SEC", "90.0"))

_DIAR_INIT_ATTEMPTED = False


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
    return np.interp(t_new, t_old, x).astype(np.float32)


def _init_diarization_models() -> None:
    """Best-effort initialization of diarization + embedding models."""

    global _DIAR_PIPE, _DIAR_DEVICE, _EMBED_INFER, _ENROLL_CACHE, _LEGACY_ENROLLED, _DIAR_INIT_ATTEMPTED

    if torch is None:
        logger.warning("torch not available; WhisperX diarization will be disabled")
        return

    if not _DIAR_INIT_ATTEMPTED:
        _DIAR_INIT_ATTEMPTED = True
        logger.info(
            "whisperx_worker diarization config: enabled=%s hf_token=%s model=%s enroll_backend=%s sim_threshold=%.3f",
            _ENABLE_DIARIZATION,
            bool(os.getenv("HF_TOKEN")),
            os.getenv("WHISPERX_DIAR_MODEL", "pyannote/speaker-diarization-3.1"),
            os.getenv("ENROLL_BACKEND", "legacy_dir"),
            ENROLL_SIM_THRESHOLD,
        )

        try:
            import importlib.metadata as _md

            logger.info(
                "whisperx_worker versions: torch=%s pyannote-audio=%s whisperx=%s",
                getattr(torch, "__version__", "unknown"),
                _md.version("pyannote-audio"),
                _md.version("whisperx"),
            )
        except Exception:
            logger.debug("whisperx_worker versions: torch=%s", getattr(torch, "__version__", "unknown"))

    if not _ENABLE_DIARIZATION:
        logger.info("WHISPERX_ENABLE_DIARIZATION=false; diarization disabled")
        return

    # If diarization pipeline is already initialized we keep it.
    # Embedding inference can be absent when enrollment mapping is disabled.
    if _DIAR_PIPE is not None:
        return

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set; WhisperX diarization will be disabled")
        return

    # Lazy imports
    try:
        from pyannote.audio import Pipeline, Model
        from pyannote.audio import Inference as EmbeddingInference
    except Exception:
        logger.warning("pyannote not available; WhisperX diarization disabled", exc_info=True)
        return

    _DIAR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    diar_model_id = os.getenv(
        "WHISPERX_DIAR_MODEL",
        "pyannote/speaker-diarization-3.1",
    ).strip()

    def _load_diar(mid: str):
        logger.info("Loading diarization pipeline (%s) on %s", mid, _DIAR_DEVICE)

        try:
            pipe = Pipeline.from_pretrained(mid, token=hf_token)
        except TypeError:
            pipe = Pipeline.from_pretrained(mid, use_auth_token=hf_token)

        pipe.to(torch.device(_DIAR_DEVICE))
        return pipe

    try:
        _DIAR_PIPE = _load_diar(diar_model_id)
    except Exception:
        fallback = "pyannote/speaker-diarization-3.1"
        if diar_model_id != fallback:
            logger.warning(
                "Failed to init diarization pipeline (%s); trying fallback %s",
                diar_model_id,
                fallback,
                exc_info=True,
            )
            try:
                _DIAR_PIPE = _load_diar(fallback)
            except Exception:
                logger.warning("Failed to init diarization pipeline; diarization disabled", exc_info=True)
                _DIAR_PIPE = None
                return
        else:
            logger.warning("Failed to init diarization pipeline; diarization disabled", exc_info=True)
            _DIAR_PIPE = None
            return

    try:
        logger.info("Loading embedding model (pyannote/embedding) on %s", _DIAR_DEVICE)
        try:
            emb_model = Model.from_pretrained("pyannote/embedding", token=hf_token)
        except TypeError:
            emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        _EMBED_INFER = EmbeddingInference(emb_model, window="whole")
    except Exception:
        logger.warning("Failed to init embedding model; enrollment mapping disabled", exc_info=True)
        _EMBED_INFER = None
        return

    backend = os.getenv("ENROLL_BACKEND", "legacy_dir").strip().lower()

    if backend in ("none", "disabled"):
        logger.info("ENROLL_BACKEND=%s; enrollment mapping disabled", backend)
        return

    if backend != "legacy_dir":
        try:
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
                import io

                b = io.BytesIO(sample_bytes)
                x, sr = sf.read(b, dtype="float32")
                if isinstance(x, np.ndarray) and x.ndim > 1:
                    x = x.mean(axis=1)
                x = np.asarray(x, dtype=np.float32).reshape(-1)
                if int(sr) != SR:
                    x = _resample_linear(x, int(sr), SR)
                emb = _embed_wave(x)
                if emb is None:
                    raise RuntimeError("Failed to embed enrollment sample")
                return emb

            _ENROLL_CACHE = EnrollmentCache(
                provider=provider,
                embed_fn=_embed_bytes,
                ttl_s=ttl_s,
                max_tenants=max_tenants,
            )
            logger.info("Enrollment cache enabled for whisperx_worker: backend=%s", backend)
        except Exception:
            logger.warning("Failed to initialize enrollment cache; falling back to legacy_dir", exc_info=True)
            _ENROLL_CACHE = None

    enroll_dir = os.getenv("ENROLL_DIR", "enrolled_speakers")
    if os.path.isdir(enroll_dir):
        per_name: Dict[str, List[np.ndarray]] = defaultdict(list)
        for fname in os.listdir(enroll_dir):
            if not fname.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                continue
            path = os.path.join(enroll_dir, fname)
            base = os.path.splitext(fname)[0]
            name = base.split("_")[0]
            try:
                # IMPORTANT:
                # Passing a filesystem path to pyannote's Inference triggers its
                # internal audio decoding (AudioDecoder/torchcodec), which is
                # brittle in CI environments.
                #
                # Instead, load the audio ourselves with soundfile and pass an
                # in-memory waveform dict.
                wav, sr = sf.read(path, dtype="float32")
                if isinstance(wav, np.ndarray) and wav.ndim > 1:
                    wav = wav.mean(axis=1)
                wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                if int(sr) != SR:
                    wav = _resample_linear(wav, int(sr), SR)

                with _EMBED_LOCK:
                    emb = _EMBED_INFER(
                        {"waveform": torch.from_numpy(wav).unsqueeze(0), "sample_rate": SR}
                    )  # type: ignore[operator]

                emb = np.array(emb, dtype=np.float32)
                if emb.ndim > 1:
                    emb = emb.mean(axis=0)
                per_name[name].append(emb)
            except Exception:
                logger.warning("Failed to legacy-enroll from %s", fname, exc_info=True)

        for name, embs in per_name.items():
            if not embs:
                continue
            arr = np.stack(embs, axis=0)
            mean = arr.mean(axis=0)
            mean /= np.linalg.norm(mean) + 1e-12
            _LEGACY_ENROLLED[name] = mean

        if _LEGACY_ENROLLED:
            logger.info("Legacy enrollment loaded: %s", ", ".join(sorted(_LEGACY_ENROLLED.keys())))


def _embed_wave(wave: np.ndarray) -> Optional[np.ndarray]:
    if _EMBED_INFER is None or wave.size < int(0.25 * SR):
        return None

    if torch is None:
        return None

    try:
        with _EMBED_LOCK:
            audio = {"waveform": torch.from_numpy(wave).unsqueeze(0), "sample_rate": SR}
            emb = _EMBED_INFER(audio)  # type: ignore[operator]
        emb = np.array(emb, dtype=np.float32)
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        emb /= np.linalg.norm(emb) + 1e-12
        return emb
    except Exception:
        logger.warning("Failed to embed wave", exc_info=True)
        return None


def _get_enrolled_embeddings_for_tenant(tenant: Optional[str]) -> Dict[str, np.ndarray]:
    t = (tenant or "default").strip() or "default"
    if _ENROLL_CACHE is not None and t:
        try:
            snap = _ENROLL_CACHE.get(t)
            if snap.embeddings_by_label:
                return snap.embeddings_by_label
        except Exception:
            logger.warning("Failed to load enrollment cache snapshot tenant=%s", t, exc_info=True)

    if t == "default":
        return _LEGACY_ENROLLED

    return {}


def _map_embedding_to_enrolled_label(emb: np.ndarray, enrolled: Dict[str, np.ndarray]) -> str:
    if not enrolled:
        return ""

    best_name = ""
    best_sim = -1.0
    for name, ref in enrolled.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_sim = sim
            best_name = name

    logger.debug(
        "Enrollment mapping best=%s sim=%.3f threshold=%.3f candidates=%d",
        best_name,
        best_sim,
        ENROLL_SIM_THRESHOLD,
        len(enrolled),
    )

    if best_name and best_sim >= ENROLL_SIM_THRESHOLD:
        return best_name
    return ""


def _diarize_audio(audio: np.ndarray) -> List[DiarizationSegment]:
    if _DIAR_PIPE is None:
        return []

    if torch is None:
        return []

    try:
        w = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
        out = _DIAR_PIPE({"waveform": w, "sample_rate": SR})

        diar = getattr(out, "speaker_diarization", None)
        if diar is None:
            diar = out

        segs: List[DiarizationSegment] = []
        for turn, _, spk in diar.itertracks(yield_label=True):
            segs.append(DiarizationSegment(start_s=float(turn.start), end_s=float(turn.end), speaker=str(spk)))

        if segs:
            uniq = sorted({s.speaker for s in segs if s.speaker})
            logger.debug("Diarization produced %d segments (speakers=%s)", len(segs), uniq)
        else:
            logger.info("Diarization produced 0 segments")

        return segs
    except Exception:
        logger.warning("Diarization failed", exc_info=True)
        return []


def _merge_diarization_turns(
    diar: List[DiarizationSegment],
    *,
    min_turn_sec: float,
    merge_gap_sec: float,
    max_turns: int,
) -> List[DiarizationSegment]:
    """Merge diarization segments into a stable set of speaker turns.

    We merge consecutive segments for the same speaker when gaps are small, and
    drop segments shorter than `min_turn_sec`.
    """

    if not diar:
        return []

    # Sort by time just in case.
    diar_sorted = sorted(diar, key=lambda s: (float(s.start_s), float(s.end_s)))
    out: List[DiarizationSegment] = []

    for seg in diar_sorted:
        s0 = float(seg.start_s)
        s1 = float(seg.end_s)
        spk = str(seg.speaker or "")
        if s1 <= s0:
            continue
        # Drop tiny turns (they tend to be diarization noise and amplify compute).
        if (s1 - s0) < float(min_turn_sec):
            continue
        if not spk:
            continue

        if not out:
            out.append(DiarizationSegment(start_s=s0, end_s=s1, speaker=spk))
            continue

        prev = out[-1]
        gap = s0 - float(prev.end_s)
        if spk == prev.speaker and gap <= float(merge_gap_sec):
            out[-1] = DiarizationSegment(start_s=float(prev.start_s), end_s=max(float(prev.end_s), s1), speaker=spk)
        else:
            out.append(DiarizationSegment(start_s=s0, end_s=s1, speaker=spk))

        if len(out) >= int(max_turns):
            # Hard bound to prevent compute amplification.
            break

    return out


def _transcribe_audio_array(
    audio: np.ndarray,
    *,
    lang: Optional[str],
) -> Tuple[str, List[Tuple[float, float, str]]]:
    """Transcribe a float32 waveform array with WhisperX model.

    Returns (full_text, segments) where segments are relative to the input array.
    """

    if audio.size == 0:
        return "", []

    try:
        if lang:
            result = _WHISPERX_MODEL.transcribe(audio, batch_size=16, language=lang)
        else:
            result = _WHISPERX_MODEL.transcribe(audio, batch_size=16)
    except IndexError:
        # WhisperX raises IndexError when no speech is detected in some versions.
        return "", []

    asr_segments = result.get("segments", []) or []
    segs_out: List[Tuple[float, float, str]] = []
    texts: List[str] = []
    for seg in asr_segments:
        try:
            s0 = float(seg.get("start", 0.0))
            s1 = float(seg.get("end", 0.0))
            text = (seg.get("text") or "").strip()
        except Exception:
            continue
        if not text or s1 <= s0:
            continue
        segs_out.append((s0, s1, text))
        texts.append(text)
    return " ".join(texts).strip(), segs_out


def run_whisperx_diarized(
    wav_path: str,
    *,
    tenant: Optional[str],
    lang: Optional[str],
    use_alignment: bool,
) -> Tuple[str, List[Tuple[float, float, str, str]]]:
    """Run ASR and (optionally) diarization+enrollment mapping."""

    _init_diarization_models()

    # If torch isn't available, diarization cannot run.
    if torch is None:
        full_text, segs = run_whisperx(wav_path, lang=lang, use_alignment=use_alignment)
        return full_text, segs

    # We need the slice audio regardless: diarization may drive re-segmentation.
    _ensure_whisperx_imported()
    try:
        audio = whisperx.load_audio(wav_path)
    except FileNotFoundError:
        audio, _sr = sf.read(wav_path, dtype="float32")

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    # Baseline ASR over full slice.
    full_text, segs = run_whisperx(wav_path, lang=lang, use_alignment=use_alignment)

    if not _ENABLE_DIARIZATION or _DIAR_PIPE is None:
        return full_text, segs

    diar = _diarize_audio(audio)
    if not diar:
        return full_text, segs

    uniq_speakers = sorted({d.speaker for d in diar if d.speaker})
    logger.info(
        "whisperx_worker diarization: diar_segments=%d unique_speakers=%d split_mode=%s",
        len(diar),
        len(uniq_speakers),
        _DIAR_SPLIT_MODE,
    )

    # Compute enrollment mapping (optional, best-effort).
    enrolled = _get_enrolled_embeddings_for_tenant(tenant)
    cluster_map: Dict[str, str] = {}
    if enrolled and _EMBED_INFER is not None:
        unique_clusters = sorted({d.speaker for d in diar if d.speaker})
        for cluster in unique_clusters:
            chunks = []
            for d in diar:
                if d.speaker != cluster:
                    continue
                s_idx = int(max(0.0, d.start_s) * SR)
                e_idx = int(min(float(audio.size) / SR, d.end_s) * SR)
                if e_idx > s_idx:
                    chunks.append(audio[s_idx:e_idx])

            if not chunks:
                continue

            wave = np.concatenate(chunks)
            max_samples = int(10.0 * SR)
            if wave.size > max_samples:
                wave = wave[:max_samples]

            emb = _embed_wave(wave)
            if emb is None:
                continue

            mapped = _map_embedding_to_enrolled_label(emb, enrolled)
            if mapped:
                cluster_map[cluster] = mapped

    # If split mode is enabled and multiple speakers are present, use diarization
    # turns to drive segmentation and transcribe per turn.
    if _DIAR_SPLIT_MODE and len(uniq_speakers) >= 2:
        audio_dur_s = float(audio.size) / float(SR)
        if audio_dur_s > float(_DIAR_SPLIT_MAX_AUDIO_SEC):
            logger.warning(
                "WHISPERX_DIAR_SPLIT_MODE on but slice too long (%.2fs > %.2fs). Falling back to overlap assignment.",
                audio_dur_s,
                float(_DIAR_SPLIT_MAX_AUDIO_SEC),
            )
        else:
            turns = _merge_diarization_turns(
                diar,
                min_turn_sec=float(_DIAR_SPLIT_MIN_TURN_SEC),
                merge_gap_sec=float(_DIAR_SPLIT_MERGE_GAP_SEC),
                max_turns=int(_DIAR_SPLIT_MAX_TURNS),
            )

            logger.info(
                "whisperx_worker diar split: turns=%d min_turn=%.2fs merge_gap=%.2fs max_turns=%d",
                len(turns),
                float(_DIAR_SPLIT_MIN_TURN_SEC),
                float(_DIAR_SPLIT_MERGE_GAP_SEC),
                int(_DIAR_SPLIT_MAX_TURNS),
            )

            out2: List[Tuple[float, float, str, str]] = []
            full_parts: List[str] = []
            for t in turns:
                s_idx = int(max(0.0, float(t.start_s)) * SR)
                e_idx = int(min(float(audio.size) / SR, float(t.end_s)) * SR)
                if e_idx <= s_idx:
                    continue
                wave = audio[s_idx:e_idx]
                # Safety: WhisperX expects float32.
                wave = np.asarray(wave, dtype=np.float32)
                txt, segs_rel = _transcribe_audio_array(wave, lang=lang)
                if txt:
                    full_parts.append(txt)
                spk = cluster_map.get(t.speaker) or t.speaker
                for (rs0, rs1, rtxt) in segs_rel:
                    out2.append((float(t.start_s) + rs0, float(t.start_s) + rs1, rtxt, spk))

            if out2:
                return " ".join(full_parts).strip(), out2

            logger.info("whisperx_worker diar split produced 0 ASR segments; falling back")

    # Default behavior: assign speakers to ASR segments via overlap.
    asr_time_segs = [TimeSegment(start_s=s0, end_s=s1) for (s0, s1, _txt, _spk) in segs]
    diar_speakers = assign_speaker_by_overlap(
        segments=asr_time_segs,
        diarization=diar,
        default_speaker="",
        min_coverage=0.1,
    )

    assigned = sum(1 for s in diar_speakers if s)
    logger.info(
        "Overlap assignment: asr_segments=%d assigned=%d unique=%d",
        len(asr_time_segs),
        assigned,
        len(set(diar_speakers)),
    )

    out: List[Tuple[float, float, str, str]] = []
    for (seg, cluster) in zip(segs, diar_speakers):
        s0, s1, txt, _old = seg
        speaker = cluster_map.get(cluster) or cluster or ""
        out.append((s0, s1, txt, speaker))

    return full_text, out


class WordTiming(TypedDict):
    start_s: float
    end_s: float
    text: str


class SegmentTiming(TypedDict):
    start_s: float
    end_s: float
    text: str
    words: List[WordTiming]


class DiarizedSegmentTiming(SegmentTiming):
    speaker: str


def _extract_word_timings(seg: Dict[str, Any]) -> List[WordTiming]:
    """Normalize WhisperX `align()` segment words into a stable shape.

    WhisperX returns words as objects like:
      {"word": "hello", "start": 1.23, "end": 1.45, ...}

    Some words may have missing timestamps (start/end None) → we skip them.
    """

    out: List[WordTiming] = []
    raw_words = seg.get("words") or []
    if not isinstance(raw_words, list):
        return out

    for w in raw_words:
        if not isinstance(w, dict):
            continue
        s = w.get("start")
        e = w.get("end")
        if s is None or e is None:
            continue
        txt = (w.get("word") or w.get("text") or "").strip()
        if not txt:
            continue
        try:
            out.append({"start_s": float(s), "end_s": float(e), "text": txt})
        except Exception:
            continue

    return out


def run_whisperx_words(
    wav_path: str,
    lang: Optional[str] = None,
    use_alignment: bool = False,
    alignment_min_coverage: float = 0.7,
) -> Tuple[str, List[SegmentTiming]]:
    """Run WhisperX ASR and (optionally) alignment, returning word-level timing.

    This is a superset of `run_whisperx()`:
    - When `use_alignment=False`, segments will have `words=[]`.
    - When `use_alignment=True` and WhisperX alignment succeeds, segments will have
      `words` with per-word start/end.

    Note: returned word timestamps are relative to the input WAV.
    """

    _init_whisperx(lang_hint=lang)

    try:
        audio = whisperx.load_audio(wav_path)
    except FileNotFoundError:
        logger.warning(
            "ffmpeg not found while loading %s; falling back to soundfile WAV loader",
            wav_path,
        )
        audio, _sr = sf.read(wav_path, dtype="float32")

    logger.info(
        "WhisperX: transcribing %s (lang=%s, use_alignment=%s)",
        wav_path,
        lang or "auto",
        use_alignment,
    )

    if lang:
        try:
            result = _WHISPERX_MODEL.transcribe(audio, batch_size=16, language=lang)
        except IndexError:
            logger.info("WhisperX: no active speech detected (lang=%s)", lang)
            return "", []
    else:
        try:
            result = _WHISPERX_MODEL.transcribe(audio, batch_size=16)
        except IndexError:
            logger.info("WhisperX: no active speech detected")
            return "", []

    detected_lang = result.get("language", lang or "unknown")
    logger.debug("WhisperX: detected language=%s", detected_lang)

    asr_segments = result.get("segments", []) or []

    def from_asr_segments() -> Tuple[str, List[SegmentTiming]]:
        segs_out: List[SegmentTiming] = []
        words_out: List[str] = []
        for seg in asr_segments:
            s0 = float(seg.get("start", 0.0))
            s1 = float(seg.get("end", 0.0))
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segs_out.append({"start_s": s0, "end_s": s1, "text": text, "words": []})
            words_out.append(text)
        full = " ".join(words_out).strip()
        logger.info(
            "WhisperX (no align): %d segments, total_text_len=%d",
            len(segs_out),
            len(full),
        )
        return full, segs_out

    if not use_alignment:
        return from_asr_segments()

    try:
        _ensure_align_model(detected_lang)
        aligned = whisperx.align(
            asr_segments,
            _ALIGN_MODEL,
            _ALIGN_METADATA,
            audio,
            _WHISPERX_DEVICE,
            return_char_alignments=False,
        )

        aligned_segments = aligned.get("segments", []) or []

        def total_duration(segs: List[Dict[str, Any]]) -> float:
            return sum(
                max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0))) for s in segs
            )

        asr_cov = total_duration(asr_segments)
        aln_cov = total_duration(aligned_segments)
        coverage_ratio = (aln_cov / asr_cov) if asr_cov > 0 else 1.0

        if asr_cov > 0 and coverage_ratio < alignment_min_coverage:
            logger.warning(
                "WhisperX alignment coverage too low: %.2f (ASR=%.2fs, aligned=%.2fs). Falling back.",
                coverage_ratio,
                asr_cov,
                aln_cov,
            )
            return from_asr_segments()

        segs_out: List[SegmentTiming] = []
        texts_out: List[str] = []

        for seg in aligned_segments:
            s0 = float(seg.get("start", 0.0))
            s1 = float(seg.get("end", 0.0))
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            words = _extract_word_timings(seg)
            segs_out.append({"start_s": s0, "end_s": s1, "text": text, "words": words})
            texts_out.append(text)

        full = " ".join(texts_out).strip()
        logger.info(
            "WhisperX (aligned): %d segments, total_text_len=%d, coverage_ratio=%.2f",
            len(segs_out),
            len(full),
            coverage_ratio,
        )
        return full, segs_out

    except Exception as e:
        logger.warning("WhisperX alignment failed (%s); falling back.", e)
        return from_asr_segments()


def run_whisperx_diarized_words(
    wav_path: str,
    *,
    tenant: Optional[str],
    lang: Optional[str],
    use_alignment: bool,
) -> Tuple[str, List[DiarizedSegmentTiming]]:
    """Like `run_whisperx_diarized`, but returns segments with word-level timestamps."""

    _init_diarization_models()

    # If torch isn't available, diarization cannot run.
    if torch is None:
        full_text, segs = run_whisperx_words(
            wav_path,
            lang=lang,
            use_alignment=use_alignment,
        )
        return full_text, [
            {"start_s": s["start_s"], "end_s": s["end_s"], "text": s["text"], "speaker": "", "words": s["words"]}
            for s in segs
        ]

    full_text, segs = run_whisperx_words(
        wav_path,
        lang=lang,
        use_alignment=use_alignment,
    )

    # If diarization isn't available, still return segments (speaker empty).
    if not _ENABLE_DIARIZATION or _DIAR_PIPE is None:
        return full_text, [
            {"start_s": s["start_s"], "end_s": s["end_s"], "text": s["text"], "speaker": "", "words": s["words"]}
            for s in segs
        ]

    _ensure_whisperx_imported()
    try:
        audio = whisperx.load_audio(wav_path)
    except FileNotFoundError:
        audio, _sr = sf.read(wav_path, dtype="float32")

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    diar = _diarize_audio(audio)
    if not diar:
        return full_text, [
            {"start_s": s["start_s"], "end_s": s["end_s"], "text": s["text"], "speaker": "", "words": s["words"]}
            for s in segs
        ]

    asr_time_segs = [TimeSegment(start_s=s["start_s"], end_s=s["end_s"]) for s in segs]
    diar_speakers = assign_speaker_by_overlap(
        segments=asr_time_segs,
        diarization=diar,
        default_speaker="",
        min_coverage=0.1,
    )

    assigned = sum(1 for s in diar_speakers if s)
    logger.info(
        "Overlap assignment: asr_segments=%d assigned=%d unique=%d",
        len(asr_time_segs),
        assigned,
        len(set(diar_speakers)),
    )

    cluster_map: Dict[str, str] = {}
    enrolled = _get_enrolled_embeddings_for_tenant(tenant)
    if enrolled and _EMBED_INFER is not None:
        unique_clusters = sorted({s for s in diar_speakers if s})
        for cluster in unique_clusters:
            chunks = []
            for d in diar:
                if d.speaker != cluster:
                    continue
                s_idx = int(max(0.0, d.start_s) * SR)
                e_idx = int(min(float(audio.size) / SR, d.end_s) * SR)
                if e_idx > s_idx:
                    chunks.append(audio[s_idx:e_idx])

            if not chunks:
                continue

            wave = np.concatenate(chunks)
            max_samples = int(10.0 * SR)
            if wave.size > max_samples:
                wave = wave[:max_samples]

            emb = _embed_wave(wave)
            if emb is None:
                continue

            mapped = _map_embedding_to_enrolled_label(emb, enrolled)
            if mapped:
                cluster_map[cluster] = mapped

    out: List[DiarizedSegmentTiming] = []
    for (seg, cluster) in zip(segs, diar_speakers):
        speaker = cluster_map.get(cluster) or cluster or ""
        out.append(
            {
                "start_s": seg["start_s"],
                "end_s": seg["end_s"],
                "text": seg["text"],
                "speaker": speaker,
                "words": seg["words"],
            }
        )

    return full_text, out


def _ensure_whisperx_imported() -> None:
    global whisperx
    if whisperx is not None:
        return
    import importlib

    whisperx = importlib.import_module("whisperx")


def _init_whisperx(lang_hint: Optional[str] = None) -> None:
    if torch is None:
        raise RuntimeError("torch is required for whisperx_worker inference but is not installed")

    global _WHISPERX_MODEL, _ALIGN_MODEL, _ALIGN_METADATA, _WHISPERX_DEVICE

    _ensure_whisperx_imported()

    if _WHISPERX_MODEL is not None:
        logger.debug("WhisperX model already initialized, skipping.")
        return

    _WHISPERX_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    env_compute = os.getenv("WHISPERX_COMPUTE_TYPE", "").strip()
    if env_compute:
        compute_type = env_compute
    else:
        compute_type = "float16" if _WHISPERX_DEVICE == "cuda" else "int8"

    asr_model = os.getenv("WHISPERX_MODEL", "medium").strip() or "medium"

    logger.info(
        "Loading WhisperX ASR model (%s) on %s, compute_type=%s",
        asr_model,
        _WHISPERX_DEVICE,
        compute_type,
    )

    # NOTE: `vad_method` is optional. Passing `None` can break on some whisperx
    # versions (expects a valid string). Only pass it when explicitly set.
    load_kwargs = {}
    vad_method = os.getenv("WHISPERX_VAD_METHOD", "").strip()
    if vad_method:
        load_kwargs["vad_method"] = vad_method

    _WHISPERX_MODEL = whisperx.load_model(
        asr_model,
        device=_WHISPERX_DEVICE,
        compute_type=compute_type,
        **load_kwargs,
    )

    _ALIGN_MODEL = None
    _ALIGN_METADATA = None

    logger.info("WhisperX ASR model initialized successfully on %s", _WHISPERX_DEVICE)


def _ensure_align_model(language_code: str) -> None:
    global _ALIGN_MODEL, _ALIGN_METADATA, _WHISPERX_DEVICE

    _ensure_whisperx_imported()

    if _ALIGN_MODEL is not None and _ALIGN_METADATA is not None:
        if _ALIGN_METADATA.get("language") == language_code:
            return

    logger.info("Loading WhisperX alignment model for language=%s", language_code)
    _ALIGN_MODEL, _ALIGN_METADATA = whisperx.load_align_model(
        language_code=language_code,
        device=_WHISPERX_DEVICE,
    )


def run_whisperx(
    wav_path: str,
    lang: Optional[str] = None,
    use_alignment: bool = False,
    alignment_min_coverage: float = 0.7,
) -> Tuple[str, List[Tuple[float, float, str, str]]]:
    full, segs = run_whisperx_words(
        wav_path,
        lang=lang,
        use_alignment=use_alignment,
        alignment_min_coverage=alignment_min_coverage,
    )
    return full, [(s["start_s"], s["end_s"], s["text"], "") for s in segs]
