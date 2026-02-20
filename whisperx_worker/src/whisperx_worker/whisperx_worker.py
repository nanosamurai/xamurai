import logging
import os
import tempfile
import threading
from collections import defaultdict, deque
from typing import Deque, Tuple, List, Dict, Optional, TYPE_CHECKING

import numpy as np
import soundfile as sf
from confluent_kafka import Consumer, Producer, KafkaException

from proto_gen import stream_pb2

import time

import torch

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

# NOTE: whisperx is heavy and not installed in all envs (e.g. drsynth-bff).
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

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_AUDIO = os.getenv("KAFKA_TOPIC_AUDIO", "audio.raw")
TOPIC_REFINED = os.getenv("KAFKA_TOPIC_REFINED", "transcripts.refined")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "whisperx-async")

SLICE_SECONDS = float(os.getenv("WHISPERX_SLICE_SECONDS", "60.0"))
SESSION_IDLE_SEC = float(os.getenv("WHISPERX_IDLE_SECONDS", "30.0"))
SR = 16000

last_activity: Dict[str, float] = defaultdict(lambda: 0.0)
session_lang: Dict[str, Optional[str]] = defaultdict(lambda: None)
bff_origin_uri: Dict[str, Optional[str]] = defaultdict(lambda: None)
tenant_id: Dict[str, Optional[str]] = defaultdict(lambda: None)

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
    """Best-effort initialization of diarization + embedding models.

    This is optional and can be disabled via env.
    """
    global _DIAR_PIPE, _DIAR_DEVICE, _EMBED_INFER, _ENROLL_CACHE, _LEGACY_ENROLLED

    if not _ENABLE_DIARIZATION:
        logger.info("WHISPERX_ENABLE_DIARIZATION=false; diarization disabled")
        return

    if _DIAR_PIPE is not None and _EMBED_INFER is not None:
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
        # NOTE: pyannote.audio 3.x compatible pipeline id
        "pyannote/speaker-diarization-3.1",
    ).strip()

    def _load_diar(mid: str):
        logger.info("Loading diarization pipeline (%s) on %s", mid, _DIAR_DEVICE)
        # pyannote.audio 3.x uses use_auth_token=... (older versions used token=...)
        pipe = Pipeline.from_pretrained(mid, use_auth_token=hf_token)
        pipe.to(torch.device(_DIAR_DEVICE))
        return pipe

    try:
        _DIAR_PIPE = _load_diar(diar_model_id)
    except Exception:
        # Compatibility fallback: older/community models may not load under pyannote 3.x.
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
        emb_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        _EMBED_INFER = EmbeddingInference(emb_model, window="whole")
    except Exception:
        logger.warning("Failed to init embedding model; enrollment mapping disabled", exc_info=True)
        _EMBED_INFER = None
        return

    # Enrollment: either cache-backed tenant manifests or legacy flat dir
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

    # Legacy flat dir is always allowed as a dev fallback.
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
                with _EMBED_LOCK:
                    emb = _EMBED_INFER(path)  # type: ignore[operator]
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

    # legacy fallback only for default tenant
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

    if best_name and best_sim >= ENROLL_SIM_THRESHOLD:
        return best_name
    return ""


def _diarize_audio(audio: np.ndarray) -> List[DiarizationSegment]:
    if _DIAR_PIPE is None:
        return []

    try:
        w = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
        out = _DIAR_PIPE({"waveform": w, "sample_rate": SR})
        segs: List[DiarizationSegment] = []
        # pyannote returns an Annotation; iterate speaker-labeled segments
        for turn, _, spk in out.itertracks(yield_label=True):
            segs.append(DiarizationSegment(start_s=float(turn.start), end_s=float(turn.end), speaker=str(spk)))
        return segs
    except Exception:
        logger.warning("Diarization failed", exc_info=True)
        return []


def run_whisperx_diarized(
    wav_path: str,
    *,
    tenant: Optional[str],
    lang: Optional[str],
    use_alignment: bool,
) -> Tuple[str, List[Tuple[float, float, str, str]]]:
    """Run ASR and (optionally) diarization+enrollment mapping."""

    # Make sure diarization models are ready (best-effort).
    _init_diarization_models()

    full_text, segs = run_whisperx(wav_path, lang=lang, use_alignment=use_alignment)

    if not _ENABLE_DIARIZATION or _DIAR_PIPE is None or _EMBED_INFER is None:
        return full_text, segs

    # Load audio for diarization
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
        return full_text, segs

    # Assign diarization speakers to ASR segments by overlap
    asr_time_segs = [TimeSegment(start_s=s0, end_s=s1) for (s0, s1, _txt, _spk) in segs]
    diar_speakers = assign_speaker_by_overlap(
        segments=asr_time_segs,
        diarization=diar,
        default_speaker="",
        min_coverage=0.1,
    )

    # Map diar speaker clusters -> enrolled labels (tenant-aware)
    enrolled = _get_enrolled_embeddings_for_tenant(tenant)

    cluster_map: Dict[str, str] = {}
    if enrolled:
        unique_clusters = sorted({s for s in diar_speakers if s})
        for cluster in unique_clusters:
            # build a representative embedding for this cluster
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
            # cap to 10 seconds for cost
            max_samples = int(10.0 * SR)
            if wave.size > max_samples:
                wave = wave[:max_samples]

            emb = _embed_wave(wave)
            if emb is None:
                continue

            mapped = _map_embedding_to_enrolled_label(emb, enrolled)
            if mapped:
                cluster_map[cluster] = mapped

    # Build final segments
    out: List[Tuple[float, float, str, str]] = []
    for (seg, cluster) in zip(segs, diar_speakers):
        s0, s1, txt, _old = seg
        speaker = cluster_map.get(cluster) or cluster or ""
        out.append((s0, s1, txt, speaker))

    return full_text, out


def _flush_session_partial(
    session_id: str,
    lang_hint: Optional[str],
    buffers: Dict[str, Deque[np.ndarray]],
    buf_samples: Dict[str, int],
    slice_index: Dict[str, int],
    producer: Producer,
    session_lang: Dict[str, Optional[str]],
    bff_origin_uri: Dict[str, Optional[str]],
    tenant_id: Dict[str, Optional[str]],
) -> None:
    """
    Flush whatever remains in buffers[session_id] as a final (possibly < SLICE_SECONDS)
    slice, run WhisperX, emit RefinedEvent(s), and clear the session buffers.
    """
    if buf_samples[session_id] <= 0 or not buffers[session_id]:
        # Nothing to flush
        buffers.pop(session_id, None)
        buf_samples.pop(session_id, None)
        slice_index.pop(session_id, None)
        return

    parts: List[np.ndarray] = []
    while buffers[session_id]:
        parts.append(buffers[session_id].popleft())
    buf_samples[session_id] = 0

    pcm = np.concatenate(parts).astype("<i2")

    # Write temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    x = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    sf.write(wav_path, x, SR, subtype="PCM_16")

    base_start = slice_index[session_id] * SLICE_SECONDS
    slice_index[session_id] += 1

    logger.info(
        "Idle flush for session %s: final slice dur=%.2fs → %s",
        session_id,
        len(pcm) / SR,
        wav_path,
    )

    try:
        text, segments = run_whisperx_diarized(
            wav_path,
            tenant=tenant_id.get(session_id),
            lang=lang_hint,
            use_alignment=False,  # keep streaming worker lightweight
        )
        logger.debug("Idle WhisperX result (session=%s): %s", session_id, text)

        for (s0, s1, seg_text, speaker) in segments:
            abs_start = base_start + s0
            abs_end = base_start + s1
            ev = stream_pb2.RefinedEvent(
                session_id=session_id,
                start_s=abs_start,
                end_s=abs_end,
                text=seg_text,
                speaker=speaker,
                supersedes_seq=[],
                lang=session_lang.get(session_id),
                bff_origin_uri=bff_origin_uri.get(session_id),
                tenant_id=tenant_id.get(session_id),
            )
            producer.produce(
                topic=TOPIC_REFINED,
                key=session_id.encode("utf-8"),
                value=ev.SerializeToString(),
            )
        producer.poll(0)
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    # Clear session state
    buffers.pop(session_id, None)
    buf_samples.pop(session_id, None)
    slice_index.pop(session_id, None)

def _ensure_whisperx_imported() -> None:
    global whisperx
    if whisperx is not None:
        return
    import importlib

    whisperx = importlib.import_module("whisperx")


def _init_whisperx(lang_hint: Optional[str] = None) -> None:
    """
    Global initialization of WhisperX.

    Called once at startup from main() so the first slice does not pay
    the full model download/compile cost.
    """
    global _WHISPERX_MODEL, _ALIGN_MODEL, _ALIGN_METADATA, _WHISPERX_DEVICE

    _ensure_whisperx_imported()

    if _WHISPERX_MODEL is not None:
        logger.debug("WhisperX model already initialized, skipping.")
        return

    # 🔥 Hard-coded policy: use CUDA if available, otherwise CPU
    if torch.cuda.is_available():
        _WHISPERX_DEVICE = "cuda"
    else:
        _WHISPERX_DEVICE = "cpu"

    # Optionally let compute_type be overridden via env,
    # but default to float16 on GPU, int8 on CPU.
    env_compute = os.getenv("WHISPERX_COMPUTE_TYPE", "").strip()
    if env_compute:
        compute_type = env_compute
    else:
        compute_type = "float16" if _WHISPERX_DEVICE == "cuda" else "int8"

    logger.info(
        "Loading WhisperX ASR model (medium) on %s, compute_type=%s",
        _WHISPERX_DEVICE,
        compute_type,
    )

    _WHISPERX_MODEL = whisperx.load_model(
        "medium",
        device=_WHISPERX_DEVICE,
        compute_type=compute_type,
    )

    # Align model will be loaded lazily for the detected language
    _ALIGN_MODEL = None
    _ALIGN_METADATA = None

    logger.info("WhisperX ASR model initialized successfully on %s", _WHISPERX_DEVICE)

def _ensure_align_model(language_code: str) -> None:
    """
    Ensure we have an alignment model for the given language code.
    Will lazy-load and cache it in globals.
    """
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
    """
    Run WhisperX on a wav file.

    Args:
      wav_path: path to 16k mono wav.
      lang: optional language code (e.g. "cs", "en"). If None, WhisperX will detect.
      use_alignment: if True, try to refine timestamps with WhisperX aligner.
                     if False, use raw ASR segments only.
      alignment_min_coverage: if alignment is enabled, require that the
          total time covered by aligned segments is at least this fraction
          of the original ASR coverage. Otherwise, fall back to non-aligned.

    Returns:
      full_text: concatenated segments text
      segments: List[(start_s, end_s, text, speaker_label)]
                speaker_label is "" for now.
    """

    _init_whisperx(lang_hint=lang)

    # whisperx.load_audio() shells out to ffmpeg; on some Windows setups ffmpeg
    # is not available on PATH. For our pipeline, inputs are already WAV, so we
    # can fall back to soundfile without losing functionality.
    try:
        audio = whisperx.load_audio(wav_path)
    except FileNotFoundError:
        import soundfile as sf

        logger.warning(
            "ffmpeg not found while loading %s; falling back to soundfile WAV loader",
            wav_path,
        )
        audio, _sr = sf.read(wav_path, dtype="float32")

    logger.info(
        "WhisperX: transcribing %s (lang=%s, use_alignment=%s)",
        wav_path, lang or "auto", use_alignment
    )

    # --- 1) ASR ---
    if lang:
        result = _WHISPERX_MODEL.transcribe(audio, batch_size=16, language=lang)
    else:
        result = _WHISPERX_MODEL.transcribe(audio, batch_size=16)

    detected_lang = result.get("language", lang or "unknown")
    logger.debug("WhisperX: detected language=%s", detected_lang)

    # Base segments from ASR
    asr_segments = result.get("segments", []) or []

    def from_asr_segments():
        segs_out: List[Tuple[float, float, str, str]] = []
        words_out: List[str] = []
        for seg in asr_segments:
            s0 = float(seg.get("start", 0.0))
            s1 = float(seg.get("end", 0.0))
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segs_out.append((s0, s1, text, ""))  # speaker="" for now
            words_out.append(text)
        full = " ".join(words_out).strip()
        logger.info(
            "WhisperX (no align): %d segments, total_text_len=%d",
            len(segs_out), len(full)
        )
        return full, segs_out

    # If alignment disabled → just return ASR segments.
    if not use_alignment:
        return from_asr_segments()

    # --- 2) Try alignment ---
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

        # Compute coverage for a simple safety check
        def total_duration(segs):
            return sum(
                max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
                for s in segs
            )

        asr_cov = total_duration(asr_segments)
        aln_cov = total_duration(aligned_segments)
        coverage_ratio = (aln_cov / asr_cov) if asr_cov > 0 else 1.0

        if asr_cov > 0 and coverage_ratio < alignment_min_coverage:
            logger.warning(
                "WhisperX alignment coverage too low: %.2f (ASR=%.2fs, aligned=%.2fs). "
                "Falling back to unaligned segments.",
                coverage_ratio, asr_cov, aln_cov,
            )
            return from_asr_segments()

        segs_out: List[Tuple[float, float, str, str]] = []
        words_out: List[str] = []

        for seg in aligned_segments:
            s0 = float(seg.get("start", 0.0))
            s1 = float(seg.get("end", 0.0))
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            segs_out.append((s0, s1, text, ""))  # speaker="" for now
            words_out.append(text)

        full = " ".join(words_out).strip()
        logger.info(
            "WhisperX (aligned): %d segments, total_text_len=%d, coverage_ratio=%.2f",
            len(segs_out), len(full), coverage_ratio
        )
        return full, segs_out

    except Exception as e:
        logger.warning(
            "WhisperX alignment failed (%s); falling back to unaligned segments.",
            e
        )
        return from_asr_segments()

def make_consumer() -> Consumer:
    logger.info("Creating Kafka consumer")
    return Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": GROUP_ID,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "max.partition.fetch.bytes": 5_000_000,
            "fetch.wait.max.ms": 50,
        }
    )

def make_producer() -> Producer:
    logger.info("Creating Kafka producer")
    return Producer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "client.id": "whisperx-async",
            "compression.type": "zstd",
            "linger.ms": 10,
            "batch.size": 131072,
        }
    )

def main():
    logger.info("Starting whisperx_worker")

    try:
        _init_whisperx()
    except Exception as e:
        logger.exception("Failed to initialize WhisperX at startup: %s", e)
        return

    c = make_consumer()
    p = make_producer()
    c.subscribe([TOPIC_AUDIO])

    buffers: Dict[str, Deque[np.ndarray]] = defaultdict(deque)
    buf_samples: Dict[str, int] = defaultdict(int)
    slice_index: Dict[str, int] = defaultdict(int)
    last_activity: Dict[str, float] = defaultdict(lambda: 0.0)
    session_lang: Dict[str, Optional[str]] = defaultdict(lambda: None)
    bff_origin_uri: Dict[str, Optional[str]] = defaultdict(lambda: None)

    try:
        while True:
            now = time.time()

            # --- 1) Idle session eviction + final partial flush ---
            idle_sessions = [
                sid for sid, ts in list(last_activity.items())
                if now - ts > SESSION_IDLE_SEC
            ]
            for sid in idle_sessions:
                logger.info(
                    "Session %s idle for %.1fs (threshold=%.1fs) → flushing & evicting",
                    sid,
                    now - last_activity[sid],
                    SESSION_IDLE_SEC,
                )
                lang_hint = session_lang.get(sid)
                _flush_session_partial(
                    sid,
                    lang_hint,
                    buffers,
                    buf_samples,
                    slice_index,
                    p,
                    session_lang,
                    bff_origin_uri,
                    tenant_id,
                )
                last_activity.pop(sid, None)
                session_lang.pop(sid, None)
                bff_origin_uri.pop(sid, None)
                tenant_id.pop(sid, None)

            # --- 2) Normal Kafka polling ---
            logger.debug("Waiting for message from Kafka")
            msg = c.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Kafka error: %s", msg.error())
                raise KafkaException(msg.error())

            key_bytes = msg.key()
            key = key_bytes.decode("utf-8") if key_bytes else ""
            logger.debug("Processing message with key=%s, offset=%i", key, msg.offset())

            audio = stream_pb2.AudioChunk()
            audio.ParseFromString(msg.value())

            session_id = audio.session_id
            lang = getattr(audio, "lang", "") or None
            bff_uri = getattr(audio, "bff_origin_uri", "") or None
            tenant = getattr(audio, "tenant_id", "") or None

            # Track lang hint & activity
            if lang:
                session_lang[session_id] = lang
            if bff_uri:
                bff_origin_uri[session_id] = bff_uri
            if tenant:
                tenant_id[session_id] = tenant
            last_activity[session_id] = now

            # Append to buffer
            arr = np.frombuffer(audio.pcm16_le, dtype="<i2")
            buffers[session_id].append(arr)
            buf_samples[session_id] += arr.size

            logger.debug(
                "Buffer size for session %s: %d samples",
                session_id,
                buf_samples[session_id],
            )

            # --- 3) Full-slice processing as before ---
            if buf_samples[session_id] >= int(SLICE_SECONDS * SR):
                need = int(SLICE_SECONDS * SR)
                parts: List[np.ndarray] = []
                while need > 0 and buffers[session_id]:
                    ch = buffers[session_id][0]
                    if ch.size <= need:
                        parts.append(buffers[session_id].popleft())
                        need -= ch.size
                        buf_samples[session_id] -= ch.size
                    else:
                        parts.append(ch[:need])
                        buffers[session_id][0] = ch[need:]
                        buf_samples[session_id] -= need
                        need = 0

                pcm = np.concatenate(parts).astype("<i2")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name

                x = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                sf.write(wav_path, x, SR, subtype="PCM_16")

                base_start = slice_index[session_id] * SLICE_SECONDS
                slice_index[session_id] += 1

                logger.info(
                    "Session %s: built %0.1fs slice → %s",
                    session_id,
                    SLICE_SECONDS,
                    wav_path,
                )

                text, segments = run_whisperx_diarized(
                    wav_path,
                    tenant=tenant_id.get(session_id),
                    lang=session_lang.get(session_id),
                    use_alignment=False,
                )
                logger.debug("Ran WhisperX inference, entire text is: %s", text)

                for (s0, s1, seg_text, speaker) in segments:
                    abs_start = base_start + s0
                    abs_end = base_start + s1
                    ev = stream_pb2.RefinedEvent(
                        session_id=session_id,
                        start_s=abs_start,
                        end_s=abs_end,
                        text=seg_text,
                        speaker=speaker,
                        supersedes_seq=[],
                        lang=session_lang.get(session_id),
                        bff_origin_uri=bff_origin_uri.get(session_id),
                        tenant_id=tenant_id.get(session_id),
                    )
                    logger.debug(
                        "Sending refined message start_s=%.1f, speaker=%s, text=%s",
                        abs_start,
                        speaker,
                        seg_text,
                    )
                    p.produce(
                        topic=TOPIC_REFINED,
                        key=session_id.encode("utf-8"),
                        value=ev.SerializeToString(),
                    )
                p.poll(0)

                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

            c.commit(msg, asynchronous=True)

    except KeyboardInterrupt:
        logger.info("Stopping whisperx_worker (KeyboardInterrupt)")
    finally:
        # Optional: flush *all* remaining sessions on shutdown
        for sid in list(buffers.keys()):
            lang_hint = session_lang.get(sid)
            _flush_session_partial(
                sid,
                lang_hint,
                buffers,
                buf_samples,
                slice_index,
                p,
                session_lang,
                bff_origin_uri,
                tenant_id,
            )
        try:
            c.close()
        except Exception:
            pass
        try:
            p.flush(2.0)
        except Exception:
            pass

if __name__ == "__main__":
    main()
