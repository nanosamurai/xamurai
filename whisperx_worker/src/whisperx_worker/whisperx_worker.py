import logging
import os
import tempfile
import threading
from collections import defaultdict, deque
from whisperx_worker.decoupled_runtime import run_decoupled
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING, Tuple, TypedDict

import numpy as np
import soundfile as sf
from confluent_kafka import Consumer, Producer, KafkaException, TopicPartition

from proto_gen import stream_pb2

import time

# NOTE: torch is not installed in lightweight unit-test environments.
# Keep this import optional so `import whisperx_worker.whisperx_worker` works
# even without torch.
try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from drsynth_common.otel_setup import setup_otel
from drsynth_common.otel_kafka import extracted_context_from_headers, with_current_trace_context
from drsynth_common.logging_setup import setup_logging
from drsynth_common.stream_controls import (
    parse_stream_controls_from_kafka_headers,
    parse_refinement_window_sec_from_kafka_headers,
)

try:
    from opentelemetry import trace
except Exception:  # pragma: no cover
    trace = None  # type: ignore[assignment]

KafkaHeader = tuple[str, Optional[bytes]]

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
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").strip().upper()
KAFKA_SSL_CA_LOCATION = os.getenv("KAFKA_SSL_CA_LOCATION", "").strip()
TOPIC_AUDIO = os.getenv("KAFKA_TOPIC_AUDIO", "audio.raw")
TOPIC_REFINED = os.getenv("KAFKA_TOPIC_REFINED", "transcripts.refined")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "whisperx-async")

SLICE_SECONDS = float(os.getenv("WHISPERX_SLICE_SECONDS", "60.0"))
SESSION_IDLE_SEC = float(os.getenv("WHISPERX_IDLE_SECONDS", "30.0"))
SR = 16000

_DECOUPLE_IO = os.getenv("WHISPERX_DECOUPLE_IO", "false").strip().lower() in ("1", "true", "yes", "y")
_COMMIT_AFTER_PRODUCE = os.getenv("WHISPERX_COMMIT_AFTER_PRODUCE", "true").strip().lower() in ("1", "true", "yes", "y")



def _should_evict_idle_session(
    session_id: str,
    *,
    now_s: float,
    last_activity_s: Dict[str, float],
    last_poll_s: float,
    idle_sec: float,
) -> bool:
    """Return True if a session is eligible for idle flush+eviction.

    Why this exists:
    - The worker historically ran Kafka polling + WhisperX inference in the same
      thread.
    - On slow inference (often CPU), the loop would not call `poll()` for longer
      than `idle_sec`.
    - That made the session appear "idle" based on wall clock and caused us to
      flush+evict mid-session, which reset `slice_index` and made refined timing
      restart from ~0.

    We therefore treat "idle" as:
    - no new audio observed for `idle_sec`, AND
    - the consumer has been polling recently enough (i.e. we are not currently
      behind because inference blocked the loop).
    """

    last = last_activity_s.get(session_id, 0.0)
    if (now_s - last) <= idle_sec:
        return False

    # If the Kafka poll loop was blocked for longer than the idle threshold,
    # we may simply be behind on consuming audio for this session.
    # Evicting now would drop per-session state (slice_index/buffers) and cause
    # refined timestamps to restart from 0.
    if (now_s - last_poll_s) > idle_sec:
        return False

    return True

last_activity: Dict[str, float] = defaultdict(lambda: 0.0)
session_lang: Dict[str, Optional[str]] = defaultdict(lambda: None)
bff_origin_uri: Dict[str, Optional[str]] = defaultdict(lambda: None)
tenant_id: Dict[str, Optional[str]] = defaultdict(lambda: None)

# Store per-session headers so idle flush uses same traceparent.
session_trace_headers: Dict[str, Optional[list[KafkaHeader]]] = defaultdict(lambda: None)

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


def _run_inference_and_publish(*, job: Dict[str, Any], producer: Producer) -> None:
    """Inference path for decoupled runtime (Kafka poll decoupled from GPU inference).

    NOTE: This function used to be inlined in the main loop. It exists so the
    `decoupled_runtime` module can call back into the same publish semantics.
    """

    session_id = str(job.get("session_id") or "")
    if not session_id:
        return

    pcm = job.get("pcm16")
    if not isinstance(pcm, np.ndarray) or pcm.size == 0:
        return

    base_start = float(job.get("base_start_s") or 0.0)
    window_sec = float(job.get("window_sec") or 0.0)
    lang_hint = job.get("lang") or None
    tenant = job.get("tenant_id") or None
    bff_uri = job.get("bff_origin_uri") or None
    trace_headers = job.get("trace_headers")
    flush_reason = str(job.get("flush_reason") or "slice")
    slice_index = int(job.get("slice_index") or 0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        x = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        sf.write(wav_path, x, SR, subtype="PCM_16")

        with extracted_context_from_headers(trace_headers):
            slice_span_cm = None
            if trace is not None:
                try:
                    tracer = trace.get_tracer("whisperx_worker")
                    slice_span_cm = tracer.start_as_current_span("whisperx.slice")
                    slice_span_cm.__enter__()
                    span = trace.get_current_span()
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("nanosamurai.session_id", session_id)
                        if tenant:
                            span.set_attribute("nanosamurai.tenant_id", tenant)
                        span.set_attribute("nanosamurai.slice_index", slice_index)
                        span.set_attribute("nanosamurai.slice_start_s", float(base_start))
                        span.set_attribute("nanosamurai.slice_duration_s", float(pcm.size) / float(SR))
                        span.set_attribute("nanosamurai.flush_reason", flush_reason)
                except Exception:
                    slice_span_cm = None

            try:
                text, segments = run_whisperx_diarized(
                    wav_path,
                    tenant=tenant,
                    lang=lang_hint,
                    use_alignment=False,
                )

                if trace is not None:
                    try:
                        span = trace.get_current_span()
                        if hasattr(span, "set_attribute"):
                            span.set_attribute("nanosamurai.segments_count", int(len(segments)))
                            span.set_attribute("nanosamurai.text_len", int(len(text or "")))
                    except Exception:
                        pass

                publish_span_cm = None
                if trace is not None:
                    try:
                        tracer = trace.get_tracer("whisperx_worker")
                        publish_span_cm = tracer.start_as_current_span("kafka.produce transcripts.refined")
                        publish_span_cm.__enter__()
                    except Exception:
                        publish_span_cm = None

                try:
                    # Publish ONE refined event per refinement window slice.
                    seg_msgs: List[stream_pb2.SessionTranscriptSegment] = []
                    for (s0, s1, seg_text, speaker) in segments:
                        seg_msgs.append(
                            stream_pb2.SessionTranscriptSegment(
                                start_s=float(base_start + s0),
                                end_s=float(base_start + s1),
                                text=str(seg_text or ""),
                                speaker=str(speaker or ""),
                                words=[],
                            )
                        )

                    # Window end is best-effort:
                    # - use the max segment end when segments exist
                    # - otherwise fall back to base_start + slice duration
                    if seg_msgs:
                        window_end_s = max(float(s.end_s) for s in seg_msgs)
                    else:
                        window_end_s = float(base_start) + (float(pcm.size) / float(SR))

                    window_start_s = float(base_start)
                    effective_window_sec = float(window_sec) if float(window_sec) > 0 else float(pcm.size) / float(SR)
                    full_text = " ".join([s.text for s in seg_msgs if (s.text or "").strip()]).strip()

                    ev = stream_pb2.RefinedEvent(
                        session_id=session_id,
                        # Legacy scalar fields filled for backwards compatibility.
                        start_s=window_start_s,
                        end_s=float(window_end_s),
                        text=full_text,
                        speaker="",
                        supersedes_seq=[],
                        lang=str(lang_hint or ""),
                        bff_origin_uri=str(bff_uri or ""),
                        tenant_id=str(tenant or ""),
                        window_sec=float(effective_window_sec),
                        slice_index=int(slice_index),
                        flush_reason=str(flush_reason),
                        segments=seg_msgs,
                        full_text=full_text,
                        created_at_ns=int(time.time_ns()),
                        refinement_model="whisperx",
                    )
                    producer.produce(
                        topic=TOPIC_REFINED,
                        key=session_id.encode("utf-8"),
                        value=ev.SerializeToString(),
                        headers=with_current_trace_context(),
                    )
                finally:
                    if publish_span_cm is not None:
                        try:
                            publish_span_cm.__exit__(None, None, None)
                        except Exception:
                            pass
                producer.poll(0)
            finally:
                if slice_span_cm is not None:
                    try:
                        slice_span_cm.__exit__(None, None, None)
                    except Exception:
                        pass
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


def _flush_session_partial(
    session_id: str,
    lang_hint: Optional[str],
    refinement_window_sec: float,
    buffers: Dict[str, Deque[np.ndarray]],
    buf_samples: Dict[str, int],
    slice_index: Dict[str, int],
    producer: Producer,
    session_lang: Dict[str, Optional[str]],
    bff_origin_uri: Dict[str, Optional[str]],
    tenant_id: Dict[str, Optional[str]],
    session_trace_headers: Dict[str, Optional[list[KafkaHeader]]],
) -> None:
    """Flush remaining buffered audio for a session."""

    if buf_samples[session_id] <= 0 or not buffers[session_id]:
        buffers.pop(session_id, None)
        buf_samples.pop(session_id, None)
        slice_index.pop(session_id, None)
        session_trace_headers.pop(session_id, None)
        return

    parts: List[np.ndarray] = []
    while buffers[session_id]:
        parts.append(buffers[session_id].popleft())
    buf_samples[session_id] = 0

    pcm = np.concatenate(parts).astype("<i2")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    x = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    sf.write(wav_path, x, SR, subtype="PCM_16")

    base_start = slice_index[session_id] * float(refinement_window_sec)
    slice_idx = int(slice_index[session_id])
    slice_index[session_id] += 1

    logger.info(
        "Idle flush for session %s: final slice dur=%.2fs → %s",
        session_id,
        len(pcm) / SR,
        wav_path,
    )

    # Re-attach trace context captured from kafka consume.
    # IMPORTANT: we also want refined publishing to have a *real* in-trace parent span,
    # otherwise downstream (samuraipersistor) spans will look "orphaned" in Grafana because
    # their parentSpanId will refer to the deterministic remote-parent seed.
    with extracted_context_from_headers(session_trace_headers.get(session_id)):
        slice_span_cm = None
        if trace is not None:
            tracer = trace.get_tracer("whisperx_worker")
            slice_span_cm = tracer.start_as_current_span("whisperx.slice")
            slice_span_cm.__enter__()

            try:
                span = trace.get_current_span()
                if hasattr(span, "set_attribute"):
                    span.set_attribute("nanosamurai.session_id", session_id)
                    if tenant_id.get(session_id):
                        span.set_attribute("nanosamurai.tenant_id", tenant_id.get(session_id))
                    span.set_attribute("nanosamurai.slice_index", int(slice_idx))
                    span.set_attribute("nanosamurai.slice_start_s", float(base_start))
                    span.set_attribute("nanosamurai.slice_duration_s", float(len(pcm) / SR))
                    span.set_attribute("nanosamurai.flush_reason", "idle")
            except Exception:
                pass

        try:
            text, segments = run_whisperx_diarized(
                wav_path,
                tenant=tenant_id.get(session_id),
                lang=lang_hint,
                use_alignment=False,
            )
            logger.debug("Idle WhisperX result (session=%s): %s", session_id, text)

            # Add summary attributes (best-effort)
            if trace is not None:
                try:
                    span = trace.get_current_span()
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("nanosamurai.segments_count", int(len(segments)))
                        span.set_attribute("nanosamurai.text_len", int(len(text or "")))
                except Exception:
                    pass

            publish_span_cm = None
            if trace is not None:
                try:
                    tracer = trace.get_tracer("whisperx_worker")
                    publish_span_cm = tracer.start_as_current_span("kafka.produce transcripts.refined")
                    publish_span_cm.__enter__()
                    span = trace.get_current_span()
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("messaging.system", "kafka")
                        span.set_attribute("messaging.destination", TOPIC_REFINED)
                        span.set_attribute("nanosamurai.session_id", session_id)
                except Exception:
                    publish_span_cm = None

            try:
                seg_msgs: List[stream_pb2.SessionTranscriptSegment] = []
                for (s0, s1, seg_text, speaker) in segments:
                    seg_msgs.append(
                        stream_pb2.SessionTranscriptSegment(
                            start_s=float(base_start + s0),
                            end_s=float(base_start + s1),
                            text=str(seg_text or ""),
                            speaker=str(speaker or ""),
                            words=[],
                        )
                    )

                if seg_msgs:
                    window_end_s = max(float(s.end_s) for s in seg_msgs)
                else:
                    window_end_s = float(base_start) + (float(len(pcm)) / float(SR))

                window_start_s = float(base_start)
                full_text = " ".join([s.text for s in seg_msgs if (s.text or "").strip()]).strip()

                ev = stream_pb2.RefinedEvent(
                    session_id=session_id,
                    start_s=window_start_s,
                    end_s=float(window_end_s),
                    text=full_text,
                    speaker="",
                    supersedes_seq=[],
                    lang=session_lang.get(session_id) or "",
                    bff_origin_uri=bff_origin_uri.get(session_id) or "",
                    tenant_id=tenant_id.get(session_id) or "",
                    window_sec=float(refinement_window_sec),
                    slice_index=int(slice_idx),
                    flush_reason="idle",
                    segments=seg_msgs,
                    full_text=full_text,
                    created_at_ns=int(time.time_ns()),
                    refinement_model="whisperx",
                )
                producer.produce(
                    topic=TOPIC_REFINED,
                    key=session_id.encode("utf-8"),
                    value=ev.SerializeToString(),
                    headers=with_current_trace_context(),
                )
            finally:
                if publish_span_cm is not None:
                    try:
                        publish_span_cm.__exit__(None, None, None)
                    except Exception:
                        pass
            producer.poll(0)
        finally:
            if slice_span_cm is not None:
                try:
                    slice_span_cm.__exit__(None, None, None)
                except Exception:
                    pass
            try:
                os.unlink(wav_path)
            except Exception:
                pass

    buffers.pop(session_id, None)
    buf_samples.pop(session_id, None)
    slice_index.pop(session_id, None)
    session_trace_headers.pop(session_id, None)


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


def make_consumer() -> Consumer:
    logger.info("Creating Kafka consumer")
    cfg = {
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": GROUP_ID,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "max.partition.fetch.bytes": 5_000_000,
        "fetch.wait.max.ms": 50,
    }
    if KAFKA_SECURITY_PROTOCOL and KAFKA_SECURITY_PROTOCOL != "PLAINTEXT":
        # librdkafka expects 'SSL' (uppercase)
        cfg["security.protocol"] = KAFKA_SECURITY_PROTOCOL
        if KAFKA_SSL_CA_LOCATION:
            cfg["ssl.ca.location"] = KAFKA_SSL_CA_LOCATION
    return Consumer(cfg)


def make_producer() -> Producer:
    logger.info("Creating Kafka producer")
    cfg = {
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "client.id": "whisperx-async",
        "compression.type": "zstd",
        "linger.ms": 10,
        "batch.size": 131072,
    }
    if KAFKA_SECURITY_PROTOCOL and KAFKA_SECURITY_PROTOCOL != "PLAINTEXT":
        cfg["security.protocol"] = KAFKA_SECURITY_PROTOCOL
        if KAFKA_SSL_CA_LOCATION:
            cfg["ssl.ca.location"] = KAFKA_SSL_CA_LOCATION
    return Producer(cfg)


def main():
    setup_logging(default_level="INFO")
    setup_otel(service_name=os.getenv("OTEL_SERVICE_NAME", "whisperx-worker"))

    logger.info("Starting whisperx_worker")

    if torch is None:
        logger.exception("whisperx_worker: torch not installed; cannot start")
        return

    if not torch.cuda.is_available():
        logger.warning(
            "whisperx_worker: CUDA not available; running on CPU (torch=%s torch.version.cuda=%s)",
            getattr(torch, "__version__", "unknown"),
            getattr(getattr(torch, "version", None), "cuda", None),
        )
    else:
        logger.info(
            "whisperx_worker: CUDA available; will use GPU (torch=%s torch.version.cuda=%s)",
            getattr(torch, "__version__", "unknown"),
            getattr(getattr(torch, "version", None), "cuda", None),
        )

    try:
        _init_whisperx()
    except Exception as e:
        logger.exception("Failed to initialize WhisperX at startup: %s", e)
        return

    c = make_consumer()
    p = make_producer()
    c.subscribe([TOPIC_AUDIO])

    if _DECOUPLE_IO:
        def _parse_audio_chunk(v: bytes) -> stream_pb2.AudioChunk:
            audio = stream_pb2.AudioChunk()
            audio.ParseFromString(v)
            return audio

        run_decoupled(
            consumer=c,
            topic_audio=TOPIC_AUDIO,
            slice_seconds=SLICE_SECONDS,
            sample_rate=SR,
            session_idle_sec=SESSION_IDLE_SEC,
            should_evict_idle_session=_should_evict_idle_session,
            parse_audio_chunk=_parse_audio_chunk,
            build_job=lambda **kw: {
                "session_id": kw["session_id"],
                "tenant_id": kw.get("tenant"),
                "lang": kw.get("lang"),
                "bff_origin_uri": kw.get("bff_uri"),
                "trace_headers": kw.get("trace_headers"),
                "pcm16": kw["pcm16"],
                "base_start_s": float(kw["base_start_s"]),
                "window_sec": float(kw.get("window_sec") or float(SLICE_SECONDS)),
                "slice_index": int(kw["slice_index"]),
                "flush_reason": kw.get("flush_reason", "slice"),
                "topic": kw.get("msg_topic", TOPIC_AUDIO),
                "partition": int(kw.get("msg_partition", 0)),
                "offset": int(kw.get("msg_offset", -1)),
            },
            run_inference_and_publish=lambda job: _run_inference_and_publish(job=job, producer=p),
        )
        return

    buffers: Dict[str, Deque[np.ndarray]] = defaultdict(deque)
    buf_samples: Dict[str, int] = defaultdict(int)
    slice_index: Dict[str, int] = defaultdict(int)
    session_refinement_window_sec: Dict[str, float] = defaultdict(lambda: float(SLICE_SECONDS))
    session_refinement_slice_samples: Dict[str, int] = defaultdict(lambda: int(round(float(SLICE_SECONDS) * SR)))
    last_activity: Dict[str, float] = defaultdict(lambda: 0.0)
    session_lang: Dict[str, Optional[str]] = defaultdict(lambda: None)
    bff_origin_uri: Dict[str, Optional[str]] = defaultdict(lambda: None)

    # Wall-clock time when we last attempted to poll Kafka.
    # Used to detect when WhisperX inference blocked the loop long enough that
    # wall-clock "idle" heuristics would be misleading.
    last_poll_s: float = time.time()

    try:
        while True:
            now = time.time()

            # --- 1) Idle session eviction + final partial flush ---
            idle_sessions = [
                sid
                for sid, _ts in list(last_activity.items())
                if _should_evict_idle_session(
                    sid,
                    now_s=now,
                    last_activity_s=last_activity,
                    last_poll_s=last_poll_s,
                    idle_sec=SESSION_IDLE_SEC,
                )
            ]
            for sid in idle_sessions:
                logger.info(
                    "Session %s idle for %.1fs (threshold=%.1fs) → flushing & evicting",
                    sid,
                    now - last_activity[sid],
                    SESSION_IDLE_SEC,
                )
                lang_hint = session_lang.get(sid)
                win_sec = float(session_refinement_window_sec.get(sid, float(SLICE_SECONDS)))
                _flush_session_partial(
                    sid,
                    lang_hint,
                    win_sec,
                    buffers,
                    buf_samples,
                    slice_index,
                    p,
                    session_lang,
                    bff_origin_uri,
                    tenant_id,
                    session_trace_headers,
                )
                last_activity.pop(sid, None)
                session_lang.pop(sid, None)
                bff_origin_uri.pop(sid, None)
                tenant_id.pop(sid, None)
                session_refinement_window_sec.pop(sid, None)
                session_refinement_slice_samples.pop(sid, None)

            # --- 2) Normal Kafka polling ---
            logger.debug("Waiting for message from Kafka")
            last_poll_s = time.time()
            msg = c.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Kafka error: %s", msg.error())
                raise KafkaException(msg.error())

            hdrs = msg.headers() or None
            controls = parse_stream_controls_from_kafka_headers(hdrs)
            if not controls.want_refined:
                # Skip refined processing entirely (saves GPU/CPU). We still commit
                # offsets so this consumer group keeps up.
                logger.debug(
                    "Skipping refined processing due to x-outputs (session unknown yet) topic=%s partition=%s offset=%s",
                    msg.topic(),
                    msg.partition(),
                    msg.offset(),
                )
                if _COMMIT_AFTER_PRODUCE:
                    c.commit(msg, asynchronous=False)
                else:
                    c.commit(msg, asynchronous=True)
                continue

            with extracted_context_from_headers(hdrs):
                try:
                    audio = stream_pb2.AudioChunk()
                    audio.ParseFromString(msg.value())

                    session_id = audio.session_id
                    lang = getattr(audio, "lang", "") or None
                    bff_uri = getattr(audio, "bff_origin_uri", "") or None
                    tenant = getattr(audio, "tenant_id", "") or None

                    # Keep latest kafka headers for end-to-end propagation.
                    session_trace_headers[session_id] = msg.headers() or session_trace_headers.get(session_id)

                    if lang:
                        session_lang[session_id] = lang
                    if bff_uri:
                        bff_origin_uri[session_id] = bff_uri
                    if tenant:
                        tenant_id[session_id] = tenant
                    last_activity[session_id] = now

                    # Per-session refinement window (slice duration) override.
                    # Expected Kafka header (samuraibff): x-refinement-window-sec
                    if session_id not in session_refinement_window_sec:
                        win_sec = parse_refinement_window_sec_from_kafka_headers(
                            hdrs,
                            default_sec=SLICE_SECONDS,
                        )
                        session_refinement_window_sec[session_id] = float(win_sec)
                        session_refinement_slice_samples[session_id] = int(round(float(win_sec) * SR))
                    else:
                        win_sec = float(session_refinement_window_sec.get(session_id, float(SLICE_SECONDS)))

                    refinement_window_sec = win_sec
                    refinement_slice_samples = int(session_refinement_slice_samples.get(session_id, int(round(win_sec * SR))))

                    arr = np.frombuffer(audio.pcm16_le, dtype="<i2")
                    buffers[session_id].append(arr)
                    buf_samples[session_id] += arr.size

                    logger.debug(
                        "Buffer size for session %s: %d samples",
                        session_id,
                        buf_samples[session_id],
                    )

                    if buf_samples[session_id] >= refinement_slice_samples:
                        need = refinement_slice_samples
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

                        base_start = slice_index[session_id] * refinement_window_sec
                        slice_idx = int(slice_index[session_id])
                        slice_index[session_id] += 1

                        logger.info(
                            "Session %s: built %0.1fs slice → %s",
                            session_id,
                            refinement_window_sec,
                            wav_path,
                        )

                        # Only span per slice (not per chunk).
                        slice_span_cm = None
                        if trace is not None:
                            tracer = trace.get_tracer("whisperx_worker")
                            slice_span_cm = tracer.start_as_current_span("whisperx.slice")
                            slice_span_cm.__enter__()
                            try:
                                span = trace.get_current_span()
                                if hasattr(span, "set_attribute"):
                                    span.set_attribute("nanosamurai.session_id", session_id)
                                    if tenant_id.get(session_id):
                                        span.set_attribute("nanosamurai.tenant_id", tenant_id.get(session_id))
                                    span.set_attribute("nanosamurai.slice_index", int(slice_idx))
                                    span.set_attribute("nanosamurai.slice_start_s", float(base_start))
                                    span.set_attribute("nanosamurai.slice_duration_s", float(refinement_window_sec))
                                    span.set_attribute("nanosamurai.flush_reason", "slice")
                            except Exception:
                                pass

                        try:
                            text, segments = run_whisperx_diarized(
                                wav_path,
                                tenant=tenant_id.get(session_id),
                                lang=session_lang.get(session_id),
                                use_alignment=False,
                            )
                            logger.debug("Ran WhisperX inference, entire text is: %s", text)

                            if trace is not None:
                                try:
                                    span = trace.get_current_span()
                                    if hasattr(span, "set_attribute"):
                                        span.set_attribute("nanosamurai.segments_count", int(len(segments)))
                                        span.set_attribute("nanosamurai.text_len", int(len(text or "")))
                                except Exception:
                                    pass

                            publish_span_cm = None
                            if trace is not None:
                                try:
                                    tracer = trace.get_tracer("whisperx_worker")
                                    publish_span_cm = tracer.start_as_current_span("kafka.produce transcripts.refined")
                                    publish_span_cm.__enter__()
                                    span = trace.get_current_span()
                                    if hasattr(span, "set_attribute"):
                                        span.set_attribute("messaging.system", "kafka")
                                        span.set_attribute("messaging.destination", TOPIC_REFINED)
                                        span.set_attribute("nanosamurai.session_id", session_id)
                                except Exception:
                                    publish_span_cm = None

                            try:
                                seg_msgs: List[stream_pb2.SessionTranscriptSegment] = []
                                for (s0, s1, seg_text, speaker) in segments:
                                    seg_msgs.append(
                                        stream_pb2.SessionTranscriptSegment(
                                            start_s=float(base_start + s0),
                                            end_s=float(base_start + s1),
                                            text=str(seg_text or ""),
                                            speaker=str(speaker or ""),
                                            words=[],
                                        )
                                    )

                                if seg_msgs:
                                    window_end_s = max(float(s.end_s) for s in seg_msgs)
                                else:
                                    window_end_s = float(base_start) + (float(pcm.size) / float(SR))

                                window_start_s = float(base_start)
                                full_text = " ".join([s.text for s in seg_msgs if (s.text or "").strip()]).strip()

                                ev = stream_pb2.RefinedEvent(
                                    session_id=session_id,
                                    start_s=window_start_s,
                                    end_s=float(window_end_s),
                                    text=full_text,
                                    speaker="",
                                    supersedes_seq=[],
                                    lang=session_lang.get(session_id) or "",
                                    bff_origin_uri=bff_origin_uri.get(session_id) or "",
                                    tenant_id=tenant_id.get(session_id) or "",
                                    window_sec=float(refinement_window_sec),
                                    slice_index=int(slice_index[session_id] - 1),
                                    flush_reason="slice",
                                    segments=seg_msgs,
                                    full_text=full_text,
                                    created_at_ns=int(time.time_ns()),
                                    refinement_model="whisperx",
                                )
                                logger.debug(
                                    "Sending refined window message slice=%d start_s=%.1f segments=%d text_len=%d",
                                    int(slice_index[session_id] - 1),
                                    window_start_s,
                                    len(seg_msgs),
                                    len(full_text),
                                )
                                p.produce(
                                    topic=TOPIC_REFINED,
                                    key=session_id.encode("utf-8"),
                                    value=ev.SerializeToString(),
                                    headers=with_current_trace_context(),
                                )
                            finally:
                                if publish_span_cm is not None:
                                    try:
                                        publish_span_cm.__exit__(None, None, None)
                                    except Exception:
                                        pass
                            p.poll(0)
                        finally:
                            if slice_span_cm is not None:
                                try:
                                    slice_span_cm.__exit__(None, None, None)
                                except Exception:
                                    pass

                        try:
                            os.unlink(wav_path)
                        except Exception:
                            pass

                    if _COMMIT_AFTER_PRODUCE:
                        c.commit(msg, asynchronous=False)
                    else:
                        c.commit(msg, asynchronous=True)
                finally:
                    pass

    except KeyboardInterrupt:
        logger.info("Stopping whisperx_worker (KeyboardInterrupt)")
    finally:
        for sid in list(buffers.keys()):
            lang_hint = session_lang.get(sid)
            win_sec = float(session_refinement_window_sec.get(sid, float(SLICE_SECONDS)))
            _flush_session_partial(
                sid,
                lang_hint,
                win_sec,
                buffers,
                buf_samples,
                slice_index,
                p,
                session_lang,
                bff_origin_uri,
                tenant_id,
                session_trace_headers,
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
