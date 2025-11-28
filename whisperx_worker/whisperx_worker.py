import logging
import os
import tempfile
from collections import defaultdict, deque
from typing import Deque, Tuple, List, Dict, Optional

import numpy as np
import soundfile as sf
from confluent_kafka import Consumer, Producer, KafkaException

import stream_pb2

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_AUDIO = os.getenv("KAFKA_TOPIC_AUDIO", "audio.raw")
TOPIC_REFINED = os.getenv("KAFKA_TOPIC_REFINED", "transcripts.refined")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "whisperx-async")

SLICE_SECONDS = float(os.getenv("WHISPERX_SLICE_SECONDS", "60.0"))
SR = 16000

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


def _init_whisperx(lang_hint: Optional[str] = None) -> None:
    """
    Global initialization of WhisperX.

    Called once at startup from main() so the first slice does not pay
    the full model download/compile cost.

    lang_hint is currently unused, but kept for future extension.
    """
    global _WHISPERX_MODEL, _ALIGN_MODEL, _ALIGN_METADATA, _WHISPERX_DEVICE

    if _WHISPERX_MODEL is not None:
        logger.debug("WhisperX model already initialized, skipping.")
        return

    import torch
    import whisperx

    # Allow overriding device via env if you ever want CPU-only for tests
    env_device = os.getenv("WHISPERX_DEVICE", "").lower().strip()
    if env_device in ("cuda", "gpu"):
        _WHISPERX_DEVICE = "cuda"
    elif env_device in ("cpu",):
        _WHISPERX_DEVICE = "cpu"
    else:
        _WHISPERX_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute type: float16 on GPU, int8 on CPU (you can tweak via env)
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

    # We will load the alignment model lazily per language (first segment)
    _ALIGN_MODEL = None
    _ALIGN_METADATA = None

    logger.info("WhisperX ASR model initialized successfully on %s", _WHISPERX_DEVICE)


def _ensure_align_model(language_code: str) -> None:
    """
    Ensure we have an alignment model for the given language code.
    Will lazy-load and cache it in globals.
    """
    global _ALIGN_MODEL, _ALIGN_METADATA, _WHISPERX_DEVICE

    import whisperx

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
    lang: Optional[str] = None
) -> Tuple[str, List[Tuple[float, float, str, str]]]:
    """
    Run WhisperX on a wav file.

    Returns:
      - full_text: concatenated text over all segments
      - segments: list of (start_s, end_s, text, speaker_label)

    For now, speaker_label is "" (no offline diarization).
    """
    import whisperx

    if _WHISPERX_MODEL is None:
        # Should not happen because we init at startup, but keep a guard.
        logger.warning("WhisperX model not initialized yet; initializing now.")
        _init_whisperx(lang_hint=lang)

    # Load audio
    audio = whisperx.load_audio(wav_path)
    logger.info("WhisperX: transcribing %s (lang=%s)", wav_path, lang or "auto")

    # Transcribe using WhisperX wrapper
    if lang:
        result = _WHISPERX_MODEL.transcribe(audio, batch_size=16, language=lang)
    else:
        result = _WHISPERX_MODEL.transcribe(audio, batch_size=16)

    detected_lang = result.get("language", lang or "unknown")
    logger.debug("WhisperX: detected language=%s", detected_lang)

    # Alignment
    _ensure_align_model(detected_lang)
    aligned = whisperx.align(
        result["segments"],
        _ALIGN_MODEL,
        _ALIGN_METADATA,
        audio,
        _WHISPERX_DEVICE,
        return_char_alignments=False,
    )

    segments_out: List[Tuple[float, float, str, str]] = []
    words_out: List[str] = []

    for seg in aligned["segments"]:
        s0 = float(seg["start"])
        s1 = float(seg["end"])
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments_out.append((s0, s1, text, ""))  # speaker="" for now
        words_out.append(text)

    full_text = " ".join(words_out).strip()
    logger.info(
        "WhisperX: got %d segments, total_text_len=%d",
        len(segments_out),
        len(full_text),
    )

    return full_text, segments_out


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

    # ---------- Initialize WhisperX at startup ----------
    try:
        _init_whisperx()
    except Exception as e:
        logger.exception("Failed to initialize WhisperX at startup: %s", e)
        return
    # ---------------------------------------------------

    c = make_consumer()
    p = make_producer()
    c.subscribe([TOPIC_AUDIO])

    # Per-session rolling PCM16 mono buffer
    buffers: Dict[str, Deque[np.ndarray]] = defaultdict(deque)
    buf_samples: Dict[str, int] = defaultdict(int)

    # Per-session slice index → used to compute absolute time of each 60s slice
    slice_index: Dict[str, int] = defaultdict(int)

    try:
        while True:
            logger.debug("Waiting for message from Kafka")
            msg = c.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Kafka error: %s", msg.error())
                raise KafkaException(msg.error())

            key_bytes = msg.key()
            key = key_bytes.decode("utf-8") if key_bytes else ""
            logger.debug(
                "Processing message with key=%s, offset=%i",
                key,
                msg.offset(),
            )

            audio = stream_pb2.AudioChunk()
            audio.ParseFromString(msg.value())

            session_id = audio.session_id
            lang = getattr(audio, "lang", "") or None

            # Append to session buffer
            arr = np.frombuffer(audio.pcm16_le, dtype="<i2")
            buffers[session_id].append(arr)
            buf_samples[session_id] += arr.size

            logger.debug(
                "Buffer size for session %s: %d samples",
                session_id,
                buf_samples[session_id],
            )

            # If we have ≥ SLICE_SECONDS, dump to WAV and process
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
                # Write temp WAV
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav_path = tmp.name

                x = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                sf.write(wav_path, x, SR, subtype="PCM_16")

                logger.info(
                    "Session %s: built %0.1fs slice → %s",
                    session_id,
                    SLICE_SECONDS,
                    wav_path,
                )

                # WhisperX inference
                text, segments = run_whisperx(wav_path, lang=lang)
                logger.debug(
                    "Ran WhisperX inference, entire text is: %s",
                    text,
                )

                # Compute base offset for this slice
                base_start = slice_index[session_id] * SLICE_SECONDS
                slice_index[session_id] += 1

                # Publish RefinedEvent(s)
                for (s0, s1, seg_text, speaker) in segments:
                    abs_start = base_start + s0
                    abs_end = base_start + s1
                    ev = stream_pb2.RefinedEvent(
                        session_id=session_id,
                        start_s=abs_start,
                        end_s=abs_end,
                        text=seg_text,
                        speaker=speaker,
                        # later you can fill supersedes_seq with real-time seq ids
                        supersedes_seq=[],
                    )
                    logger.debug(
                        "Sending a refined message with start_s:%0.1f, speaker:%s ; text:%s ",
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

            # Commit offset (you can switch to batched commits)
            c.commit(msg, asynchronous=True)

    except KeyboardInterrupt:
        logger.info("Stopping whisperx_worker (KeyboardInterrupt)")
    finally:
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
