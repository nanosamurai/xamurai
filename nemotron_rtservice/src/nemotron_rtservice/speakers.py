"""Native word-to-turn conversion and bounded, optional S3 speaker identification."""
from __future__ import annotations

import io
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np

from drsynth_common.enrollment.models import SpeakerManifest
from nemotron_rtservice.native import SpeakerWord, _verified_model_path


logger = logging.getLogger(__name__)
EMBEDDING_MODEL_ID = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
EMBEDDING_MODEL_REVISION = "f0c48c298fd835726c27956a5d617bad7115627e"
EMBEDDING_MODEL_FILENAME = "voxceleb_resnet34_LM.onnx"
EMBEDDING_MODEL_DIGEST = "sha256:7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
SAMPLE_RATE = 16_000
MIN_SPEECH_S = 1.5
MAX_EMBEDDING_S = 10
MAX_SAMPLE_BYTES = 4 * 1024 * 1024
MAX_SPEAKERS = 256
MAX_SAMPLES = 5
LOOKUP_BUDGET_S = 15


@dataclass(frozen=True)
class SpeakerTurn:
    text: str
    start_s: float
    end_s: float
    speaker: int


def speaker_turns(text: str, words: tuple[SpeakerWord, ...], start_s: float,
                  end_s: float) -> tuple[SpeakerTurn, ...]:
    """Preserve every transcript character; unusable native words mean fallback."""
    if not words:
        return ()
    positions = []
    cursor = 0
    previous_start = start_s
    for word in words:
        token = word.text.strip()
        position = text.find(token, cursor) if token else -1
        if (position < 0 or text[cursor:position].strip()
                or not math.isfinite(word.start_s) or not math.isfinite(word.end_s)
                or word.start_s < previous_start - 0.02
                or word.start_s > end_s or word.end_s < word.start_s
                or not 0 <= word.speaker <= 4):
            return ()
        positions.append(position)
        cursor = position + len(token)
        previous_start = word.start_s
    if text[cursor:].strip():
        return ()
    turns = []
    first = 0
    for index in range(1, len(words) + 1):
        if index < len(words) and words[index].speaker == words[first].speaker:
            continue
        begin = 0 if first == 0 else positions[first]
        end = positions[index] if index < len(words) else len(text)
        t0 = max(start_s, words[first].start_s)
        # RNNT word ends include decoder/punctuation lookahead and may
        # exceed the final's consumed-audio boundary. Keep valid onset tags
        # and clip their ends, instead of losing every speaker in the final.
        t1 = min(end_s, max(word.end_s for word in words[first:index]))
        if index < len(words):
            # RNNT punctuation can stretch a word's end into the next turn.
            # Match the native onset-based attribution and avoid embedding
            # the next speaker's audio as evidence for this one.
            t1 = min(t1, words[index].start_s)
        if t1 <= t0:
            return ()
        turns.append(SpeakerTurn(text[begin:end], t0, t1, words[first].speaker))
        first = index
    return tuple(turns)


class SpeakerEmbedding:
    """Fixed WeSpeaker ONNX encoder on CPU; no PyTorch or GPU reservation."""

    def __init__(self):
        import onnxruntime as ort

        path = _verified_model_path(
            EMBEDDING_MODEL_ID, EMBEDDING_MODEL_FILENAME,
            EMBEDDING_MODEL_REVISION, EMBEDDING_MODEL_DIGEST,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self._session.disable_fallback()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)[:MAX_EMBEDDING_S * SAMPLE_RATE]
        if len(audio) < MIN_SPEECH_S * SAMPLE_RATE or not np.isfinite(audio).all():
            raise ValueError("insufficient valid speaker audio")
        options = knf.FbankOptions()
        options.frame_opts.dither = 0
        options.frame_opts.samp_freq = SAMPLE_RATE
        options.frame_opts.window_type = "hamming"
        options.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(options)
        fbank.accept_waveform(SAMPLE_RATE, audio * 32768.0)
        fbank.input_finished()
        features = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        features -= features.mean(axis=0, keepdims=True)
        embedding = self._session.run(None, {self._session.get_inputs()[0].name: features[None]})[0]
        return _normalize(embedding)


def _normalize(embedding) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not vector.size or not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("invalid speaker embedding")
    return vector / norm


def _read_object(client, bucket: str, key: str, limit: int) -> bytes:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        if response.get("ContentLength", 0) > limit:
            raise ValueError("enrollment object too large")
        data = body.read(limit + 1)
        if len(data) > limit:
            raise ValueError("enrollment object too large")
        return data
    finally:
        body.close()


class EnrolledSpeakers:
    """Replica-local TTL/LRU gallery; tenant scope is enforced before every read."""

    def __init__(self, client, bucket: str, prefix: str, embed, *, threshold=0.65,
                 margin=0.1, ttl_s=300.0, max_tenants=32):
        if not bucket or not 0 <= threshold <= 1 or not 0 <= margin <= 1:
            raise ValueError("invalid enrollment bucket or matching threshold")
        self._client, self._bucket = client, bucket
        self._prefix, self._embed = prefix.strip("/"), embed
        self._threshold, self._margin = threshold, margin
        self._ttl_s, self._max_tenants = ttl_s, max_tenants
        self._cache = OrderedDict()
        # Serialize cache refresh and CPU embedding work within a replica.
        self._lock = threading.Lock()

    def _gallery(self, tenant: str, check_budget):
        now = time.monotonic()
        if tenant in self._cache:
            loaded, gallery = self._cache[tenant]
            if now - loaded < self._ttl_s:
                self._cache.move_to_end(tenant)
                return gallery
        root = "/".join(p for p in (self._prefix, tenant, "speakers") if p) + "/"
        keys = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=root, Delimiter="/"):
            check_budget()
            for item in page.get("CommonPrefixes", []):
                speaker = item["Prefix"][len(root):].rstrip("/")
                if not IDENTIFIER.fullmatch(speaker):
                    continue
                keys.append((speaker, root + speaker + "/"))
                if len(keys) > MAX_SPEAKERS:
                    raise ValueError("tenant enrollment exceeds resource limit")
        gallery = {}
        duplicate_labels = set()
        for speaker, folder in keys:
            check_budget()
            try:
                manifest = SpeakerManifest.loads(
                    _read_object(self._client, self._bucket, folder + "speaker.json", 65536).decode("utf-8")
                )
                manifest.validate()
                label = manifest.label.strip()
                if (manifest.speaker_id != speaker or not label or len(label) > 128
                        or any(ord(c) < 32 for c in label) or label.startswith("SPEAKER_")
                        or len(manifest.samples) > MAX_SAMPLES):
                    raise ValueError("invalid enrollment manifest")
                embeddings = []
                for sample in manifest.samples:
                    check_budget()
                    url = urlsplit(sample.url)
                    key = url.path.lstrip("/")
                    if (url.scheme != "s3" or url.netloc != self._bucket
                            or url.query or url.fragment or not key.startswith(folder + "samples/")
                            or any(p in (".", "..", "") for p in key.split("/"))):
                        raise ValueError("enrollment sample is outside its speaker scope")
                    data = _read_object(self._client, self._bucket, key, MAX_SAMPLE_BYTES)
                    embeddings.append(self._embed(self._decode_sample(data)))
                vector = _normalize(np.mean(embeddings, axis=0))
                if label in gallery or label in duplicate_labels:
                    gallery.pop(label, None)
                    duplicate_labels.add(label)
                else:
                    gallery[label] = vector
            except TimeoutError:
                raise
            except Exception as exc:
                logger.warning("Enrollment speaker unavailable error_type=%s", type(exc).__name__)
        self._cache[tenant] = (time.monotonic(), gallery)
        self._cache.move_to_end(tenant)
        while len(self._cache) > self._max_tenants:
            self._cache.popitem(last=False)
        return gallery

    @staticmethod
    def _decode_sample(data):
        import soundfile as sf
        from scipy.signal import resample_poly

        with sf.SoundFile(io.BytesIO(data)) as source:
            if (source.format != "WAV" or not 8000 <= source.samplerate <= 48000
                    or source.channels > 2 or source.frames > source.samplerate * 30):
                raise ValueError("enrollment sample must be a bounded WAV")
            rate = source.samplerate
            audio = source.read(frames=MAX_EMBEDDING_S * rate, dtype="float32", always_2d=True).mean(axis=1)
        if rate != SAMPLE_RATE:
            divisor = math.gcd(rate, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
        return audio

    def identify(self, tenant: str, audio: np.ndarray, *, is_active=lambda: True) -> str:
        if not IDENTIFIER.fullmatch(tenant) or len(audio) < MIN_SPEECH_S * SAMPLE_RATE:
            return ""
        deadline = time.monotonic() + LOOKUP_BUDGET_S

        def check_budget():
            if not is_active() or time.monotonic() >= deadline:
                raise TimeoutError("enrollment lookup stopped")

        try:
            if not self._lock.acquire(timeout=LOOKUP_BUDGET_S):
                return ""
            try:
                check_budget()
                gallery = self._gallery(tenant, check_budget)
                if not gallery:
                    return ""
                check_budget()
                embedding = _normalize(self._embed(audio))
                ranked = sorted(((float(embedding @ reference), label)
                                 for label, reference in gallery.items()), reverse=True)
                best, label = ranked[0]
                runner_up = ranked[1][0] if len(ranked) > 1 else -1.0
                return label if best >= self._threshold and best - runner_up >= self._margin else ""
            finally:
                self._lock.release()
        except Exception as exc:
            logger.warning("Enrollment matching unavailable error_type=%s", type(exc).__name__)
            return ""


def enrollment_from_env():
    backend = os.getenv("ENROLL_BACKEND", "disabled").strip().lower()
    if backend in ("none", "disabled"):
        return None
    if backend != "s3_manifest":
        raise ValueError("Nemotron ENROLL_BACKEND must be disabled or s3_manifest")
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3", endpoint_url=os.getenv("ENROLL_S3_ENDPOINT") or None,
        region_name=os.getenv("ENROLL_S3_REGION") or None,
        aws_access_key_id=os.getenv("ENROLL_S3_ACCESS_KEY") or None,
        aws_secret_access_key=os.getenv("ENROLL_S3_SECRET_KEY") or None,
        config=Config(connect_timeout=2, read_timeout=3, retries={"max_attempts": 1},
                      s3={"addressing_style": "path"}),
    )
    return EnrolledSpeakers(
        client, os.getenv("ENROLL_S3_BUCKET", ""), os.getenv("ENROLL_S3_PREFIX", "enrollment"),
        SpeakerEmbedding(), threshold=float(os.getenv("ENROLL_SIM_THRESHOLD", "0.65")),
        margin=float(os.getenv("NEMOTRON_ENROLL_MATCH_MARGIN", "0.1")),
    )
