"""Stream a WAV to rtservice and print AsrEvents in receive order.

Usage (Windows):

  py utilities/rtservice_stream_wav_print_events.py \
    --wav tests/data/test_cs.wav \
    --addr localhost:50052 \
    --session-id demo-1 \
    --lang cs \
    --chunk-samples 2048 \
    --metadata x-rt-emit-every-sec=0.7 x-rt-window-sec=5.0 x-rt-overlap-sec=0.5

Notes:
- This is a debugging utility intended for local validation.
- It prints events exactly as received on the gRPC stream.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import grpc
import numpy as np
import soundfile as sf

# Ensure repo root is on sys.path so `proto_gen` import works when running this file directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from proto_gen import stream_pb2, stream_pb2_grpc


SR = 16000


def _load_mono(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if int(sr) != SR:
        raise SystemExit(f"Expected {SR} Hz WAV for this utility, got sr={sr}")
    return x


def _chunks(
    *,
    audio: np.ndarray,
    session_id: str,
    lang: str,
    tenant_id: str,
    chunk_samples: int,
) -> Iterable[stream_pb2.AudioChunk]:
    pcm16 = (audio * 32768.0).clip(-32768, 32767).astype("<i2")
    seq = 0
    for start in range(0, pcm16.size, chunk_samples):
        end = min(start + chunk_samples, pcm16.size)
        frame = pcm16[start:end]
        seq += 1
        yield stream_pb2.AudioChunk(
            session_id=session_id,
            seq=seq,
            t0_ns=0,
            sample_rate=SR,
            pcm16_le=frame.tobytes(),
            lang=lang or "",
            tenant_id=tenant_id,
        )

    # Flush chunk
    yield stream_pb2.AudioChunk(
        session_id=session_id,
        seq=seq + 1,
        t0_ns=0,
        sample_rate=SR,
        pcm16_le=b"",
        lang=lang or "",
        tenant_id=tenant_id,
    )


def _asr_type_name(t: int) -> str:
    if t == stream_pb2.PARTIAL:
        return "PARTIAL"
    if t == stream_pb2.FINAL:
        return "FINAL"
    return str(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="localhost:50052")
    ap.add_argument("--wav", default=str(Path("tests/data/test_cs.wav")))
    ap.add_argument("--session-id", default="rtservice-debug")
    ap.add_argument("--tenant-id", default="t-debug")
    ap.add_argument("--lang", default="cs")
    ap.add_argument("--chunk-samples", type=int, default=2048)
    ap.add_argument(
        "--metadata",
        nargs="*",
        default=[],
        help="Optional gRPC metadata headers as k=v, e.g. x-rt-emit-every-sec=0.7",
    )
    args = ap.parse_args()

    wav_path = Path(args.wav)
    audio = _load_mono(wav_path)
    dur = audio.size / SR

    md = []
    for kv in args.metadata:
        if "=" not in kv:
            raise SystemExit(f"Bad --metadata entry (expected k=v): {kv!r}")
        k, v = kv.split("=", 1)
        md.append((k, v))

    # Print headers in UTF-8 (stdout may be cp1252).
    sys.stdout.buffer.write(
        (f"[client] addr={args.addr} wav={wav_path} dur={dur:.2f}s chunk_samples={args.chunk_samples}\n").encode(
            "utf-8", errors="replace"
        )
    )
    if md:
        sys.stdout.buffer.write((f"[client] metadata={md}\n").encode("utf-8", errors="replace"))
    sys.stdout.flush()

    t0 = time.time()
    with grpc.insecure_channel(args.addr) as channel:
        stub = stream_pb2_grpc.RealtimeASRStub(channel)
        stream = stub.Stream(
            _chunks(
                audio=audio,
                session_id=args.session_id,
                lang=args.lang,
                tenant_id=args.tenant_id,
                chunk_samples=args.chunk_samples,
            ),
            metadata=md or None,
        )

        i = 0
        for ev in stream:
            i += 1
            dt = time.time() - t0
            typ = _asr_type_name(ev.type)
            text = (ev.text or "").replace("\n", " ").strip()
            if len(text) > 160:
                text = text[:160] + "…"
            line = (
                f"[{i:03d} +{dt:7.3f}s] {typ:7s} session={ev.session_id!r} "
                f"[{ev.start_s:7.2f}, {ev.end_s:7.2f}] "
                f"speaker={ev.speaker!r} lang={ev.lang!r} text_len={len(ev.text)} text={text!r}"
            )
            # Windows terminals can be cp1252 by default; be robust.
            # Write UTF-8 bytes directly.
            sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
