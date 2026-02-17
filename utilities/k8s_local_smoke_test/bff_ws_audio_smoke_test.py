"""DEPRECATED: use tiered smoke tests in this directory.

This script is kept for backward compatibility. Prefer:
- tier1_bff_connectivity.py
- tier2_realtime_asr.py
- tier3_kafka_audio_raw.py (optional)
- tier4_async_pipeline.py (optional)

---

Local Kubernetes smoke test (BFF WebSocket audio path).

Goal
----
Validate the local k8s wiring end-to-end without requiring browser auth:

1) Create a session via BFF: POST /api/sessions
2) Connect to WS events: /ws/events?session_id=...
3) Connect to WS audio: /ws/audio?session_id=...&lang=...&sample_rate=16000
4) Stream a short WAV (converted to PCM16LE) as binary frames
5) Assert we receive at least one event message on /ws/events

This primarily tests:
- BFF HTTP+WS is reachable
- rtservice gRPC is reachable from BFF (BFF should emit status/asr events)
- Kafka is reachable from BFF (BFF publishes AudioChunk to audio.raw)
- overall cluster networking is sane

It does NOT attempt to verify persistence in Postgres or refined/final transcripts,
just the realtime ingestion path.

Usage
-----
1) Port-forward BFF:
   kubectl port-forward svc/nanosamurai-stack-bff 8000:8000

2) Install deps (venv recommended):
   python -m venv .venv
   .venv\\Scripts\\pip install -r utilities/k8s_local_smoke_test/requirements.txt

3) Run:
   .venv\\Scripts\\python utilities/k8s_local_smoke_test/bff_ws_audio_smoke_test.py --wav tests/data/test_cs.wav

Exit codes
----------
0  success
1  failure
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import urllib.parse
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
import websocket  # websocket-client

DEFAULT_BASE_URL = os.environ.get("BFF_BASE_URL", "http://localhost:8000")


@dataclass(frozen=True)
class WavPcm16:
    sample_rate: int
    pcm16le: bytes


def _read_wav_as_pcm16le(path: str, target_sr: int = 16000) -> WavPcm16:
    """Read a WAV file and return raw PCM16LE bytes.

    For smoke tests we keep this intentionally strict: require 16-bit PCM.
    If sample rate differs from target, we do a simple linear resample.
    """

    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        comptype = wf.getcomptype()

        if comptype != "NONE":
            raise ValueError(f"Unsupported WAV compression: {comptype}")
        if sampwidth != 2:
            raise ValueError(f"Expected 16-bit PCM WAV (sampwidth=2), got sampwidth={sampwidth}")

        raw = wf.readframes(nframes)

    # Convert to mono int16
    audio_i16 = np.frombuffer(raw, dtype="<i2")
    if nch > 1:
        audio_i16 = audio_i16.reshape(-1, nch).mean(axis=1).astype(np.int16)

    if sr != target_sr:
        # Linear resample (good enough for smoke tests)
        x = audio_i16.astype(np.float32)
        duration = x.shape[0] / sr
        t_old = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
        n_new = int(round(duration * target_sr))
        t_new = np.linspace(0.0, duration, num=n_new, endpoint=False)
        x_new = np.interp(t_new, t_old, x).astype(np.int16)
        audio_i16 = x_new
        sr = target_sr

    return WavPcm16(sample_rate=sr, pcm16le=audio_i16.astype("<i2").tobytes())


def _create_session(base_url: str, timeout_s: float = 10.0) -> str:
    r = requests.post(f"{base_url}/api/sessions", timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    session_id = data.get("session_id")
    if not session_id:
        raise RuntimeError(f"No session_id in response: {data}")
    return str(session_id)


def _http_to_ws(base_url: str) -> str:
    # http://localhost:8000 -> ws://localhost:8000
    # https://... -> wss://...
    u = urllib.parse.urlparse(base_url)
    if u.scheme == "http":
        scheme = "ws"
    elif u.scheme == "https":
        scheme = "wss"
    else:
        raise ValueError(f"Unsupported base URL scheme: {u.scheme}")
    return urllib.parse.urlunparse((scheme, u.netloc, "", "", "", ""))


def _start_events_ws(ws_url: str, out_q: queue.Queue[str]) -> websocket.WebSocketApp:
    def on_message(_ws, message):
        # message can be str or bytes
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8", errors="replace")
            except Exception:
                message = repr(message)
        out_q.put(str(message))

    def on_error(_ws, err):
        out_q.put(f"__error__:{err!r}")

    app = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error)
    t = threading.Thread(target=app.run_forever, kwargs={"ping_interval": 10, "ping_timeout": 5}, daemon=True)
    t.start()
    return app


def _connect_audio_ws(ws_url: str) -> websocket.WebSocket:
    # Use the blocking websocket for binary send
    return websocket.create_connection(ws_url, timeout=10)


def _stream_audio(ws: websocket.WebSocket, pcm: WavPcm16, frame_ms: int = 20, max_seconds: float = 3.0) -> None:
    sr = pcm.sample_rate
    samples_per_frame = int(sr * (frame_ms / 1000.0))
    bytes_per_sample = 2
    frame_bytes = samples_per_frame * bytes_per_sample

    total_bytes = min(len(pcm.pcm16le), int(max_seconds * sr) * bytes_per_sample)
    sent = 0
    while sent < total_bytes:
        chunk = pcm.pcm16le[sent : sent + frame_bytes]
        if not chunk:
            break
        ws.send(chunk, opcode=websocket.ABNF.OPCODE_BINARY)
        sent += len(chunk)
        time.sleep(frame_ms / 1000.0)


def _wait_for_event(out_q: queue.Queue[str], timeout_s: float = 15.0) -> Optional[dict]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        remaining = max(0.1, deadline - time.time())
        try:
            msg = out_q.get(timeout=remaining)
        except queue.Empty:
            continue

        if msg.startswith("__error__:"):
            # keep going; errors are still useful for debugging
            continue

        # Try to parse JSON event
        try:
            return json.loads(msg)
        except Exception:
            # ignore non-JSON payloads
            continue

    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="BFF base URL (default: http://localhost:8000)")
    ap.add_argument("--wav", default="tests/data/test_cs.wav", help="Path to a WAV file")
    ap.add_argument("--lang", default="en", help="Language hint (e.g. en, cs)")
    ap.add_argument("--stream-seconds", type=float, default=3.0, help="How many seconds of audio to stream")
    ap.add_argument("--events-timeout", type=float, default=20.0, help="Seconds to wait for at least one WS event")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    ws_base = _http_to_ws(base_url)

    print(f"[smoke] base_url={base_url}")

    session_id = _create_session(base_url)
    print(f"[smoke] created session_id={session_id}")

    pcm = _read_wav_as_pcm16le(args.wav, target_sr=16000)
    print(f"[smoke] loaded wav={args.wav} sr={pcm.sample_rate} bytes={len(pcm.pcm16le)}")

    # 1) WS events
    events_q: queue.Queue[str] = queue.Queue()
    events_url = f"{ws_base}/ws/events?session_id={urllib.parse.quote(session_id)}"
    print(f"[smoke] connecting events ws: {events_url}")
    events_app = _start_events_ws(events_url, events_q)

    # Give it a moment to connect
    time.sleep(0.5)

    # 2) WS audio
    audio_url = (
        f"{ws_base}/ws/audio?session_id={urllib.parse.quote(session_id)}"
        f"&lang={urllib.parse.quote(args.lang)}&sample_rate={pcm.sample_rate}"
    )
    print(f"[smoke] connecting audio ws: {audio_url}")
    audio_ws = _connect_audio_ws(audio_url)

    try:
        print(f"[smoke] streaming {args.stream_seconds:.1f}s of audio...")
        _stream_audio(audio_ws, pcm, frame_ms=20, max_seconds=args.stream_seconds)
        print("[smoke] done streaming; waiting for events...")

        ev = _wait_for_event(events_q, timeout_s=args.events_timeout)
        if not ev:
            print("[smoke] FAIL: did not receive any JSON event on /ws/events")
            return 1

        print(f"[smoke] PASS: received event: type={ev.get('type')} keys={list(ev.keys())}")
        return 0

    finally:
        try:
            audio_ws.close()
        except Exception:
            pass
        try:
            events_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
