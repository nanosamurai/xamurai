"""Tier 4 (optional) smoke test: verify async pipeline signals via Kafka.

PASS criteria (configurable):
- Observe at least one downstream event for the session_id, e.g.:
  - recordings.finished (RecordingFinished)
  - transcripts.refined (RefinedEvent)
  - transcripts.final (SessionTranscript)

Default mode focuses on recordings.finished because it is the quickest bounded async signal
(RECORDER_IDLE_SECONDS, default 30s).

Requires:
  pip install -r utilities/k8s_local_smoke_test/requirements.kafka.txt

Notes on time:
- recorder emits only after session idle (no audio) for RECORDER_IDLE_SECONDS.
- whisperx refined emits per slice (default 60s)
- final transcript depends on full-session processing and can be slow on CPU.

This test is therefore opt-in and has larger timeouts.
"""

from __future__ import annotations

import pathlib
import sys

# Allow running this file directly (without `python -m ...`).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse
import queue
import time
import urllib.parse

from confluent_kafka import Consumer

from proto_gen import stream_pb2
from utilities.k8s_local_smoke_test import _lib


def _make_consumer(bootstrap: str, group_id: str) -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=_lib.DEFAULT_BASE_URL)
    ap.add_argument("--wav", default="tests/data/test_cs.wav")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--stream-seconds", type=float, default=2.0)
    ap.add_argument("--kafka-bootstrap", required=True)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--check-recording-finished", action="store_true", default=True)
    ap.add_argument("--check-refined", action="store_true", default=False)
    ap.add_argument("--check-final", action="store_true", default=False)
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    ws_base = _lib.http_to_ws(base_url)

    session_id = _lib.create_session(base_url)
    print(f"[tier4] session_id={session_id}")

    pcm = _lib.read_wav_as_pcm16le(args.wav, target_sr=16000)

    topics = []
    if args.check_recording_finished:
        topics.append("recordings.finished")
    if args.check_refined:
        topics.append("transcripts.refined")
    if args.check_final:
        topics.append("transcripts.final")

    if not topics:
        raise SystemExit("No topics selected")

    consumer = _make_consumer(args.kafka_bootstrap, group_id=f"tier4-{int(time.time())}")
    consumer.subscribe(topics)

    events_q: queue.Queue[str] = queue.Queue()
    events_url = f"{ws_base}/ws/events?session_id={urllib.parse.quote(session_id)}"
    audio_url = (
        f"{ws_base}/ws/audio?session_id={urllib.parse.quote(session_id)}"
        f"&lang={urllib.parse.quote(args.lang)}&sample_rate={pcm.sample_rate}"
    )

    events_app = _lib.start_events_ws(events_url, events_q)
    time.sleep(0.5)
    audio_ws = _lib.connect_audio_ws(audio_url)

    try:
        print(f"[tier4] streaming {args.stream_seconds:.1f}s audio")
        _lib.stream_audio(audio_ws, pcm, frame_ms=20, max_seconds=args.stream_seconds)
        print("[tier4] waiting for async pipeline events (stop sending audio to allow idle)...")

        deadline = time.time() + args.timeout
        while time.time() < deadline:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                continue

            topic = msg.topic()
            payload = msg.value()

            if topic == "recordings.finished":
                ev = stream_pb2.RecordingFinished()
                ev.ParseFromString(payload)
                if ev.session_id == session_id:
                    print(f"[tier4] PASS: got RecordingFinished for session={session_id} url={ev.recording_url}")
                    return 0

            if topic == "transcripts.refined":
                ev = stream_pb2.RefinedEvent()
                ev.ParseFromString(payload)
                if ev.session_id == session_id:
                    print(f"[tier4] PASS: got RefinedEvent for session={session_id} text_len={len(ev.text)}")
                    return 0

            if topic == "transcripts.final":
                ev = stream_pb2.SessionTranscript()
                ev.ParseFromString(payload)
                if ev.session_id == session_id:
                    print(f"[tier4] PASS: got SessionTranscript for session={session_id} segments={len(ev.segments)}")
                    return 0

        print(f"[tier4] FAIL: did not observe any selected events within {args.timeout:.0f}s")
        return 1

    finally:
        try:
            audio_ws.close()
        except Exception:
            pass
        try:
            events_app.close()
        except Exception:
            pass
        try:
            consumer.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
