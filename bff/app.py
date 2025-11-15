import json
import time
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from confluent_kafka import Producer

from bff.settings import settings
from bff.kafka_io import make_producer, produce_audio_chunk
from bff.engine import RealtimeEngine

app = FastAPI(title="BFF WS (Realtime + Kafka)")

# serve / -> bff/web/index.html and any other assets in that folder
app.mount("/", StaticFiles(directory="bff/web", html=True), name="web")

# in-proc singletons (ok for MVP)
engine = RealtimeEngine()
producer: Producer = make_producer()

# Track sequence per session
session_seq: Dict[str, int] = {}


@app.websocket("/ws")
async def ws_audio(websocket: WebSocket, session_id: str = Query(...)):
    await websocket.accept()
    session_seq.setdefault(session_id, 0)
    print(f"[WS] connected session={session_id}")

    try:
        while True:
            data = await websocket.receive()
            if "bytes" not in data or data["bytes"] is None:
                # ignore non-binary control frames for MVP
                continue

            pcm16 = data["bytes"]
            # 1) Produce to Kafka
            session_seq[session_id] += 1
            produce_audio_chunk(
                producer,
                session_id=session_id,
                seq=session_seq[session_id],
                pcm16_bytes=pcm16,
                sample_rate=settings.sample_rate,
            )
            # Flush occasionally; tune in prod
            if session_seq[session_id] % 10 == 0:
                producer.poll(0)

            # 2) Realtime engine in-process → events → push to client
            results = engine.feed(session_id, pcm16)
            for r in results:
                ev = engine.to_asr_events(session_id, r)
                # For dev ergonomics we send JSON over WS; you can switch to binary/protobuf
                await websocket.send_text(json.dumps({
                    "type": "asr",
                    "session_id": ev.session_id,
                    "start_s": ev.start_s,
                    "end_s": ev.end_s,
                    "text": ev.text,
                    "final": (ev.type == 1),
                }))

    except WebSocketDisconnect:
        print(f"[WS] disconnected session={session_id}")
    except Exception as e:
        print(f"[WS] error: {e}")
    finally:
        try:
            producer.flush(2.0)
        except Exception:
            pass
