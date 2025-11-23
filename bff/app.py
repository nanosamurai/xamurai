# bff/app.py
import json
import logging
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from confluent_kafka import Producer

from bff.settings import settings
from bff.kafka_io import make_producer, produce_audio_chunk
from bff.engine import RealtimeEngine

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


app = FastAPI(title="BFF WS (Realtime + Kafka)")

# Serve your test GUI from /ui (bff/web/index.html)
app.mount("/ui", StaticFiles(directory="bff/web", html=True), name="ui")

# In-proc singletons (OK for MVP)
engine = RealtimeEngine()
producer: Producer = make_producer()
session_seq: Dict[str, int] = {}

@app.get("/")
async def root():
    # optional: simple pointer page
    return HTMLResponse("<a href='/ui'>Open WS test UI</a>")

@app.websocket("/ws")
async def ws_audio(websocket: WebSocket, session_id: str = Query(...), session_lang: str = Query(...)):
    await websocket.accept()
    session_seq.setdefault(session_id, 0)
    logger.info("WS connected: session=%s lang=%s", session_id, session_lang)

    try:
        while True:
            data = await websocket.receive()

            if "bytes" not in data or data["bytes"] is None:
                # ignore non-binary frames for now
                continue

            pcm16 = data["bytes"]

            # 1) Produce to Kafka (with basic error protection)
            session_seq[session_id] += 1
            try:
                produce_audio_chunk(
                    producer,
                    session_id=session_id,
                    seq=session_seq[session_id],
                    pcm16_bytes=pcm16,
                    sample_rate=settings.sample_rate,
                )
                if session_seq[session_id] % 10 == 0:
                    producer.poll(0)  # service delivery callbacks
            except Exception as e:
                # Just log for now; do not kill the WS session
                logger.error("KAFKA produce failed: %s", e)

            # 2) Realtime engine → events → WS
            results = engine.feed(session_id, pcm16, lang=session_lang)
            for r in results:
                ev = engine.to_asr_events(session_id, r)
                await websocket.send_text(json.dumps({
                    "type": "asr",
                    "session_id": ev.session_id,
                    "start_s": ev.start_s,
                    "end_s": ev.end_s,
                    "text": ev.text,
                    "lang": ev.lang,
                    "final": (ev.type == 1),  # AsrType.FINAL
                }, ensure_ascii=False))

    except WebSocketDisconnect:
        logger.info("WS disconnected: session=%s", session_id)
    except Exception as e:
        logger.exception(f"[WS] error: {e}")
    finally:
        try:
            producer.flush(2.0)
        except Exception:
            logger.exception("[WS] error while flushing Kafka producer")
