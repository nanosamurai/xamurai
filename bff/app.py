import asyncio
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
from bff.refined_bus import RefinedBus  # <— your RefinedBus class

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

# Single refined-bus instance for all sessions
refined_bus = RefinedBus()


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    refined_bus.start(loop)
    logger.info("BFF startup complete (RefinedBus running).")


@app.on_event("shutdown")
async def on_shutdown():
    refined_bus.stop()
    try:
        producer.flush(2.0)
    except Exception:
        logger.exception("Error flushing Kafka producer on shutdown")
    logger.info("BFF shutdown complete.")


@app.get("/")
async def root():
    # optional: simple pointer page
    return HTMLResponse("<a href='/ui'>Open WS test UI</a>")


@app.websocket("/ws")
async def ws_audio(
    websocket: WebSocket,
    session_id: str = Query(...),
    session_lang: str = Query(...),
):
    await websocket.accept()
    session_seq.setdefault(session_id, 0)

    # register for refined (offline) events for this session
    refined_queue = await refined_bus.register(session_id)

    logger.info("WS connected: session=%s lang=%s", session_id, session_lang)

    async def recv_audio_loop():
        """Receive audio frames, feed realtime engine, and push ASR events."""
        try:
            while True:
                data = await websocket.receive()

                if "bytes" not in data or data["bytes"] is None:
                    # ignore non-binary frames for now
                    continue

                pcm16 = data["bytes"]

                # 1) Produce to Kafka
                session_seq[session_id] += 1
                try:
                    produce_audio_chunk(
                        producer,
                        session_id=session_id,
                        seq=session_seq[session_id],
                        pcm16_bytes=pcm16,
                        sample_rate=settings.sample_rate,
                    )
                    # poll occasionally for delivery callbacks
                    if session_seq[session_id] % 10 == 0:
                        producer.poll(0)
                except Exception as e:
                    # Just log for now; do not kill the WS session
                    logger.error("KAFKA produce failed (session=%s): %s", session_id, e)

                # 2) Realtime engine → events → WS
                results = engine.feed(session_id, pcm16, lang=session_lang)
                for r in results:
                    ev = engine.to_asr_events(session_id, r)
                    payload = {
                        "type": "asr",
                        "session_id": ev.session_id,
                        "start_s": ev.start_s,
                        "end_s": ev.end_s,
                        "text": ev.text,
                        "lang": ev.lang,
                        "speaker": ev.speaker,
                        "final": (ev.type == 1),  # AsrType.FINAL
                    }
                    await websocket.send_text(json.dumps(payload, ensure_ascii=False))

        except WebSocketDisconnect:
            logger.info("WS disconnected (recv loop): session=%s", session_id)
        except Exception as e:
            logger.exception("[WS recv] error (session=%s): %s", session_id, e)

    async def send_refined_loop():
        """Forward refined WhisperX events for this session via the same WS."""
        try:
            while True:
                ev = await refined_queue.get()
                # ev is already a dict prepared by RefinedBus
                await websocket.send_text(json.dumps(ev, ensure_ascii=False))
        except WebSocketDisconnect:
            logger.info("WS disconnected (refined loop): session=%s", session_id)
        except Exception:
            logger.exception("[WS refined] error (session=%s)", session_id)

    # Run both coroutines until one finishes / fails
    recv_task = asyncio.create_task(recv_audio_loop())
    refined_task = asyncio.create_task(send_refined_loop())

    try:
        done, pending = await asyncio.wait(
            {recv_task, refined_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for t in pending:
            t.cancel()
    finally:
        refined_bus.unregister(session_id)
        try:
            producer.flush(2.0)
        except Exception:
            logger.exception("[WS final] error while flushing Kafka producer")
        logger.info("WS closed: session=%s", session_id)
