import asyncio
import json
import logging
import os
import queue
import threading
import time
from typing import Dict, Optional

import grpc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from confluent_kafka import Producer

from bff.refined_bus import RefinedBus
from bff.settings import settings
from bff.kafka_io import make_producer, produce_audio_chunk
from bff.auth import verify_token, OIDCError, OIDCUser

from proto import stream_pb2, stream_pb2_grpc

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "false").lower() == "true"

# Address of realtime gRPC service
RTSERVICE_ADDR = os.getenv("RTSERVICE_ADDR", "localhost:50052")

app = FastAPI(title="BFF WS (Realtime + Kafka + gRPC)")

# Serve your test GUI from /ui (bff/web/index.html)
app.mount("/ui", StaticFiles(directory="bff/web", html=True), name="ui")

# In-proc singletons (OK for MVP)
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


async def _authenticate_ws(websocket: WebSocket) -> Optional[OIDCUser]:
    """
    Extract and verify a Bearer token from the WS handshake.

    We check:
    1) Authorization: Bearer <token>
    2) ?token=<token> query param

    Returns OIDCUser on success, or None if no/invalid token AND AUTH_REQUIRED is False.
    Raises WebSocketDisconnect if AUTH_REQUIRED and auth fails.
    """
    # 1) From headers
    auth_header = websocket.headers.get("authorization")
    token: Optional[str] = None

    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()

    # 2) From query param (e.g. ws://.../ws?token=eyJ...)
    if token is None:
        token_q = websocket.query_params.get("token")
        if token_q:
            token = token_q.strip()

    if not token:
        if AUTH_REQUIRED:
            logger.info("WS auth failed: missing token")
            await websocket.close(code=1008)  # policy violation
            raise WebSocketDisconnect(code=1008)
        else:
            logger.warning("WS: no token provided, continuing (AUTH_REQUIRED=false)")
            return None

    try:
        user = verify_token(token)
        logger.info(
            "WS auth success: sub=%s username=%s email=%s",
            user.sub,
            user.preferred_username,
            user.email,
        )
        return user
    except OIDCError as e:
        if AUTH_REQUIRED:
            logger.info("WS auth failed: %s", e)
            await websocket.close(code=1008)
            raise WebSocketDisconnect(code=1008)
        else:
            logger.warning("WS auth error (ignored because AUTH_REQUIRED=false): %s", e)
            return None


def _start_grpc_stream(
    session_id: str,
    lang: str,
    req_q: "queue.Queue[Optional[stream_pb2.AudioChunk]]",
    resp_q: "queue.Queue[Optional[stream_pb2.AsrEvent]]",
) -> threading.Thread:
    """
    Start a background thread that:
      - reads AudioChunk from req_q
      - sends them to RealtimeASR.Stream
      - pushes AsrEvent into resp_q

    Sentinels:
      - None in req_q => close request iterator
      - Puts None in resp_q when stream finishes
    """

    def run():
        logger.info("Starting gRPC stream thread for session=%s", session_id)
        channel = grpc.insecure_channel(RTSERVICE_ADDR)
        stub = stream_pb2_grpc.RealtimeASRStub(channel)

        def request_iter():
            while True:
                chunk = req_q.get()
                if chunk is None:
                    logger.info(
                        "gRPC request iterator closing for session=%s", session_id
                    )
                    break
                yield chunk

        try:
            for ev in stub.Stream(request_iter()):
                resp_q.put(ev)
        except grpc.RpcError as e:
            logger.error(
                "gRPC stream error for session=%s: %s (code=%s)",
                session_id,
                e,
                getattr(e, "code", lambda: None)(),
            )
        except Exception as e:
            logger.exception("gRPC stream unexpected error for session=%s: %s", session_id, e)
        finally:
            try:
                channel.close()
            except Exception:
                pass
            # Sentinel to tell async loop we're done
            resp_q.put(None)
            logger.info("gRPC stream thread finished for session=%s", session_id)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


@app.websocket("/ws")
async def ws_audio(
    websocket: WebSocket,
    session_id: str = Query(...),
    session_lang: str = Query(...),
):
    await websocket.accept()

    # Authenticate (or not, depending on AUTH_REQUIRED)
    user: Optional[OIDCUser] = None
    try:
        user = await _authenticate_ws(websocket)
    except WebSocketDisconnect:
        # Already closed in _authenticate_ws
        return

    session_seq.setdefault(session_id, 0)

    # register for refined (offline) events for this session
    refined_queue = await refined_bus.register(session_id)

    # per-session gRPC queues and thread
    req_q: "queue.Queue[Optional[stream_pb2.AudioChunk]]" = queue.Queue()
    resp_q: "queue.Queue[Optional[stream_pb2.AsrEvent]]" = queue.Queue()
    grpc_thread = _start_grpc_stream(session_id, session_lang, req_q, resp_q)

    logger.info(
        "WS connected: session=%s lang=%s user=%s",
        session_id,
        session_lang,
        getattr(user, "preferred_username", None),
    )

    async def recv_audio_loop():
        """Receive audio frames, put them on Kafka + gRPC request queue."""
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
                    logger.error(
                        "KAFKA produce failed (session=%s): %s", session_id, e
                    )

                # 2) Forward to gRPC realtime ASR
                chunk_msg = stream_pb2.AudioChunk(
                    session_id=session_id,
                    t0_ns=time.time_ns(),
                    pcm16_le=pcm16,
                    sample_rate=settings.sample_rate,
                    lang=session_lang or "",
                )
                try:
                    req_q.put_nowait(chunk_msg)
                except queue.Full:
                    # with unbounded Queue this shouldn't happen, but log just in case
                    logger.warning(
                        "gRPC request queue is full (session=%s), dropping frame",
                        session_id,
                    )

        except WebSocketDisconnect:
            logger.info("WS disconnected (recv loop): session=%s", session_id)
        except Exception as e:
            logger.exception("[WS recv] error (session=%s): %s", session_id, e)
        finally:
            # signal gRPC thread to close request iterator
            try:
                req_q.put_nowait(None)
            except Exception:
                pass

    async def send_asr_loop():
        """Read AsrEvent from gRPC response queue and push via WS."""
        try:
            loop = asyncio.get_running_loop()
            while True:
                ev = await asyncio.to_thread(resp_q.get)
                if ev is None:
                    logger.info(
                        "gRPC stream closed (send_asr_loop) for session=%s", session_id
                    )
                    break

                payload = {
                    "type": "asr",
                    "session_id": ev.session_id,
                    "start_s": ev.start_s,
                    "end_s": ev.end_s,
                    "text": ev.text,
                    "lang": ev.lang,
                    "speaker": ev.speaker,
                    "final": (ev.type == stream_pb2.FINAL),
                }
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except WebSocketDisconnect:
            logger.info("WS disconnected (asr loop): session=%s", session_id)
        except Exception:
            logger.exception("[WS asr] error (session=%s)", session_id)

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

    # Run all three coroutines until one finishes / fails
    recv_task = asyncio.create_task(recv_audio_loop())
    asr_task = asyncio.create_task(send_asr_loop())
    refined_task = asyncio.create_task(send_refined_loop())

    try:
        done, pending = await asyncio.wait(
            {recv_task, asr_task, refined_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for t in pending:
            t.cancel()
    finally:
        refined_bus.unregister(session_id)
        # close gRPC request side if not already
        try:
            req_q.put_nowait(None)
        except Exception:
            pass
        # give the gRPC thread a moment to finish
        try:
            grpc_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            producer.flush(2.0)
        except Exception:
            logger.exception("[WS final] error while flushing Kafka producer")
        logger.info("WS closed: session=%s", session_id)
