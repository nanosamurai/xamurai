# bff/src/bff/app.py
import asyncio
import json
import logging
import os
import queue
import threading
import time
from typing import Dict, Optional

import grpc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from confluent_kafka import Producer

from bff.refined_bus import RefinedBus
from bff.settings import settings
from bff.kafka_io import make_producer, produce_audio_chunk
from bff.auth import verify_token, OIDCError, OIDCUser
from bff.sessions import start_session_db

import stream_pb2, stream_pb2_grpc

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "false").lower() == "true"

# Address of realtime gRPC service
RTSERVICE_ADDR = os.getenv("RTSERVICE_ADDR", "localhost:50052")

# to run locally in powershell:
# $env:PYTHONPATH = ".;bff\src;proto_gen"
# python -m uvicorn bff.app:app --host 0.0.0.0 --port 8000
app = FastAPI(title="BFF WS (Realtime + Kafka + gRPC)")

# Serve GUI from /ui (bff/web/index.html)
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
    return HTMLResponse("<a href='/ui'>Open WS test UI</a>")


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #
def _extract_token(websocket: WebSocket, token_q: Optional[str]) -> Optional[str]:
    """
    Token sources (in order):
      1) Authorization: Bearer <token>
      2) ?token=<token>
    """
    auth_header = websocket.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    if token_q:
        return token_q.strip()
    return None


def _extract_tenant_from_claims(user: Optional[OIDCUser]) -> Optional[str]:
    """
    Best-effort tenant extraction from token claims.
    Adjust these keys once we standardize Keycloak mapper.
    """
    if not user:
        return None

    claims = user.raw or {}
    for k in ("tenant_id", "tenant", "org_id", "organization_id", "company_id"):
        v = claims.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Optional role encoding: realm_access.roles contains e.g. "tenant:<uuid>"
    try:
        roles = (claims.get("realm_access") or {}).get("roles") or []
        for r in roles:
            if isinstance(r, str) and r.startswith("tenant:"):
                return r.split("tenant:", 1)[1].strip() or None
    except Exception:
        pass

    return None


async def _authenticate_before_accept(
    websocket: WebSocket,
    token_q: Optional[str],
) -> Optional[OIDCUser]:
    """
    Authenticate *before* websocket.accept().

    If AUTH_REQUIRED is True:
      - missing/invalid token => HTTP 403 (handshake rejected)
    If AUTH_REQUIRED is False:
      - missing/invalid token => return None
      - valid token => return OIDCUser
    """
    token = _extract_token(websocket, token_q)

    if not token:
        if AUTH_REQUIRED:
            logger.info("WS auth failed: missing token (AUTH_REQUIRED=true)")
            raise HTTPException(status_code=403, detail="Missing token")
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
            logger.info("WS auth failed: %s (AUTH_REQUIRED=true)", e)
            raise HTTPException(status_code=403, detail="Invalid token")
        logger.warning("WS auth error ignored (AUTH_REQUIRED=false): %s", e)
        return None


# --------------------------------------------------------------------------- #
# gRPC stream helper
# --------------------------------------------------------------------------- #
def _start_grpc_stream(
    session_id: str,
    lang: str,
    req_q: "queue.Queue[Optional[stream_pb2.AudioChunk]]",
    resp_q: "queue.Queue[Optional[stream_pb2.AsrEvent]]",
) -> threading.Thread:
    def run():
        logger.info("Starting gRPC stream thread for session=%s", session_id)
        channel = grpc.insecure_channel(RTSERVICE_ADDR)
        stub = stream_pb2_grpc.RealtimeASRStub(channel)

        def request_iter():
            while True:
                chunk = req_q.get()
                if chunk is None:
                    logger.info("gRPC request iterator closing for session=%s", session_id)
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
            resp_q.put(None)
            logger.info("gRPC stream thread finished for session=%s", session_id)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


# --------------------------------------------------------------------------- #
# WebSocket endpoint
# --------------------------------------------------------------------------- #
@app.websocket("/ws")
async def ws_audio(
    websocket: WebSocket,
    session_id: str = Query(...),
    session_lang: str = Query(...),

    # Browser-friendly auth + tenant
    token: str | None = Query(None),
    tenant_id: str | None = Query(None),
    user_id: str | None = Query(None),

    # Non-browser clients may still send headers (optional!)
    x_tenant_id: str | None = Header(None),
    x_user_id: str | None = Header(None),
):
    # 1) Authenticate BEFORE accept (reject handshake cleanly if needed)
    user: Optional[OIDCUser] = await _authenticate_before_accept(websocket, token_q=token)

    # 2) Resolve tenant/user identifiers
    resolved_tenant_id = tenant_id or x_tenant_id or _extract_tenant_from_claims(user)
    resolved_user_id = user_id or x_user_id

    # Decide whether we require tenant at all
    # For MVP, we *allow* WS streaming without tenant, but we skip persistence.
    if AUTH_REQUIRED and not resolved_tenant_id:
        # If auth is mandatory, tenant should usually be present too.
        # Rejecting now avoids "works but no persistence" surprises.
        logger.info("WS rejected: authenticated but missing tenant_id")
        raise HTTPException(status_code=403, detail="Missing tenant_id")

    # 3) Accept only after we’re satisfied
    await websocket.accept()

    # 4) Persist session (best-effort; does not kill WS if DB down)
    if resolved_tenant_id:
        try:
            session_data = start_session_db(
                x_tenant_id=resolved_tenant_id,
                x_user_id=resolved_user_id,
            )
            logger.info(
                "Session persisted: session_id=%s session_key=%s",
                session_data["session_id"],
                session_data["session_key"],
            )
        except Exception as e:
            logger.error("Failed to persist session: %s", e)
    else:
        logger.warning("Skipping session persistence: tenant_id not provided/resolved")

    session_seq.setdefault(session_id, 0)

    # register for refined (offline) events for this session
    refined_queue = await refined_bus.register(session_id)

    # per-session gRPC queues and thread
    req_q: "queue.Queue[Optional[stream_pb2.AudioChunk]]" = queue.Queue()
    resp_q: "queue.Queue[Optional[stream_pb2.AsrEvent]]" = queue.Queue()
    grpc_thread = _start_grpc_stream(session_id, session_lang, req_q, resp_q)

    logger.info(
        "WS connected: session=%s lang=%s user=%s tenant=%s",
        session_id,
        session_lang,
        getattr(user, "preferred_username", None),
        resolved_tenant_id,
    )

    async def recv_audio_loop():
        try:
            while True:
                data = await websocket.receive()

                if data.get("type") == "websocket.disconnect":
                    logger.info(
                        "WS disconnect frame received (recv loop): session=%s code=%s",
                        session_id,
                        data.get("code"),
                    )
                    break

                if "bytes" not in data or data["bytes"] is None:
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
                    if session_seq[session_id] % 10 == 0:
                        producer.poll(0)
                except Exception as e:
                    logger.error("KAFKA produce failed (session=%s): %s", session_id, e)

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
                    logger.warning("gRPC request queue full (session=%s), dropping frame", session_id)

        except WebSocketDisconnect:
            logger.info("WS disconnected (recv loop): session=%s", session_id)
        except Exception as e:
            logger.exception("[WS recv] error (session=%s): %s", session_id, e)
        finally:
            try:
                req_q.put_nowait(None)
            except Exception:
                pass

    async def send_asr_loop():
        try:
            while True:
                ev = await asyncio.to_thread(resp_q.get)
                if ev is None:
                    logger.info("gRPC stream closed (send_asr_loop) for session=%s", session_id)
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
        try:
            while True:
                ev = await refined_queue.get()
                await websocket.send_text(json.dumps(ev, ensure_ascii=False))
        except WebSocketDisconnect:
            logger.info("WS disconnected (refined loop): session=%s", session_id)
        except Exception:
            logger.exception("[WS refined] error (session=%s)", session_id)

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

        try:
            req_q.put_nowait(None)
        except Exception:
            pass

        try:
            grpc_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            producer.flush(2.0)
        except Exception:
            logger.exception("[WS final] error while flushing Kafka producer")

        logger.info("WS closed: session=%s", session_id)
