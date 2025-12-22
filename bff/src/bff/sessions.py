# bff/src/bff/sessions.py
import uuid
from fastapi import APIRouter, Header
from drsynth_common.db import fetch_one, exec1

router = APIRouter()

def uuid7() -> uuid.UUID:
    # Use a real uuid7 lib if you have it. For now fallback to uuid4.
    # Recommended: pip install uuid6  (it provides uuid7())
    try:
        from uuid6 import uuid7 as _uuid7  # type: ignore
        return _uuid7()
    except Exception:
        return uuid.uuid4()

def _start_session_db(x_tenant_id: str, x_user_id: str | None):
    """
    Internal function to start a session and persist it to the database.
    Returns the session data as a dict.
    """
    session_id = uuid7()
    session_key = str(session_id)  # <- simplest: make proto session_id == this string

    exec1(
        """
        INSERT INTO sessions (id, tenant_id, user_id, session_key, status)
        VALUES (%s, %s, %s, %s, 'active')
        """,
        (session_id, x_tenant_id, x_user_id, session_key),
    )

    return {"session_id": session_id, "session_key": session_key}

@router.post("/sessions/start")
def start_session(
    x_tenant_id: str = Header(...),
    x_user_id: str | None = Header(None),
):
    """
    FastAPI route handler for starting a session.
    """
    return _start_session_db(x_tenant_id, x_user_id)

# Export the internal function for use in app.py
start_session_db = _start_session_db
