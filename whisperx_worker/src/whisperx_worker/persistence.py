# whisperx_worker/src/whisperx_worker/persistence.py
import uuid
from drsynth_common.db import exec1

def ensure_session(session_key: str, tenant_id: str | None, user_id: str | None = None) -> None:
    if not tenant_id:
        tenant_id = "00000000-0000-0000-0000-000000000000"  # or skip if you prefer strictness

    # Create a stub if missing (idempotent)
    exec1(
        """
        INSERT INTO sessions (id, tenant_id, user_id, session_key, status)
        VALUES (%s, %s, %s, %s, 'active')
        ON CONFLICT (session_key) DO NOTHING
        """,
        (uuid.uuid4(), tenant_id, user_id, session_key),
    )
