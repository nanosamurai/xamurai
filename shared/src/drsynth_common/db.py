# drsynth_common/db.py
from __future__ import annotations
import os
from contextlib import contextmanager
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ.get("DATABASE_URL")  # e.g. postgresql://user:pass@localhost:5432/db

_pool = None

def _get_pool():
    """Lazy initialization of the connection pool."""
    global _pool
    if _pool is None:
        if DATABASE_URL is None:
            raise ValueError(
                "DATABASE_URL environment variable is not set. "
                "Please set it to a valid PostgreSQL connection string."
            )
        _pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=10, open=True)
    return _pool

@contextmanager
def get_conn():
    with _get_pool().connection() as conn:
        yield conn

def exec1(sql: str, params: tuple = ()) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()

def fetch_one(sql: str, params: tuple = ()):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

def fetch_val(sql: str, params: tuple = ()):
    row = fetch_one(sql, params)
    return row[0] if row else None
