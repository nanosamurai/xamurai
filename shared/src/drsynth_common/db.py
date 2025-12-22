# drsynth_common/db.py
from __future__ import annotations
import os
from contextlib import contextmanager
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ["DATABASE_URL"]  # e.g. postgresql://user:pass@localhost:5432/db

_pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=10, open=True)

@contextmanager
def get_conn():
    with _pool.connection() as conn:
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
