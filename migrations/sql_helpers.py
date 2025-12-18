# migrations/sql_helpers.py
from pathlib import Path
from alembic import op

def run_sql_pair(revision_py_file: str, base_name: str):
    path = Path(revision_py_file).resolve()
    up = path.with_name(f"{base_name}.up.sql").read_text(encoding="utf-8")
    op.execute(up)

def run_sql_pair_down(revision_py_file: str, base_name: str):
    path = Path(revision_py_file).resolve()
    down = path.with_name(f"{base_name}.down.sql").read_text(encoding="utf-8")
    op.execute(down)