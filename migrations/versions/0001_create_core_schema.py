#migrations/versions/0001_create_core_schema.py
from migrations.sql_helpers import run_sql_pair, run_sql_pair_down

revision = "0001_create_core_schema"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    run_sql_pair(__file__, "0001_create_core_schema")

def downgrade():
    run_sql_pair_down(__file__, "0001_create_core_schema")