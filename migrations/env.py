# migrations/env.py
from __future__ import annotations

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

def get_db_url() -> str:
    """Get the database URL, preferring SQLALCHEMY_DATABASE_URL but falling back to DATABASE_URL."""
    # Use SQLALCHEMY_DATABASE_URL for Alembic/SQLAlchemy
    # This should be in the format: postgresql+psycopg://user:pass@host/db
    db_url = os.getenv("SQLALCHEMY_DATABASE_URL")
    if not db_url:
        # Fallback to DATABASE_URL if SQLALCHEMY_DATABASE_URL is not set
        # but convert it to SQLAlchemy format if needed
        db_url = os.getenv("DATABASE_URL")
        if db_url and not db_url.startswith("postgresql+psycopg"):
            # Convert psycopg-native URL to SQLAlchemy format
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://")

    if not db_url:
        raise RuntimeError("Neither SQLALCHEMY_DATABASE_URL nor DATABASE_URL is set")

    return db_url

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    config.set_main_option("sqlalchemy.url", get_db_url())
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Use the already-computed db_url from get_db_url()
    # This ensures consistency between offline and online modes
    config.set_main_option("sqlalchemy.url", get_db_url())
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
