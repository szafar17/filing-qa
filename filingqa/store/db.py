"""Postgres connections and a tiny migration runner (numbered .sql files, applied once)."""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


# JIT compilation costs ~0.4 s per query and only pays off for long analytical scans; our
# queries are short index lookups, so it is disabled per connection.
SESSION_OPTIONS = "-c jit=off"


def connect(url: str) -> psycopg.Connection:
    """Autocommit connection; multi-statement work uses explicit ``conn.transaction()``."""
    return psycopg.connect(url, autocommit=True, options=SESSION_OPTIONS)


def make_pool(url: str, min_size: int = 1, max_size: int = 8):
    from psycopg_pool import ConnectionPool

    return ConnectionPool(
        url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True, "options": SESSION_OPTIONS},
        open=True,
    )


def migrate(conn: psycopg.Connection, migrations_dir: Path | None = None) -> list[str]:
    folder = migrations_dir or MIGRATIONS_DIR
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
    applied: list[str] = []
    for path in sorted(folder.glob("*.sql")):
        if path.name in done:
            continue
        with conn.transaction():
            conn.execute(path.read_text())
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,))
        applied.append(path.name)
        log.info("applied migration", extra={"data": {"event": "migration", "file": path.name}})
    return applied
