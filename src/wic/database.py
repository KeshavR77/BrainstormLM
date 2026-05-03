"""
database.py — Async PostgreSQL connection pool and schema initialization
for the Recursive What-If Cascade (WIC) Engine.
"""

import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wic_engine",
)

pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the existing connection pool or create one."""
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(dsn=DATABASE_URL)
    return pool


async def close_pool() -> None:
    """Gracefully close the connection pool."""
    global pool
    if pool is not None:
        await pool.close()
        pool = None


# ── DDL ──────────────────────────────────────────────────────────────────

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   UUID         PRIMARY KEY,
    user_id      TEXT,
    status       TEXT         NOT NULL,
    current_depth INT         DEFAULT 0,
    created_at   TIMESTAMPTZ  DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id      UUID         PRIMARY KEY,
    session_id   UUID         REFERENCES sessions(session_id),
    parent_id    UUID         REFERENCES nodes(node_id),
    depth        INT          NOT NULL,
    node_type    TEXT         NOT NULL,
    payload      JSONB        NOT NULL,
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id         UUID      PRIMARY KEY,
    session_id      UUID      REFERENCES sessions(session_id),
    parent_node_id  UUID      REFERENCES nodes(node_id),
    child_node_id   UUID      REFERENCES nodes(node_id),
    edge_type       TEXT      NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""


async def init_db() -> None:
    """Create tables if they don't already exist."""
    db = await get_pool()
    async with db.acquire() as conn:
        await conn.execute(SCHEMA_DDL)
    print("✅ Database schema initialised.")


# ── Convenience: run standalone ──────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    asyncio.run(init_db())
