"""
agent/query_logger.py

Logs every agent question + answer to PostgreSQL (ml.agent_queries table).
Demonstrates agent observability alongside model observability.

Schema (auto-created on first use)
───────────────────────────────────
  ml.agent_queries
    id            SERIAL PRIMARY KEY
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT now()
    question      TEXT NOT NULL
    answer        TEXT NOT NULL
    tools_called  JSONB          -- list of tool names invoked
    latency_ms    FLOAT
    model_version TEXT           -- production model version at query time
    error         TEXT           -- null if successful

Public API
──────────
  log_query(question, answer, tools_called, latency_ms, model_version, error)
  get_recent_queries(limit) -> list[dict]

Uses psycopg2 (already in project deps via PostgreSQL usage).
Connection params read from same env vars as the rest of the project.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ── Connection config (matches project .env) ──────────────────────────────────

_DSN = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'ml_monitoring')} "
    f"user={os.getenv('POSTGRES_USER', 'ml_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', '')} "
    f"options='-c search_path=ml,public'"
)

_CREATE_TABLE_SQL = """
CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.agent_queries (
    id            SERIAL PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),
    question      TEXT        NOT NULL,
    answer        TEXT        NOT NULL,
    tools_called  JSONB,
    latency_ms    FLOAT,
    model_version TEXT,
    error         TEXT
);
"""

_INSERT_SQL = """
INSERT INTO ml.agent_queries
    (timestamp, question, answer, tools_called, latency_ms, model_version, error)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

_SELECT_SQL = """
SELECT id, timestamp, question, answer, tools_called, latency_ms, model_version, error
FROM ml.agent_queries
ORDER BY timestamp DESC
LIMIT %s
"""

# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_conn():
    """Open a new psycopg2 connection. Caller must close it."""
    try:
        import psycopg2
        return psycopg2.connect(_DSN)
    except Exception as exc:
        log.error("DB connection failed: %s", exc)
        raise


def _ensure_table() -> None:
    """Create schema + table if they don't exist. Idempotent."""
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
        conn.close()
    except Exception as exc:
        log.error("Could not ensure agent_queries table: %s", exc)


# Run once at import time — non-fatal if DB is unreachable
try:
    _ensure_table()
except Exception:
    pass


# ── Public API ────────────────────────────────────────────────────────────────


def log_query(
    question: str,
    answer: str,
    tools_called: Optional[list[str]] = None,
    latency_ms: Optional[float] = None,
    model_version: Optional[str] = None,
    error: Optional[str] = None,
) -> bool:
    """
    Persist one agent interaction to ml.agent_queries.

    Args:
        question:      Raw user question.
        answer:        Final agent answer text.
        tools_called:  List of tool names invoked during the ReAct loop.
        latency_ms:    End-to-end wall-clock time in milliseconds.
        model_version: Production model version tag at query time.
        error:         Error message if the agent failed, else None.

    Returns:
        True if written successfully, False otherwise (non-fatal).
    """
    try:
        conn = _get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(_INSERT_SQL, (
                    datetime.now(timezone.utc),
                    question,
                    answer,
                    json.dumps(tools_called or []),
                    latency_ms,
                    model_version,
                    error,
                ))
        conn.close()
        log.debug(
            "Agent query logged — tools=%s latency=%.1fms",
            tools_called,
            latency_ms or 0,
        )
        return True
    except Exception as exc:
        log.error("Failed to log agent query: %s", exc)
        return False


def get_recent_queries(limit: int = 50) -> list[dict]:
    """
    Fetch the most recent agent queries from PostgreSQL.
    Used by the Streamlit Logs page to show agent activity.

    Args:
        limit: Max rows to return (default 50).

    Returns:
        List of dicts with keys:
        id, timestamp, question, answer, tools_called,
        latency_ms, model_version, error
    """
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(_SELECT_SQL, (max(1, min(limit, 500)),))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        conn.close()

        # Deserialise tools_called JSONB → Python list
        for row in rows:
            if isinstance(row.get("tools_called"), str):
                try:
                    row["tools_called"] = json.loads(row["tools_called"])
                except Exception:
                    pass
            # Stringify timestamp for JSON serialisation
            if hasattr(row.get("timestamp"), "isoformat"):
                row["timestamp"] = row["timestamp"].isoformat()

        return rows
    except Exception as exc:
        log.error("Failed to fetch agent queries: %s", exc)
        return []