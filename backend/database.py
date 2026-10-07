"""
upay ActivateAI — SQLite Persistence Layer for Campaign Approvals.

Replaces the in-memory Python list with a durable SQLite ledger.
Per AGENTS.md guardrail #5: every campaign approval is recorded with
reviewer name, timestamp, and notes — now persisted to disk.
"""

from __future__ import annotations

import pathlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Database path — stored alongside other data artifacts
# ---------------------------------------------------------------------------
_DB_DIR = pathlib.Path(__file__).parent.parent / "data"
_DB_PATH = _DB_DIR / "campaign_approvals.db"


def _db_path() -> pathlib.Path:
    """Return the database file path, ensuring the parent directory exists."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


@contextmanager
def _get_conn():
    """Context manager for a short-lived SQLite connection."""
    conn = sqlite3.connect(str(_db_path()), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS campaign_approvals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id     TEXT    NOT NULL UNIQUE,
    admin_user      TEXT    NOT NULL,
    allocated_budget REAL   NOT NULL,
    notes           TEXT    NOT NULL DEFAULT '',
    timestamp       TEXT    NOT NULL
);
"""

_CREATE_REGISTRY_SQL = """
CREATE TABLE IF NOT EXISTS model_registry_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version   TEXT    NOT NULL,
    feature_name    TEXT    NOT NULL,
    drift_score     REAL    NOT NULL,
    status          TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL
);
"""

def init_db() -> None:
    """Create tables if they do not exist."""
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_REGISTRY_SQL)

def log_model_drift(
    model_version: str,
    feature_name: str,
    drift_score: float,
    status: str
) -> dict[str, Any]:
    """Insert a new drift monitor log."""
    ts = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO model_registry_logs
                (model_version, feature_name, drift_score, status, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (model_version, feature_name, drift_score, status, ts),
        )
    return {"status": "logged", "timestamp": ts}


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def insert_approval(
    campaign_id: str,
    admin_user: str,
    allocated_budget: float,
    notes: str = "",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Insert a new campaign approval record and return it as a dict."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO campaign_approvals
                (campaign_id, admin_user, allocated_budget, notes, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (campaign_id, admin_user, allocated_budget, notes, ts),
        )
    return {
        "approved": True,
        "campaign_id": campaign_id,
        "admin_user": admin_user,
        "allocated_budget": allocated_budget,
        "notes": notes,
        "timestamp": ts,
    }


def list_approvals() -> list[dict[str, Any]]:
    """Return all campaign approvals ordered by most recent first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT campaign_id, admin_user, allocated_budget, notes, timestamp "
            "FROM campaign_approvals ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def count_approvals() -> int:
    """Return the total number of approved campaigns (used for ID generation)."""
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM campaign_approvals").fetchone()
    return row["cnt"]


# ---------------------------------------------------------------------------
# Test helper — allow tests to use an isolated in-memory DB
# ---------------------------------------------------------------------------

def override_db_path(path: pathlib.Path | str) -> None:
    """Override the DB path (for testing with temp files). Not thread-safe."""
    global _DB_PATH
    _DB_PATH = pathlib.Path(path)
    init_db()


# Auto-initialize on import (helps with TestClient tests)
init_db()
