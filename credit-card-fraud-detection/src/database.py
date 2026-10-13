"""
src/database.py
================
Modular SQLite data-access layer for the demonstration prediction history.

Design notes:
  * A fresh connection is opened per operation and closed immediately
    afterwards (via context manager) rather than sharing one global
    connection across Flask requests/threads. This avoids SQLite
    thread-affinity issues under the Flask development server's threaded
    mode and under pytest's parallel-ish fixtures.
  * All SQL uses parameterized queries; no string interpolation of
    user-controlled values ever reaches a SQL statement.
  * Only derived prediction metadata is stored (timestamp, predicted class,
    label, score, source). Raw transaction feature vectors and any payment
    details are never persisted here.
  * The database path is resolved from config.DATABASE_PATH by default but
    can be overridden (e.g. by tests) by passing an explicit path, which
    keeps tests isolated from the real demo database file.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config import DATABASE_PATH

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    predicted_class INTEGER NOT NULL,
    predicted_label TEXT NOT NULL,
    score REAL,
    decision_threshold REAL,
    model_is_demo INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);
"""


@contextmanager
def get_connection(db_path: Path | str = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    """
    Open a short-lived SQLite connection as a context manager. The parent
    directory is created automatically if it does not exist (unless the
    path is the special ':memory:' identifier).
    """
    path_str = str(db_path)
    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path_str, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path | str = DATABASE_PATH) -> None:
    """Create the database file and required tables/indexes if missing."""
    with get_connection(db_path) as conn:
        with conn:  # transaction
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
    logger.info("Database initialized at %s", db_path)


def insert_prediction(
    db_path: Path | str,
    source: str,
    predicted_class: int,
    predicted_label: str,
    score: float | None,
    decision_threshold: float | None,
    model_is_demo: bool,
) -> int:
    """Insert one prediction history record and return its new row id."""
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO predictions
                    (created_at, source, predicted_class, predicted_label, score, decision_threshold, model_is_demo)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    source,
                    int(predicted_class),
                    predicted_label,
                    float(score) if score is not None else None,
                    float(decision_threshold) if decision_threshold is not None else None,
                    int(bool(model_is_demo)),
                ),
            )
            return int(cursor.lastrowid)


def get_recent_predictions(db_path: Path | str = DATABASE_PATH, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent prediction history records, newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, source, predicted_class, predicted_label,
                   score, decision_threshold, model_is_demo
            FROM predictions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def count_predictions(db_path: Path | str = DATABASE_PATH) -> int:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM predictions").fetchone()
    return int(row["total"]) if row else 0


def clear_predictions(db_path: Path | str = DATABASE_PATH) -> None:
    """Remove all prediction history rows (used by tests / demo reset)."""
    with get_connection(db_path) as conn:
        with conn:
            conn.execute("DELETE FROM predictions")
