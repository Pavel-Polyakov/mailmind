"""SQLite storage.

Source email data and derived LLM results live in separate tables. Bodies are
stored once per gmail message id and referenced by every run, so rerunning a
different model over the same mail costs no extra body storage.

Classification rows are seeded as `pending` when a run starts. That makes the
membership of a run explicit, so resuming means "process the rows that are not
ok yet" rather than guessing from a re-issued Gmail query.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import secure_file

SCHEMA = """
CREATE TABLE IF NOT EXISTS email (
    message_id       TEXT PRIMARY KEY,
    thread_id        TEXT,
    date             TEXT,
    internal_date    INTEGER,
    sender           TEXT,
    subject          TEXT,
    snippet          TEXT,
    body             TEXT,
    labels           TEXT,
    list_unsubscribe TEXT,
    fetched_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS classification_run (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint    TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('in_progress', 'completed', 'failed')),
    model          TEXT NOT NULL,
    base_url       TEXT,
    prompt_version TEXT NOT NULL,
    prompt_text    TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    query          TEXT,
    limit_n        INTEGER,
    max_body_chars INTEGER NOT NULL,
    reuse_labels   INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_run_fingerprint
    ON classification_run (fingerprint, status);

CREATE TABLE IF NOT EXISTS classification (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL REFERENCES classification_run (id) ON DELETE CASCADE,
    message_id       TEXT NOT NULL REFERENCES email (message_id),
    status           TEXT NOT NULL CHECK (status IN ('pending', 'ok', 'error')),
    summary          TEXT,
    type             TEXT,
    category         TEXT,
    importance       TEXT,
    needs_action     INTEGER,
    suggested_action TEXT,
    confidence       REAL,
    reasoning_short  TEXT,
    raw_response     TEXT,
    error            TEXT,
    attempts         INTEGER NOT NULL DEFAULT 0,
    updated_at       TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (run_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_classification_run_status
    ON classification (run_id, status);

CREATE TABLE IF NOT EXISTS refine_run (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_run_id  INTEGER NOT NULL REFERENCES classification_run (id) ON DELETE CASCADE,
    kind           TEXT NOT NULL CHECK (kind IN ('type', 'category')),
    model          TEXT NOT NULL,
    base_url       TEXT,
    prompt_version TEXT NOT NULL,
    prompt_text    TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS label_map (
    refine_run_id   INTEGER NOT NULL REFERENCES refine_run (id) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('type', 'category')),
    source_label    TEXT NOT NULL,
    canonical_label TEXT NOT NULL,
    PRIMARY KEY (refine_run_id, kind, source_label)
);

CREATE TABLE IF NOT EXISTS scan_watermark (
    query            TEXT PRIMARY KEY,
    last_internal_ms INTEGER NOT NULL,
    last_scanned_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class DatabaseError(Exception):
    pass


@dataclass(frozen=True)
class Email:
    message_id: str
    thread_id: str | None
    date: str | None
    internal_date: int | None
    sender: str | None
    subject: str | None
    snippet: str | None
    body: str | None
    labels: list[str]
    list_unsubscribe: str | None


def connect(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) a mailmind database.

    The file holds plaintext mail, including login codes, so it is created
    owner-only.
    """
    path = Path(path).expanduser()
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    conn = sqlite3.connect(path, timeout=30.0)
    if new:
        secure_file(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# --- emails ---------------------------------------------------------------


def upsert_email(conn: sqlite3.Connection, email: Email, *, store_body: bool = True) -> None:
    conn.execute(
        """
        INSERT INTO email (message_id, thread_id, date, internal_date, sender,
                           subject, snippet, body, labels, list_unsubscribe)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (message_id) DO UPDATE SET
            thread_id = excluded.thread_id,
            date = excluded.date,
            internal_date = excluded.internal_date,
            sender = excluded.sender,
            subject = excluded.subject,
            snippet = excluded.snippet,
            body = COALESCE(excluded.body, email.body),
            labels = excluded.labels,
            list_unsubscribe = excluded.list_unsubscribe
        """,
        (
            email.message_id,
            email.thread_id,
            email.date,
            email.internal_date,
            email.sender,
            email.subject,
            email.snippet,
            email.body if store_body else None,
            json.dumps(email.labels),
            email.list_unsubscribe,
        ),
    )


def get_email(conn: sqlite3.Connection, message_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM email WHERE message_id = ?", (message_id,)
    ).fetchone()


# --- runs -----------------------------------------------------------------


def create_run(
    conn: sqlite3.Connection,
    *,
    fingerprint: str,
    model: str,
    base_url: str | None,
    prompt_version: str,
    prompt_text: str,
    schema_version: str,
    query: str | None,
    limit_n: int | None,
    max_body_chars: int,
    reuse_labels: bool,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO classification_run
            (fingerprint, status, model, base_url, prompt_version, prompt_text,
             schema_version, query, limit_n, max_body_chars, reuse_labels)
        VALUES (?, 'in_progress', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fingerprint,
            model,
            base_url,
            prompt_version,
            prompt_text,
            schema_version,
            query,
            limit_n,
            max_body_chars,
            int(reuse_labels),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def find_resumable_run(conn: sqlite3.Connection, fingerprint: str) -> sqlite3.Row | None:
    """Most recent unfinished run with an identical fingerprint, if any.

    `failed` counts as unfinished: a run that ended with error rows still has
    work left, and rerunning is how those errors get retried.
    """
    return conn.execute(
        """
        SELECT * FROM classification_run
        WHERE fingerprint = ? AND status IN ('in_progress', 'failed')
        ORDER BY id DESC LIMIT 1
        """,
        (fingerprint,),
    ).fetchone()


def find_matching_run(conn: sqlite3.Connection, fingerprint: str) -> sqlite3.Row | None:
    """Most recent run with an identical fingerprint, finished or not."""
    return conn.execute(
        """
        SELECT * FROM classification_run
        WHERE fingerprint = ?
        ORDER BY id DESC LIMIT 1
        """,
        (fingerprint,),
    ).fetchone()


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM classification_run WHERE id = ?", (run_id,)
    ).fetchone()


def set_run_status(conn: sqlite3.Connection, run_id: int, status: str) -> None:
    completed = "datetime('now')" if status != "in_progress" else "NULL"
    conn.execute(
        f"UPDATE classification_run SET status = ?, completed_at = {completed} WHERE id = ?",
        (status, run_id),
    )
    conn.commit()


def list_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT r.id, r.status, r.model, r.prompt_version, r.query, r.created_at,
               COUNT(c.id)                                          AS total,
               SUM(CASE WHEN c.status = 'ok' THEN 1 ELSE 0 END)     AS ok,
               SUM(CASE WHEN c.status = 'error' THEN 1 ELSE 0 END)  AS errors,
               SUM(CASE WHEN c.status = 'pending' THEN 1 ELSE 0 END) AS pending
        FROM classification_run r
        LEFT JOIN classification c ON c.run_id = r.id
        GROUP BY r.id
        ORDER BY r.id
        """
    ).fetchall()


def list_refine_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT f.id, f.source_run_id, f.kind, f.model, f.prompt_version, f.created_at,
               COUNT(m.source_label)                    AS mapped,
               COUNT(DISTINCT m.canonical_label)        AS canonical
        FROM refine_run f
        LEFT JOIN label_map m ON m.refine_run_id = f.id
        GROUP BY f.id
        ORDER BY f.id
        """
    ).fetchall()


# --- classifications ------------------------------------------------------


def seed_pending(conn: sqlite3.Connection, run_id: int, message_ids: Iterable[str]) -> int:
    """Register the run's membership. Existing rows are left untouched."""
    rows = [(run_id, mid) for mid in message_ids]
    conn.executemany(
        """
        INSERT INTO classification (run_id, message_id, status)
        VALUES (?, ?, 'pending')
        ON CONFLICT (run_id, message_id) DO NOTHING
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def unfinished(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """Emails in the run that are not successfully classified yet."""
    return conn.execute(
        """
        SELECT e.*, c.attempts
        FROM classification c
        JOIN email e ON e.message_id = c.message_id
        WHERE c.run_id = ? AND c.status != 'ok'
        ORDER BY e.internal_date DESC
        """,
        (run_id,),
    ).fetchall()


def record_success(
    conn: sqlite3.Connection,
    run_id: int,
    message_id: str,
    result: dict[str, Any],
    raw_response: str,
    attempts: int,
) -> None:
    conn.execute(
        """
        UPDATE classification SET
            status = 'ok', summary = ?, type = ?, category = ?, importance = ?,
            needs_action = ?, suggested_action = ?, confidence = ?,
            reasoning_short = ?, raw_response = ?, error = NULL,
            attempts = ?, updated_at = datetime('now')
        WHERE run_id = ? AND message_id = ?
        """,
        (
            result["summary"],
            result["type"],
            result["category"],
            result["importance"],
            int(result["needs_action"]),
            result["suggested_action"],
            result["confidence"],
            result["reasoning_short"],
            raw_response,
            attempts,
            run_id,
            message_id,
        ),
    )
    conn.commit()


def record_error(
    conn: sqlite3.Connection,
    run_id: int,
    message_id: str,
    error: str,
    raw_response: str | None,
    attempts: int,
) -> None:
    """Failures are recorded, never dropped, so a rerun can target them."""
    conn.execute(
        """
        UPDATE classification SET
            status = 'error', error = ?, raw_response = ?,
            attempts = ?, updated_at = datetime('now')
        WHERE run_id = ? AND message_id = ?
        """,
        (error, raw_response, attempts, run_id, message_id),
    )
    conn.commit()


def label_counts(conn: sqlite3.Connection, run_id: int, kind: str) -> list[sqlite3.Row]:
    column = _label_column(kind)
    return conn.execute(
        f"""
        SELECT {column} AS label, COUNT(*) AS n
        FROM classification
        WHERE run_id = ? AND status = 'ok' AND {column} IS NOT NULL
        GROUP BY label
        ORDER BY n DESC, label
        """,
        (run_id,),
    ).fetchall()


def label_examples(
    conn: sqlite3.Connection, run_id: int, kind: str, label: str, limit: int = 3
) -> list[str]:
    column = _label_column(kind)
    rows = conn.execute(
        f"""
        SELECT summary FROM classification
        WHERE run_id = ? AND status = 'ok' AND {column} = ? AND summary IS NOT NULL
        LIMIT ?
        """,
        (run_id, label, limit),
    ).fetchall()
    return [r["summary"] for r in rows]


def _label_column(kind: str) -> str:
    if kind not in ("type", "category"):
        raise DatabaseError(f"unknown label kind: {kind}")
    return kind


# --- refine ---------------------------------------------------------------


def create_refine_run(
    conn: sqlite3.Connection,
    *,
    source_run_id: int,
    kind: str,
    model: str,
    base_url: str | None,
    prompt_version: str,
    prompt_text: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO refine_run
            (source_run_id, kind, model, base_url, prompt_version, prompt_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_run_id, kind, model, base_url, prompt_version, prompt_text),
    )
    conn.commit()
    return int(cur.lastrowid)


def save_label_map(
    conn: sqlite3.Connection, refine_run_id: int, kind: str, mappings: dict[str, str]
) -> None:
    conn.executemany(
        """
        INSERT INTO label_map (refine_run_id, kind, source_label, canonical_label)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (refine_run_id, kind, source_label)
        DO UPDATE SET canonical_label = excluded.canonical_label
        """,
        [(refine_run_id, kind, src, dst) for src, dst in mappings.items()],
    )
    conn.commit()


def get_refine_run(conn: sqlite3.Connection, refine_run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM refine_run WHERE id = ?", (refine_run_id,)).fetchone()


def load_label_map(conn: sqlite3.Connection, refine_run_id: int, kind: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT source_label, canonical_label FROM label_map WHERE refine_run_id = ? AND kind = ?",
        (refine_run_id, kind),
    ).fetchall()
    return {r["source_label"]: r["canonical_label"] for r in rows}


# --- watermark ------------------------------------------------------------


def get_watermark(conn: sqlite3.Connection, query: str) -> int | None:
    row = conn.execute(
        "SELECT last_internal_ms FROM scan_watermark WHERE query = ?", (query,)
    ).fetchone()
    return int(row["last_internal_ms"]) if row else None


def set_watermark(conn: sqlite3.Connection, query: str, internal_ms: int) -> None:
    conn.execute(
        """
        INSERT INTO scan_watermark (query, last_internal_ms, last_scanned_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT (query) DO UPDATE SET
            last_internal_ms = MAX(scan_watermark.last_internal_ms, excluded.last_internal_ms),
            last_scanned_at = datetime('now')
        """,
        (query, internal_ms),
    )
    conn.commit()
