"""Comparing two classification runs: the `diff` command.

This is what the immutability rules are for. Keeping every run intact is only
useful if you can ask what changed when you swapped the model or the prompt.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import db

FIELDS = ("type", "category", "importance", "suggested_action", "needs_action")


class DiffError(Exception):
    pass


@dataclass
class FieldDiff:
    field: str
    agreed: int
    total: int
    changes: Counter = field(default_factory=Counter)

    @property
    def agreement(self) -> float:
        return self.agreed / self.total if self.total else 0.0


@dataclass
class DiffResult:
    left_run: int
    right_run: int
    overlap: int
    only_left: int
    only_right: int
    fields: list[FieldDiff]
    examples: list[dict]


def compare(conn: sqlite3.Connection, left: int, right: int, *, examples: int = 10) -> DiffResult:
    for run_id in (left, right):
        if db.get_run(conn, run_id) is None:
            raise DiffError(f"no run {run_id} in this database")
    if left == right:
        raise DiffError("give two different runs")

    rows = conn.execute(
        f"""
        SELECT a.message_id, e.sender, e.subject,
               {', '.join(f'a.{f} AS left_{f}' for f in FIELDS)},
               {', '.join(f'b.{f} AS right_{f}' for f in FIELDS)}
        FROM classification a
        JOIN classification b ON b.message_id = a.message_id AND b.run_id = ?
        JOIN email e ON e.message_id = a.message_id
        WHERE a.run_id = ? AND a.status = 'ok' AND b.status = 'ok'
        ORDER BY e.internal_date DESC
        """,
        (right, left),
    ).fetchall()

    diffs = {f: FieldDiff(f, 0, len(rows)) for f in FIELDS}
    sample: list[dict] = []

    for row in rows:
        changed: list[str] = []
        for name in FIELDS:
            before, after = row[f"left_{name}"], row[f"right_{name}"]
            if before == after:
                diffs[name].agreed += 1
            else:
                diffs[name].changes[(before, after)] += 1
                changed.append(name)
        if changed and len(sample) < examples:
            sample.append(
                {
                    "sender": row["sender"],
                    "subject": row["subject"],
                    "changed": changed,
                    "before": {n: row[f"left_{n}"] for n in changed},
                    "after": {n: row[f"right_{n}"] for n in changed},
                }
            )

    return DiffResult(
        left_run=left,
        right_run=right,
        overlap=len(rows),
        only_left=_exclusive(conn, left, right),
        only_right=_exclusive(conn, right, left),
        fields=list(diffs.values()),
        examples=sample,
    )


def _exclusive(conn: sqlite3.Connection, run_id: int, other: int) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM classification a
        WHERE a.run_id = ? AND a.status = 'ok'
          AND NOT EXISTS (
              SELECT 1 FROM classification b
              WHERE b.run_id = ? AND b.message_id = a.message_id AND b.status = 'ok'
          )
        """,
        (run_id, other),
    ).fetchone()
    return int(row["n"])


def run_diff(db_path: Path, left: int, right: int, *, examples: int = 10) -> DiffResult:
    conn = db.connect(db_path)
    try:
        return compare(conn, left, right, examples=examples)
    finally:
        conn.close()
