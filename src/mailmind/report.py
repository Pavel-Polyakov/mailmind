"""Reading the dataset back: the `report` command.

Refinement is applied as a left join at query time rather than baked into the
rows, so any refine run can be layered over any classification run and labels
the mapping has never seen pass through unchanged.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import db

VIEWS = (
    "type",
    "category",
    "sender",
    "action",
    "needs-action",
    "low-confidence",
    "emails",
    "errors",
)


class ReportError(Exception):
    pass


@dataclass
class Table:
    title: str
    columns: list[str]
    rows: list[tuple]


def resolve_refine(conn: sqlite3.Connection, refine_ids: list[int]) -> dict[str, int]:
    """Map each label kind to the refine run that should rewrite it."""
    chosen: dict[str, int] = {}
    for refine_id in refine_ids:
        row = db.get_refine_run(conn, refine_id)
        if row is None:
            raise ReportError(f"no refine run {refine_id} in this database")
        kind = row["kind"]
        if kind in chosen:
            raise ReportError(
                f"two refine runs given for {kind} ({chosen[kind]} and {refine_id}); pick one"
            )
        chosen[kind] = refine_id
    return chosen


def _label(kind: str, refine: dict[str, int]) -> tuple[str, str, int | None]:
    """SQL fragments that apply a mapping when one was supplied."""
    refine_id = refine.get(kind)
    expr = f"COALESCE(m_{kind}.canonical_label, c.{kind})"
    join = (
        f"LEFT JOIN label_map m_{kind} "
        f"ON m_{kind}.refine_run_id = ? AND m_{kind}.kind = '{kind}' "
        f"AND m_{kind}.source_label = c.{kind}"
    )
    return expr, join, refine_id


def by_label(conn, run_id: int, kind: str, refine: dict[str, int]) -> Table:
    expr, join, refine_id = _label(kind, refine)
    rows = conn.execute(
        f"""
        SELECT {expr} AS label,
               COUNT(*) AS n,
               SUM(c.needs_action) AS needs_action,
               ROUND(AVG(c.confidence), 2) AS avg_confidence
        FROM classification c
        {join}
        WHERE c.run_id = ? AND c.status = 'ok'
        GROUP BY label
        ORDER BY n DESC, label
        """,
        (refine_id, run_id),
    ).fetchall()
    suffix = f" (refined by run {refine_id})" if refine_id else ""
    return Table(
        f"Emails by {kind}{suffix}",
        ["label", "emails", "needs action", "avg confidence"],
        [tuple(r) for r in rows],
    )


def by_sender(conn, run_id: int, refine: dict[str, int]) -> Table:
    rows = conn.execute(
        """
        SELECT LOWER(
                   CASE WHEN INSTR(e.sender, '@') > 0
                        THEN RTRIM(SUBSTR(e.sender, INSTR(e.sender, '@') + 1), '>')
                        ELSE COALESCE(e.sender, '(unknown)') END
               ) AS domain,
               COUNT(*) AS n,
               SUM(CASE WHEN c.suggested_action IN ('delete', 'unsubscribe')
                        THEN 1 ELSE 0 END) AS disposable
        FROM classification c
        JOIN email e ON e.message_id = c.message_id
        WHERE c.run_id = ? AND c.status = 'ok'
        GROUP BY domain
        ORDER BY n DESC, domain
        """,
        (run_id,),
    ).fetchall()
    return Table("Emails by sender domain", ["domain", "emails", "delete/unsub"], [tuple(r) for r in rows])


def by_action(conn, run_id: int, refine: dict[str, int]) -> Table:
    rows = conn.execute(
        """
        SELECT c.suggested_action AS action, COUNT(*) AS n
        FROM classification c
        WHERE c.run_id = ? AND c.status = 'ok'
        GROUP BY action
        ORDER BY n DESC
        """,
        (run_id,),
    ).fetchall()
    return Table("Emails by suggested action", ["action", "emails"], [tuple(r) for r in rows])


def needs_action(conn, run_id: int, refine: dict[str, int], limit: int | None) -> Table:
    expr, join, refine_id = _label("type", refine)
    rows = conn.execute(
        f"""
        SELECT e.date, e.sender, {expr} AS type, c.importance, c.summary
        FROM classification c
        JOIN email e ON e.message_id = c.message_id
        {join}
        WHERE c.run_id = ? AND c.status = 'ok' AND c.needs_action = 1
        ORDER BY CASE c.importance
                     WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                     WHEN 'normal' THEN 2 ELSE 3 END,
                 e.internal_date DESC
        LIMIT ?
        """,
        (refine_id, run_id, limit or -1),
    ).fetchall()
    return Table(
        "Emails requiring action",
        ["date", "sender", "type", "importance", "summary"],
        [tuple(r) for r in rows],
    )


def low_confidence(conn, run_id: int, refine: dict[str, int], threshold: float, limit: int | None) -> Table:
    expr, join, refine_id = _label("type", refine)
    rows = conn.execute(
        f"""
        SELECT ROUND(c.confidence, 2) AS confidence, {expr} AS type, e.sender,
               e.subject, c.reasoning_short
        FROM classification c
        JOIN email e ON e.message_id = c.message_id
        {join}
        WHERE c.run_id = ? AND c.status = 'ok' AND c.confidence < ?
        ORDER BY c.confidence
        LIMIT ?
        """,
        (refine_id, run_id, threshold, limit or -1),
    ).fetchall()
    return Table(
        f"Classifications below confidence {threshold}",
        ["confidence", "type", "sender", "subject", "why"],
        [tuple(r) for r in rows],
    )


def emails(
    conn,
    run_id: int,
    refine: dict[str, int],
    *,
    type_filter: str | None = None,
    category_filter: str | None = None,
    min_confidence: float | None = None,
    limit: int | None = None,
) -> Table:
    type_expr, type_join, type_rid = _label("type", refine)
    cat_expr, cat_join, cat_rid = _label("category", refine)

    where = ["c.run_id = ?", "c.status = 'ok'"]
    params: list = [type_rid, cat_rid, run_id]
    if type_filter:
        where.append(f"{type_expr} = ?")
        params.append(type_filter)
    if category_filter:
        where.append(f"{cat_expr} = ?")
        params.append(category_filter)
    if min_confidence is not None:
        where.append("c.confidence >= ?")
        params.append(min_confidence)
    params.append(limit or -1)

    rows = conn.execute(
        f"""
        SELECT e.date, e.sender, e.subject, {type_expr} AS type, {cat_expr} AS category,
               c.importance, c.confidence, c.summary
        FROM classification c
        JOIN email e ON e.message_id = c.message_id
        {type_join}
        {cat_join}
        WHERE {' AND '.join(where)}
        ORDER BY e.internal_date DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return Table(
        "Emails",
        ["date", "sender", "subject", "type", "category", "importance", "confidence", "summary"],
        [tuple(r) for r in rows],
    )


def errors(conn, run_id: int) -> Table:
    rows = conn.execute(
        """
        SELECT e.sender, e.subject, c.attempts, c.error
        FROM classification c
        JOIN email e ON e.message_id = c.message_id
        WHERE c.run_id = ? AND c.status = 'error'
        ORDER BY e.internal_date DESC
        """,
        (run_id,),
    ).fetchall()
    return Table("Failed classifications", ["sender", "subject", "attempts", "error"], [tuple(r) for r in rows])


def build(
    db_path: Path,
    *,
    view: str,
    run_id: int | None,
    refine_ids: list[int],
    type_filter: str | None = None,
    category_filter: str | None = None,
    min_confidence: float | None = None,
    limit: int | None = None,
) -> Table:
    conn = db.connect(db_path)
    try:
        run_id = run_id if run_id is not None else _latest_run(conn)
        if db.get_run(conn, run_id) is None:
            raise ReportError(f"no run {run_id} in this database")
        refine = resolve_refine(conn, refine_ids)

        match view:
            case "type" | "category":
                return by_label(conn, run_id, view, refine)
            case "sender":
                return by_sender(conn, run_id, refine)
            case "action":
                return by_action(conn, run_id, refine)
            case "needs-action":
                return needs_action(conn, run_id, refine, limit)
            case "low-confidence":
                return low_confidence(conn, run_id, refine, min_confidence or 0.7, limit)
            case "emails":
                return emails(
                    conn,
                    run_id,
                    refine,
                    type_filter=type_filter,
                    category_filter=category_filter,
                    min_confidence=min_confidence,
                    limit=limit,
                )
            case "errors":
                return errors(conn, run_id)
            case _:
                raise ReportError(f"unknown view {view!r}. Known: {', '.join(VIEWS)}")
    finally:
        conn.close()


def _latest_run(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT id FROM classification_run ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise ReportError("this database has no classification runs yet")
    return int(row["id"])
