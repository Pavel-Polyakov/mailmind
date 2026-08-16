"""Rendering results.

json and csv exist because this database is meant to feed the later
rule-proposal stage; a tool that can only draw terminal tables is a dead end.
"""

from __future__ import annotations

import csv
import io
import json
import sys

from rich.console import Console
from rich.table import Table as RichTable

from .report import Table

FORMATS = ("table", "json", "csv")
MAX_CELL = 70


def render(table: Table, fmt: str, console: Console | None = None) -> None:
    match fmt:
        case "json":
            sys.stdout.write(to_json(table) + "\n")
        case "csv":
            sys.stdout.write(to_csv(table))
        case "table":
            (console or Console()).print(to_rich(table))
        case _:
            raise ValueError(f"unknown format {fmt!r}. Known: {', '.join(FORMATS)}")


def to_json(table: Table) -> str:
    payload = {
        "title": table.title,
        "columns": table.columns,
        "rows": [dict(zip(table.columns, row)) for row in table.rows],
    }
    return json.dumps(payload, indent=2, default=str)


def to_csv(table: Table) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(table.columns)
    writer.writerows(table.rows)
    return buffer.getvalue()


def to_rich(table: Table) -> RichTable:
    rich = RichTable(title=table.title, header_style="bold", title_justify="left")
    for column in table.columns:
        justify = "right" if column in ("emails", "n", "confidence", "attempts") else "left"
        rich.add_column(column, justify=justify, overflow="fold")
    if not table.rows:
        rich.add_row(*["-"] * len(table.columns))
    for row in table.rows:
        rich.add_row(*[_cell(value) for value in row])
    return rich


def _cell(value: object) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text if len(text) <= MAX_CELL else text[: MAX_CELL - 1] + "…"
