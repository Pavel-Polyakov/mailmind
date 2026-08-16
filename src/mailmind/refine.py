"""Taxonomy consolidation: the `refine` command.

A refine run produces a *mapping*, never a new classification. Source rows stay
untouched, and reports apply the mapping as a left join at query time -- so a
mapping made today still works after tomorrow's scan introduces labels it has
never seen. Unmapped labels simply pass through.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from . import db, prompts
from .identity import prompt_version
from .models import LabelMap, json_schema
from .providers import LLM, ProviderError, resolve, with_retry

KINDS = ("type", "category")
EXAMPLES_PER_LABEL = 3


class RefineError(Exception):
    pass


@dataclass
class RefineOutcome:
    kind: str
    refine_run_id: int
    source_labels: int
    canonical_labels: int
    mapping: dict[str, str]


def refine_prompt(kind: str) -> str:
    return prompts.REFINE_SYSTEM.format(kind=kind, style=prompts.STYLE[kind])


def render_labels(conn: sqlite3.Connection, run_id: int, kind: str) -> tuple[str, list[str]]:
    """Label list with frequencies and example summaries, for the prompt."""
    counts = db.label_counts(conn, run_id, kind)
    lines: list[str] = []
    labels: list[str] = []
    for row in counts:
        label = row["label"]
        labels.append(label)
        examples = db.label_examples(conn, run_id, kind, label, EXAMPLES_PER_LABEL)
        lines.append(f"- {label} (n={row['n']})")
        lines.extend(f"    e.g. {summary}" for summary in examples)
    return "\n".join(lines), labels


def reconcile(mapping: dict[str, str], source_labels: list[str]) -> dict[str, str]:
    """Keep the mapping total and closed over the labels that actually exist.

    Labels the model invented are dropped; labels it forgot map to themselves.
    Without this a refined report would silently lose emails.
    """
    known = set(source_labels)
    clean = {src: dst for src, dst in mapping.items() if src in known and dst.strip()}
    for label in source_labels:
        clean.setdefault(label, label)
    return clean


def run_refine(
    db_path: Path,
    *,
    source_run_id: int,
    kinds: tuple[str, ...] = KINDS,
    model: str,
    base_url: str | None = None,
    llm: LLM | None = None,
    reporter=None,
) -> list[RefineOutcome]:
    say = reporter or (lambda *_a, **_k: None)
    conn = db.connect(db_path)
    try:
        run = db.get_run(conn, source_run_id)
        if run is None:
            raise RefineError(f"no run {source_run_id} in this database")

        client = llm or resolve(model, base_url)
        schema = json_schema(LabelMap)
        outcomes: list[RefineOutcome] = []

        for kind in kinds:
            rendered, labels = render_labels(conn, source_run_id, kind)
            if not labels:
                say("empty", kind=kind)
                continue

            system = refine_prompt(kind)
            user = prompts.REFINE_USER.format(labels=rendered)
            say("refining", kind=kind, labels=len(labels))

            mapping = _ask(client, system, user, schema)
            mapping = reconcile(mapping, labels)

            refine_run_id = db.create_refine_run(
                conn,
                source_run_id=source_run_id,
                kind=kind,
                model=model,
                base_url=base_url,
                prompt_version=prompt_version(system, schema),
                prompt_text=system,
            )
            db.save_label_map(conn, refine_run_id, kind, mapping)

            outcome = RefineOutcome(
                kind=kind,
                refine_run_id=refine_run_id,
                source_labels=len(labels),
                canonical_labels=len(set(mapping.values())),
                mapping=mapping,
            )
            outcomes.append(outcome)
            say("refined", outcome=outcome)

        return outcomes
    finally:
        conn.close()


def _ask(client: LLM, system: str, user: str, schema: dict) -> dict[str, str]:
    """One call, with a single repair retry on schema-invalid output."""
    prompt = user
    last_error = ""
    for attempt in (1, 2):
        try:
            raw = with_retry(
                lambda: client.generate_json(
                    system=system, user=prompt, schema=schema, schema_name="label_map"
                )
            )
        except ProviderError as exc:
            raise RefineError(f"refinement failed: {exc}") from exc

        try:
            parsed = LabelMap.model_validate_json(raw)
            return {m.source_label: m.canonical_label for m in parsed.mappings}
        except ValidationError as exc:
            last_error = str(exc.errors()[:2])
            if attempt == 1:
                prompt = (
                    f"{user}\n\nYour previous response was invalid: {last_error}\n"
                    f"Return corrected JSON matching the schema exactly."
                )
    raise RefineError(f"model did not return a valid label map: {last_error}")
