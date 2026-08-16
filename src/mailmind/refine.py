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

# Example summaries are meant to help the model tell near-duplicates apart.
# Measured against gpt-4o-mini on a real 46-label taxonomy, two trials each:
#
#   no examples     ~0.3k tokens   46 -> 21, 21 labels
#   3 examples each ~3.5k tokens   46 -> 21, 25 labels
#
# So they buy nothing measurable while multiplying prompt size by ten. This
# budget caps total example lines rather than lines per label: a short list
# still gets three each, and a long list gets none rather than tens of
# thousands of tokens of noise. It is a cost control, not a quality fix.
EXAMPLE_LINE_BUDGET = 150


class RefineError(Exception):
    pass


@dataclass
class RefineOutcome:
    kind: str
    refine_run_id: int
    source_labels: int
    canonical_labels: int
    mapping: dict[str, str]
    # How many usable mappings the model actually returned, before the rest
    # were identity-filled. Without this, a model that answered with nothing
    # is indistinguishable from a taxonomy that needed no work.
    returned: int = 0

    @property
    def merged(self) -> int:
        return sum(1 for src, dst in self.mapping.items() if src != dst)

    @property
    def looks_like_a_no_op(self) -> bool:
        return self.merged == 0 and self.source_labels > 1


def refine_prompt(kind: str) -> str:
    return prompts.REFINE_SYSTEM.format(kind=kind, style=prompts.STYLE[kind])


def examples_per_label(label_count: int) -> int:
    """How many example summaries each label can afford. See the budget above."""
    if label_count <= 0:
        return EXAMPLES_PER_LABEL
    return min(EXAMPLES_PER_LABEL, EXAMPLE_LINE_BUDGET // label_count)


def render_labels(conn: sqlite3.Connection, run_id: int, kind: str) -> tuple[str, list[str]]:
    """Label list with frequencies and example summaries, for the prompt."""
    counts = db.label_counts(conn, run_id, kind)
    per_label = examples_per_label(len(counts))
    lines: list[str] = []
    labels: list[str] = []
    for row in counts:
        label = row["label"]
        labels.append(label)
        examples = db.label_examples(conn, run_id, kind, label, per_label)
        lines.append(f"- {label} (n={row['n']})")
        lines.extend(f"    e.g. {summary}" for summary in examples)
    return "\n".join(lines), labels


def reconcile(mapping: dict[str, str], source_labels: list[str]) -> tuple[dict[str, str], int]:
    """Keep the mapping total and closed over the labels that actually exist.

    Labels the model invented are dropped; labels it omitted map to themselves,
    because the prompt asks only for labels that change. Without the identity
    fill a refined report would silently lose emails.

    Returns the mapping and how many real changes the model contributed, so a
    caller can tell "nothing needed doing" apart from "the model said nothing".
    """
    known = set(source_labels)
    clean = {
        src: dst.strip()
        for src, dst in mapping.items()
        if src in known and dst.strip() and dst.strip() != src
    }
    returned = len(clean)
    clean = _collapse_chains(clean)
    for label in source_labels:
        clean.setdefault(label, label)
    return clean, returned


def _collapse_chains(mapping: dict[str, str]) -> dict[str, str]:
    """Follow a -> b -> c down to a -> c.

    Models readily produce chains: event_invitation -> event_notification and
    event_notification -> marketing_email in the same response. Reports apply
    the mapping as a single join, so without this the two labels land in
    different buckets and the taxonomy contradicts itself.
    """
    resolved: dict[str, str] = {}
    for source in mapping:
        seen = [source]
        target = mapping[source]
        while target in mapping and target not in seen:
            seen.append(target)
            target = mapping[target]
        # A cycle (a -> b -> a) has no canonical member; the most common label
        # wins by virtue of being the one we stopped on.
        resolved[source] = target
    return resolved


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

            raw, proposed = _ask(client, system, user, schema)
            mapping, returned = reconcile(proposed, labels)

            refine_run_id = db.create_refine_run(
                conn,
                source_run_id=source_run_id,
                kind=kind,
                model=model,
                base_url=base_url,
                prompt_version=prompt_version(system, schema),
                prompt_text=system,
                raw_response=raw,
            )
            db.save_label_map(conn, refine_run_id, kind, mapping)

            outcome = RefineOutcome(
                kind=kind,
                refine_run_id=refine_run_id,
                source_labels=len(labels),
                canonical_labels=len(set(mapping.values())),
                mapping=mapping,
                returned=returned,
            )
            outcomes.append(outcome)
            say("refined", outcome=outcome)

        return outcomes
    finally:
        conn.close()


def _ask(client: LLM, system: str, user: str, schema: dict) -> tuple[str, dict[str, str]]:
    """One call, with a single repair retry on schema-invalid output.

    Returns the raw response alongside the parsed mapping; it is stored on the
    refine run so a disappointing result can be inspected rather than guessed at.
    """
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
            return raw, {m.source_label: m.canonical_label for m in parsed.mappings}
        except ValidationError as exc:
            last_error = str(exc.errors()[:2])
            if attempt == 1:
                prompt = (
                    f"{user}\n\nYour previous response was invalid: {last_error}\n"
                    f"Return corrected JSON matching the schema exactly."
                )
    raise RefineError(f"model did not return a valid label map: {last_error}")
