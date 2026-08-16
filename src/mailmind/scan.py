"""Fetching and classifying: the `scan` command.

Two rules shape this module:

* Emails are always classified from what is stored in the database, never from
  what happens to be in memory. A resumed run therefore sees exactly what the
  original run saw.
* A failure is a recorded row, not a lost email. Reruns retry only what is not
  already `ok`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from . import db, prompts
from .db import Email
from .identity import prompt_version, run_fingerprint
from .models import SCHEMA_VERSION, Classification, json_schema
from .providers import LLM, ProviderError, resolve, with_retry

MAX_VALIDATION_ATTEMPTS = 2
ESTIMATED_OUTPUT_TOKENS = 150


class ScanError(Exception):
    pass


@dataclass
class ScanOptions:
    query: str | None = None
    after: str | None = None
    before: str | None = None
    since_last: bool = False
    limit: int | None = None
    model: str = "openai:gpt-4o-mini"
    base_url: str | None = None
    db_path: Path | None = None
    stdout: bool = False
    concurrency: int = 8
    max_body_chars: int = 4000
    store_body: bool = True
    reuse_labels: bool = False
    new_run: bool = False
    resume: int | None = None
    dry_run: bool = False


@dataclass
class ScanResult:
    run_id: int | None
    resumed: bool = False
    already_complete: bool = False
    fetched: int = 0
    classified: int = 0
    failed: int = 0
    skipped: int = 0
    estimated_input_tokens: int = 0
    results: list[dict] = field(default_factory=list)


# --- prompt assembly ------------------------------------------------------


def classify_prompt(reuse_labels: bool) -> str:
    """The run-level system prompt. Per-email text is not part of the version."""
    text = prompts.CLASSIFY_SYSTEM
    if reuse_labels:
        text += prompts.REUSE_LABELS_HINT
    return text


def render_email(row: sqlite3.Row, known_labels: dict[str, list[str]] | None = None) -> str:
    labels = row["labels"]
    try:
        label_list = ", ".join(json.loads(labels)) if labels else ""
    except (TypeError, json.JSONDecodeError):
        label_list = str(labels or "")

    body = row["body"]
    if not body:
        # --no-store-body: the snippet is all we kept, and all we will ever use.
        body = row["snippet"] or ""

    text = prompts.CLASSIFY_USER.format(
        sender=row["sender"] or "(unknown)",
        date=row["date"] or "(unknown)",
        subject=row["subject"] or "(no subject)",
        labels=label_list or "(none)",
        list_unsubscribe=row["list_unsubscribe"] or "(none)",
        body=body,
    )
    if known_labels:
        text += (
            f"\nTypes used so far: {', '.join(known_labels['type']) or '(none)'}"
            f"\nCategories used so far: {', '.join(known_labels['category']) or '(none)'}\n"
        )
    return text


# --- one email ------------------------------------------------------------


@dataclass
class Attempt:
    message_id: str
    result: dict | None
    raw: str | None
    error: str | None
    attempts: int


def classify_email(
    llm: LLM, system: str, user: str, message_id: str, schema: dict
) -> Attempt:
    """One email, with a single repair retry on schema-invalid output."""
    raw: str | None = None
    error: str | None = None
    prompt = user

    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        try:
            raw = with_retry(
                lambda: llm.generate_json(
                    system=system, user=prompt, schema=schema, schema_name="classification"
                )
            )
        except ProviderError as exc:
            return Attempt(message_id, None, None, str(exc), attempt)

        try:
            parsed = Classification.model_validate_json(raw)
            return Attempt(message_id, parsed.model_dump(), raw, None, attempt)
        except ValidationError as exc:
            error = _summarise_validation(exc)
            if attempt < MAX_VALIDATION_ATTEMPTS:
                prompt = (
                    f"{user}\n\nYour previous response was invalid: {error}\n"
                    f"Return corrected JSON matching the schema exactly."
                )

    return Attempt(message_id, None, raw, f"invalid output: {error}", MAX_VALIDATION_ATTEMPTS)


def _summarise_validation(exc: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
        for e in exc.errors()[:4]
    ]
    return "; ".join(parts)


# --- run resolution -------------------------------------------------------


def resolve_run(
    conn: sqlite3.Connection, opts: ScanOptions, fingerprint: str, prompt_text: str, version: str
) -> tuple[int, bool]:
    """Decide whether this invocation resumes a run or starts a new one.

    Order: --resume wins, then --new-run, then an interrupted run with an
    identical fingerprint, then a fresh run.
    """
    if opts.resume is not None:
        run = db.get_run(conn, opts.resume)
        if run is None:
            raise ScanError(f"no run {opts.resume} in this database")
        if run["status"] == "completed":
            raise ScanError(
                f"run {opts.resume} is already completed; use --new-run to classify again"
            )
        if run["fingerprint"] != fingerprint:
            raise ScanError(
                f"run {opts.resume} was produced with different settings "
                f"(model, prompt, or scan parameters). Use --new-run instead."
            )
        return int(run["id"]), True

    if not opts.new_run:
        existing = db.find_resumable_run(conn, fingerprint)
        if existing is not None:
            return int(existing["id"]), True

    run_id = db.create_run(
        conn,
        fingerprint=fingerprint,
        model=opts.model,
        base_url=opts.base_url,
        prompt_version=version,
        prompt_text=prompt_text,
        schema_version=SCHEMA_VERSION,
        query=_effective_query(opts),
        limit_n=opts.limit,
        max_body_chars=opts.max_body_chars,
        reuse_labels=opts.reuse_labels,
    )
    return run_id, False


def _effective_query(opts: ScanOptions) -> str:
    from .gmail import build_query

    return build_query(opts.query, opts.after, opts.before)


# --- the command ----------------------------------------------------------


def run_scan(opts: ScanOptions, *, gmail=None, llm=None, reporter=None) -> ScanResult:
    """Fetch, classify, persist. `gmail` and `llm` are injectable for tests."""
    from .gmail import GmailClient, build_query

    say = reporter or (lambda *_args, **_kw: None)

    # Without --db everything still runs, just in memory: one code path, and
    # quick experiments stay cheap.
    path = opts.db_path or Path(":memory:")
    conn = db.connect(path) if opts.db_path else _memory_db()

    try:
        watermark_query = build_query(opts.query, opts.after, opts.before)
        since_ms = None
        if opts.since_last:
            if opts.after:
                raise ScanError("--since-last and --after are mutually exclusive")
            since_ms = db.get_watermark(conn, watermark_query)
        query = build_query(opts.query, opts.after, opts.before, since_ms)

        system = classify_prompt(opts.reuse_labels)
        schema = json_schema(Classification)
        version = prompt_version(system, schema)
        fingerprint = run_fingerprint(
            model=opts.model,
            base_url=opts.base_url,
            prompt_version=version,
            schema_version=SCHEMA_VERSION,
            query=query,
            limit=opts.limit,
            max_body_chars=opts.max_body_chars,
            reuse_labels=opts.reuse_labels,
        )

        result = ScanResult(run_id=None)

        # Resuming an unfinished run must not re-fetch: the stored mail is the
        # input, so a resumed run sees exactly what the original run saw.
        resuming_only = opts.resume is not None
        finished_match = None
        if opts.resume is None and not opts.new_run:
            match = db.find_matching_run(conn, fingerprint)
            if match is not None:
                if match["status"] == "completed":
                    finished_match = match
                else:
                    resuming_only = True

        # Identical settings over an already-completed run is redundant spend.
        # Comparing models or prompts changes the fingerprint; deliberately
        # redoing the same work is what --new-run is for.
        if finished_match is not None and not opts.dry_run:
            result.run_id = int(finished_match["id"])
            result.already_complete = True
            result.skipped = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM classification "
                    "WHERE run_id = ? AND status = 'ok'",
                    (result.run_id,),
                ).fetchone()["n"]
            )
            say("already-done", run_id=result.run_id, emails=result.skipped)
            return result

        if not resuming_only:
            client = gmail or GmailClient.connect(max_body_chars=opts.max_body_chars)
            message_ids = client.search(query, opts.limit)
            say("fetch", total=len(message_ids))
            newest = 0
            for message_id in message_ids:
                mail = client.fetch(message_id)
                db.upsert_email(conn, mail, store_body=opts.store_body)
                newest = max(newest, mail.internal_date or 0)
                result.fetched += 1
                say("fetched", message_id=message_id, subject=mail.subject)
            conn.commit()
            # A dry run must not advance the watermark, or the emails it only
            # costed would be skipped by the next --since-last.
            if newest and opts.db_path and not opts.dry_run:
                db.set_watermark(conn, watermark_query, newest)
        else:
            message_ids = []

        if opts.dry_run:
            return _dry_run(conn, opts, system, message_ids, result, say)

        run_id, resumed = resolve_run(conn, opts, fingerprint, system, version)
        result.run_id = run_id
        result.resumed = resumed
        if message_ids:
            db.seed_pending(conn, run_id, message_ids)

        todo = db.unfinished(conn, run_id)
        done_already = conn.execute(
            "SELECT COUNT(*) AS n FROM classification WHERE run_id = ? AND status = 'ok'",
            (run_id,),
        ).fetchone()["n"]
        result.skipped = int(done_already)
        say("start", run_id=run_id, resumed=resumed, todo=len(todo), skipped=result.skipped)

        if todo:
            model = llm or resolve(opts.model, opts.base_url)
            _classify_all(conn, model, system, schema, run_id, todo, opts, result, say)

        db.set_run_status(conn, run_id, _final_status(conn, run_id))
        return result
    finally:
        conn.close()


def _final_status(conn: sqlite3.Connection, run_id: int) -> str:
    """`completed` means every email in the run is classified.

    Error rows leave the run `failed`, which is still resumable: the point of
    recording failures is that a rerun can retry exactly those.
    """
    counts = conn.execute(
        """
        SELECT
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)   AS errors
        FROM classification WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if counts["pending"]:
        return "in_progress"
    return "failed" if counts["errors"] else "completed"


def _memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(db.SCHEMA)
    conn.commit()
    return conn


def _dry_run(conn, opts, system, message_ids, result, say) -> ScanResult:
    """Prepare every prompt, call nothing, and report the bill."""
    rows = [row for mid in message_ids if (row := db.get_email(conn, mid)) is not None]
    total = sum(len(system) + len(render_email(row)) for row in rows)
    result.estimated_input_tokens = total // 4
    say(
        "dry-run",
        emails=len(rows),
        input_tokens=result.estimated_input_tokens,
        output_tokens=len(rows) * ESTIMATED_OUTPUT_TOKENS,
    )
    return result


def _classify_all(conn, model, system, schema, run_id, todo, opts, result, say) -> None:
    """Workers call the model; the main thread owns every database write."""
    lock = threading.Lock()
    seen: dict[str, list[str]] = {"type": [], "category": []}
    if opts.reuse_labels:
        for kind in ("type", "category"):
            seen[kind] = [r["label"] for r in db.label_counts(conn, run_id, kind)]

    def work(row: sqlite3.Row) -> Attempt:
        known = None
        if opts.reuse_labels:
            with lock:
                known = {k: list(v) for k, v in seen.items()}
        user = render_email(row, known)
        return classify_email(model, system, user, row["message_id"], schema)

    with ThreadPoolExecutor(max_workers=max(1, opts.concurrency)) as pool:
        futures = {pool.submit(work, row): row for row in todo}
        try:
            for future in as_completed(futures):
                row = futures[future]
                attempt = future.result()
                if attempt.result is not None:
                    db.record_success(
                        conn, run_id, attempt.message_id, attempt.result, attempt.raw, attempt.attempts
                    )
                    result.classified += 1
                    result.results.append({**attempt.result, "message_id": attempt.message_id})
                    if opts.reuse_labels:
                        with lock:
                            for kind in ("type", "category"):
                                if attempt.result[kind] not in seen[kind]:
                                    seen[kind].append(attempt.result[kind])
                    say("classified", row=row, result=attempt.result)
                else:
                    db.record_error(
                        conn, run_id, attempt.message_id, attempt.error, attempt.raw, attempt.attempts
                    )
                    result.failed += 1
                    say("failed", row=row, error=attempt.error)
        except KeyboardInterrupt:
            # Leave the run in_progress so the next invocation resumes it.
            for future in futures:
                future.cancel()
            say("interrupted", run_id=run_id)
            raise
