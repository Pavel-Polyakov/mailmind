"""Command line interface."""

from __future__ import annotations

import json as _json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from . import config, db, output, refine as refine_mod, report as report_mod
from .diff import DiffError, run_diff
from .gmail import GmailError
from .providers import ProviderError
from .report import ReportError, Table, VIEWS
from .scan import ScanError, ScanOptions, run_scan

app = typer.Typer(
    name="mailmind",
    help="Scan Gmail, classify with an LLM, keep the results in SQLite.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err = Console(stderr=True)

KNOWN_ERRORS = (
    ScanError,
    ReportError,
    DiffError,
    GmailError,
    ProviderError,
    refine_mod.RefineError,
    config.ConfigError,
    db.DatabaseError,
)


def _fail(exc: Exception) -> None:
    err.print(f"[red]error:[/red] {exc}")
    raise typer.Exit(1)


def _require_db(path: Optional[Path]) -> Path:
    if path is None:
        _fail(
            config.ConfigError(
                "no database given. Pass --db PATH, or set db in "
                f"{config.CONFIG_FILE}"
            )
        )
    return path


# --- auth -----------------------------------------------------------------


@app.command()
def auth(
    reauth: Annotated[bool, typer.Option("--reauth", help="Discard the cached token and re-consent.")] = False,
) -> None:
    """Authorise read-only Gmail access and cache the token."""
    from .gmail import authorize

    try:
        authorize(reauth=reauth)
    except KNOWN_ERRORS as exc:
        _fail(exc)
    console.print(f"[green]authorised[/green]  token cached at {config.TOKEN_FILE}")


# --- scan -----------------------------------------------------------------


@app.command()
def scan(
    query: Annotated[Optional[str], typer.Option("--query", "-q", help="Raw Gmail search query.")] = None,
    after: Annotated[Optional[str], typer.Option(help="Only mail after YYYY-MM-DD.")] = None,
    before: Annotated[Optional[str], typer.Option(help="Only mail before YYYY-MM-DD.")] = None,
    since_last: Annotated[bool, typer.Option("--since-last", help="Resume from the last scan of this query.")] = False,
    limit: Annotated[Optional[int], typer.Option(help="Stop after N emails.")] = None,
    model: Annotated[Optional[str], typer.Option(help="provider:model, e.g. openai:gpt-4o-mini.")] = None,
    base_url: Annotated[Optional[str], typer.Option(help="Base URL for an OpenAI-compatible server.")] = None,
    db_path: Annotated[Optional[Path], typer.Option("--db", help="SQLite database to write.")] = None,
    stdout: Annotated[bool, typer.Option("--stdout", help="Print each classification as it lands.")] = False,
    concurrency: Annotated[Optional[int], typer.Option(help="Parallel LLM calls.")] = None,
    max_body_chars: Annotated[Optional[int], typer.Option(help="Truncate bodies to this many characters.")] = None,
    no_store_body: Annotated[bool, typer.Option("--no-store-body", help="Keep subject and snippet only.")] = False,
    reuse_labels: Annotated[bool, typer.Option("--reuse-labels", help="Prefer labels already seen in this run.")] = False,
    new_run: Annotated[bool, typer.Option("--new-run", help="Always start a new run.")] = False,
    resume: Annotated[Optional[int], typer.Option(help="Resume a specific run id.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Fetch and cost the work, call no model.")] = False,
) -> None:
    """Fetch emails from Gmail and classify them."""
    try:
        cfg = config.load(
            db=db_path, model=model, base_url=base_url,
            concurrency=concurrency, max_body_chars=max_body_chars,
        )
    except KNOWN_ERRORS as exc:
        return _fail(exc)

    opts = ScanOptions(
        query=query,
        after=after,
        before=before,
        since_last=since_last,
        limit=limit,
        model=cfg.model,
        base_url=cfg.base_url,
        db_path=cfg.db,
        # Without a database there is nothing to inspect later, so printing is
        # the only useful output.
        stdout=stdout or cfg.db is None,
        concurrency=cfg.concurrency,
        max_body_chars=cfg.max_body_chars,
        store_body=not no_store_body,
        reuse_labels=reuse_labels,
        new_run=new_run,
        resume=resume,
        dry_run=dry_run,
    )

    try:
        result = run_scan(opts, reporter=_scan_reporter(opts))
    except KeyboardInterrupt:
        err.print("[yellow]interrupted[/yellow]  rerun the same command to continue")
        raise typer.Exit(130)
    except KNOWN_ERRORS as exc:
        return _fail(exc)

    if opts.dry_run or result.already_complete:
        return

    console.print(
        f"\n[bold]run {result.run_id}[/bold]  "
        f"classified {result.classified}  failed {result.failed}  "
        f"already done {result.skipped}"
    )
    if result.failed:
        console.print(
            f"[yellow]{result.failed} failed[/yellow]  "
            f"inspect with: mailmind report errors --run {result.run_id}, "
            f"retry by rerunning this command"
        )


def _scan_reporter(opts: ScanOptions):
    """Progress that stays legible during a long run."""

    def report(event: str, **info) -> None:
        match event:
            case "fetch":
                console.print(f"fetching {info['total']} emails from Gmail")
            case "already-done":
                console.print(
                    f"run {info['run_id']} already covers this with the same model, "
                    f"prompt, and parameters ({info['emails']} emails).\n"
                    f"[dim]classify again with --new-run, or inspect it with "
                    f"mailmind report type --run {info['run_id']}[/dim]"
                )
            case "start":
                verb = "resuming run" if info["resumed"] else "run"
                console.print(
                    f"{verb} {info['run_id']}: {info['todo']} to classify"
                    + (f", {info['skipped']} already done" if info["skipped"] else "")
                )
            case "dry-run":
                console.print(
                    f"[bold]dry run[/bold]  {info['emails']} emails  "
                    f"~{info['input_tokens']:,} input tokens  "
                    f"~{info['output_tokens']:,} output tokens\n"
                    f"multiply by your model's per-token price for the cost"
                )
            case "classified":
                if opts.stdout:
                    r = info["result"]
                    console.print(
                        f"[green]·[/green] [dim]{_short(info['row']['sender'])}[/dim] "
                        f"[bold]{r['type']}[/bold]/{r['category']} "
                        f"[dim]{r['importance']} {r['confidence']:.2f}[/dim]\n"
                        f"  {r['summary']}"
                    )
            case "failed":
                err.print(
                    f"[red]![/red] {_short(info['row']['subject'])}: {info['error']}"
                )
            case "interrupted":
                pass

    return report


def _short(value: object, width: int = 40) -> str:
    text = str(value or "-")
    return text if len(text) <= width else text[: width - 1] + "…"


# --- refine ---------------------------------------------------------------


@app.command()
def refine(
    run: Annotated[int, typer.Option(help="Classification run to refine.")],
    kind: Annotated[str, typer.Option(help="type, category, or both.")] = "both",
    model: Annotated[Optional[str], typer.Option(help="provider:model.")] = None,
    base_url: Annotated[Optional[str], typer.Option(help="Base URL for an OpenAI-compatible server.")] = None,
    db_path: Annotated[Optional[Path], typer.Option("--db", help="SQLite database.")] = None,
) -> None:
    """Consolidate the taxonomy a classification run discovered."""
    try:
        cfg = config.load(db=db_path, model=model, base_url=base_url)
    except KNOWN_ERRORS as exc:
        return _fail(exc)

    if kind not in ("type", "category", "both"):
        return _fail(refine_mod.RefineError(f"--kind must be type, category, or both; got {kind!r}"))
    kinds = refine_mod.KINDS if kind == "both" else (kind,)

    def report(event: str, **info) -> None:
        if event == "refining":
            console.print(f"refining {info['labels']} {info['kind']} labels")
        elif event == "empty":
            console.print(f"[dim]no {info['kind']} labels in this run[/dim]")

    try:
        outcomes = refine_mod.run_refine(
            _require_db(cfg.db),
            source_run_id=run,
            kinds=kinds,
            model=cfg.model,
            base_url=cfg.base_url,
            reporter=report,
        )
    except KNOWN_ERRORS as exc:
        return _fail(exc)

    for outcome in outcomes:
        console.print(
            f"[bold]refine run {outcome.refine_run_id}[/bold] ({outcome.kind}): "
            f"{outcome.source_labels} → {outcome.canonical_labels} labels, "
            f"{outcome.merged} merged"
        )
        if outcome.looks_like_a_no_op:
            # A model that answered with nothing used to look exactly like a
            # taxonomy that needed no work. Say which one happened.
            err.print(
                f"[yellow]warning:[/yellow] the model proposed no merges for "
                f"{outcome.source_labels} {outcome.kind} labels, so this refine run "
                f"changes nothing.\n"
                f"[dim]inspect what it actually returned:\n"
                f"  sqlite3 {cfg.db} \"SELECT raw_response FROM refine_run "
                f"WHERE id = {outcome.refine_run_id};\"\n"
                f"then try again with a stronger --model[/dim]"
            )
            continue
        console.print(
            f"[dim]apply with: mailmind report {outcome.kind} "
            f"--run {run} --refine {outcome.refine_run_id}[/dim]"
        )


# --- report ---------------------------------------------------------------


@app.command()
def report(
    view: Annotated[str, typer.Argument(help=f"One of: {', '.join(VIEWS)}.")] = "type",
    run: Annotated[Optional[int], typer.Option(help="Classification run. Defaults to the latest.")] = None,
    refine: Annotated[Optional[list[int]], typer.Option("--refine", help="Refine run(s) to apply. Repeatable.")] = None,
    fmt: Annotated[str, typer.Option("--format", help="table, json, or csv.")] = "table",
    type_filter: Annotated[Optional[str], typer.Option("--type", help="Filter the emails view by type.")] = None,
    category_filter: Annotated[Optional[str], typer.Option("--category", help="Filter the emails view by category.")] = None,
    min_confidence: Annotated[Optional[float], typer.Option("--min-confidence", help="Floor for the emails view; threshold for low-confidence.")] = None,
    limit: Annotated[Optional[int], typer.Option(help="Cap the number of rows.")] = None,
    db_path: Annotated[Optional[Path], typer.Option("--db", help="SQLite database.")] = None,
) -> None:
    """Inspect the stored dataset."""
    try:
        cfg = config.load(db=db_path)
        table = report_mod.build(
            _require_db(cfg.db),
            view=view,
            run_id=run,
            refine_ids=list(refine or []),
            type_filter=type_filter,
            category_filter=category_filter,
            min_confidence=min_confidence,
            limit=limit,
        )
        output.render(table, fmt, console)
    except KNOWN_ERRORS as exc:
        _fail(exc)
    except ValueError as exc:
        _fail(exc)


# --- diff -----------------------------------------------------------------


@app.command()
def diff(
    run: Annotated[list[int], typer.Option("--run", help="Give twice: --run 8 --run 11.")],
    examples: Annotated[int, typer.Option(help="How many disagreeing emails to show.")] = 10,
    fmt: Annotated[str, typer.Option("--format", help="table or json.")] = "table",
    db_path: Annotated[Optional[Path], typer.Option("--db", help="SQLite database.")] = None,
) -> None:
    """Compare two classification runs over the emails they share."""
    if len(run) != 2:
        return _fail(DiffError("give exactly two runs: --run 8 --run 11"))
    try:
        cfg = config.load(db=db_path)
        result = run_diff(_require_db(cfg.db), run[0], run[1], examples=examples)
    except KNOWN_ERRORS as exc:
        return _fail(exc)

    table = Table(
        f"Run {result.left_run} vs run {result.right_run} "
        f"({result.overlap} shared emails)",
        ["field", "agreement", "agreed", "differed"],
        [
            (f.field, f"{f.agreement:.0%}", f.agreed, f.total - f.agreed)
            for f in result.fields
        ],
    )
    if fmt == "json":
        sys.stdout.write(
            _json.dumps(
                {
                    "left_run": result.left_run,
                    "right_run": result.right_run,
                    "overlap": result.overlap,
                    "only_left": result.only_left,
                    "only_right": result.only_right,
                    "fields": [
                        {
                            "field": f.field,
                            "agreement": f.agreement,
                            "agreed": f.agreed,
                            "differed": f.total - f.agreed,
                            "changes": [
                                {"from": a, "to": b, "n": n}
                                for (a, b), n in f.changes.most_common(10)
                            ],
                        }
                        for f in result.fields
                    ],
                    "examples": result.examples,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        return

    output.render(table, "table", console)
    if result.only_left or result.only_right:
        console.print(
            f"[dim]only in run {result.left_run}: {result.only_left}  "
            f"only in run {result.right_run}: {result.only_right}[/dim]"
        )

    for field_diff in result.fields:
        if not field_diff.changes:
            continue
        top = ", ".join(
            f"{a} → {b} ({n})" for (a, b), n in field_diff.changes.most_common(5)
        )
        console.print(f"[bold]{field_diff.field}[/bold]: {top}")

    for example in result.examples:
        changes = ", ".join(
            f"{name}: {example['before'][name]} → {example['after'][name]}"
            for name in example["changed"]
        )
        console.print(f"[dim]{_short(example['subject'], 50)}[/dim]  {changes}")


# --- runs -----------------------------------------------------------------


@app.command()
def runs(
    fmt: Annotated[str, typer.Option("--format", help="table, json, or csv.")] = "table",
    db_path: Annotated[Optional[Path], typer.Option("--db", help="SQLite database.")] = None,
) -> None:
    """List classification and refine runs."""
    try:
        cfg = config.load(db=db_path)
        conn = db.connect(_require_db(cfg.db))
    except KNOWN_ERRORS as exc:
        return _fail(exc)

    try:
        classification = Table(
            "Classification runs",
            ["id", "status", "model", "prompt", "emails", "ok", "errors", "pending", "query", "created"],
            [
                (
                    r["id"], r["status"], r["model"], r["prompt_version"],
                    r["total"], r["ok"] or 0, r["errors"] or 0, r["pending"] or 0,
                    r["query"] or "-", r["created_at"],
                )
                for r in db.list_runs(conn)
            ],
        )
        refine_runs = Table(
            "Refine runs",
            ["id", "source run", "kind", "model", "prompt", "labels", "canonical", "created"],
            [
                (
                    r["id"], r["source_run_id"], r["kind"], r["model"],
                    r["prompt_version"], r["mapped"], r["canonical"], r["created_at"],
                )
                for r in db.list_refine_runs(conn)
            ],
        )
    finally:
        conn.close()

    match fmt:
        case "json":
            # One document, not two: `runs --format json | jq` has to work.
            sys.stdout.write(
                _json.dumps(
                    {
                        "classification_runs": _as_dicts(classification),
                        "refine_runs": _as_dicts(refine_runs),
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )
        case "csv":
            sys.stdout.write(output.to_csv(classification))
            sys.stdout.write("\n" + output.to_csv(refine_runs))
        case _:
            output.render(classification, fmt, console)
            output.render(refine_runs, fmt, console)


def _as_dicts(table: Table) -> list[dict]:
    return [dict(zip(table.columns, row)) for row in table.rows]


if __name__ == "__main__":  # pragma: no cover
    app()
