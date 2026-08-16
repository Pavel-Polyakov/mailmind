"""End-to-end exercise of the real commands: scan -> runs -> refine -> report -> diff."""

from __future__ import annotations

import json

import pytest
from conftest import FakeGmail, FakeLLM, default_classification, make_email
from typer.testing import CliRunner

from mailmind import cli, gmail as gmail_mod, refine as refine_mod, scan as scan_mod

runner = CliRunner()

EMAILS = [
    make_email("m1", "Order A", "shop@store.com"),
    make_email("m2", "Order B", "sales@store.com"),
    make_email("m3", "Flight moved", "ops@airline.com"),
]

RESPONSES = {
    "Order A": default_classification(type="order_receipt", category="Shopping"),
    "Order B": default_classification(type="purchase_confirmation", category="Shopping"),
    "Flight moved": default_classification(
        type="flight_change", category="Travel", importance="high", needs_action=True,
        suggested_action="inbox", summary="Flight BA281 moved to 14:10.",
    ),
    "label_map": {
        "mappings": [
            {"source_label": "order_receipt", "canonical_label": "receipt"},
            {"source_label": "purchase_confirmation", "canonical_label": "receipt"},
        ]
    },
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No network: a fake mailbox and a scripted model behind the real CLI."""
    monkeypatch.setattr(
        gmail_mod.GmailClient, "connect", classmethod(lambda cls, **kw: FakeGmail(EMAILS))
    )
    monkeypatch.setattr(scan_mod, "resolve", lambda spec, base_url=None: FakeLLM(RESPONSES))
    monkeypatch.setattr(refine_mod, "resolve", lambda spec, base_url=None: FakeLLM(RESPONSES))


def invoke(*args):
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


def test_full_pipeline(tmp_path):
    db = str(tmp_path / "mail.db")

    # scan
    out = invoke("scan", "--db", db, "--model", "fake:test", "--limit", "3")
    assert "classified 3" in out

    # runs makes the run id discoverable, which every later command needs
    listed = json.loads(invoke("runs", "--db", db, "--format", "json"))
    assert [r["id"] for r in listed["classification_runs"]] == [1]
    assert listed["classification_runs"][0]["status"] == "completed"
    assert listed["classification_runs"][0]["ok"] == 3
    assert listed["refine_runs"] == []

    # refine
    out = invoke("refine", "--run", "1", "--kind", "type", "--db", db, "--model", "fake:test")
    assert "3 → 2 labels" in out

    # report, raw then refined
    raw = invoke("report", "type", "--run", "1", "--db", db, "--format", "json")
    labels = {row["label"] for row in json.loads(raw)["rows"]}
    assert labels == {"order_receipt", "purchase_confirmation", "flight_change"}

    refined = invoke(
        "report", "type", "--run", "1", "--refine", "1", "--db", db, "--format", "json"
    )
    rows = {row["label"]: row["emails"] for row in json.loads(refined)["rows"]}
    assert rows == {"receipt": 2, "flight_change": 1}

    # drill down into the emails behind a label
    emails = invoke(
        "report", "emails", "--run", "1", "--refine", "1", "--type", "receipt",
        "--db", db, "--format", "json",
    )
    assert len(json.loads(emails)["rows"]) == 2

    # a second run with a different model, then compare
    invoke("scan", "--db", db, "--model", "fake:other", "--limit", "3")
    out = invoke("diff", "--run", "1", "--run", "2", "--db", db)
    assert "2 shared emails" in out or "3 shared emails" in out


def test_scan_twice_does_not_reclassify(tmp_path):
    db = str(tmp_path / "mail.db")
    invoke("scan", "--db", db, "--model", "fake:test", "--limit", "3")
    out = invoke("scan", "--db", db, "--model", "fake:test", "--limit", "3")
    assert "already covers this" in out
    assert "--new-run" in out


def test_dry_run_reports_cost_and_writes_no_run(tmp_path):
    db = str(tmp_path / "mail.db")
    out = invoke("scan", "--db", db, "--model", "fake:test", "--limit", "3", "--dry-run")
    assert "dry run" in out
    assert "input tokens" in out

    listed = json.loads(invoke("runs", "--db", db, "--format", "json"))
    assert listed["classification_runs"] == []


def test_scan_without_a_db_prints_results(tmp_path, monkeypatch):
    monkeypatch.setenv("MAILMIND_CONFIG_DIR", str(tmp_path))
    out = invoke("scan", "--model", "fake:test", "--limit", "3")
    assert "flight_change" in out
    assert "Flight BA281 moved to 14:10." in out


def test_report_without_a_db_explains_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("MAILMIND_CONFIG_DIR", str(tmp_path))
    result = runner.invoke(cli.app, ["report", "type"])
    assert result.exit_code == 1
    assert "no database given" in result.output


def test_diff_needs_exactly_two_runs(tmp_path):
    db = str(tmp_path / "mail.db")
    invoke("scan", "--db", db, "--model", "fake:test", "--limit", "3")
    result = runner.invoke(cli.app, ["diff", "--run", "1", "--db", db])
    assert result.exit_code == 1
    assert "exactly two runs" in result.output


def test_unknown_report_view_is_rejected(tmp_path):
    db = str(tmp_path / "mail.db")
    invoke("scan", "--db", db, "--model", "fake:test", "--limit", "3")
    result = runner.invoke(cli.app, ["report", "nonsense", "--db", db])
    assert result.exit_code == 1
    assert "unknown view" in result.output


def test_refine_rejects_a_bad_kind(tmp_path):
    db = str(tmp_path / "mail.db")
    invoke("scan", "--db", db, "--model", "fake:test", "--limit", "3")
    result = runner.invoke(
        cli.app, ["refine", "--run", "1", "--kind", "colour", "--db", db]
    )
    assert result.exit_code == 1
    assert "must be type, category, or both" in result.output
