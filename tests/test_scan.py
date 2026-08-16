from __future__ import annotations

import json

import pytest
from conftest import FakeLLM, default_classification, make_email

from mailmind import db
from mailmind.scan import ScanError, ScanOptions, run_scan


def options(db_path, **kw):
    base = dict(model="fake:test", db_path=db_path, concurrency=2, limit=None)
    base.update(kw)
    return ScanOptions(**base)


def scan(db_path, gmail, llm, **kw):
    return run_scan(options(db_path, **kw), gmail=gmail, llm=llm)


def test_scan_persists_emails_and_classifications(db_path, gmail):
    result = scan(db_path, gmail, FakeLLM())

    assert result.fetched == 3
    assert result.classified == 3
    assert result.failed == 0

    conn = db.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM email").fetchone()[0] == 3
    rows = conn.execute(
        "SELECT * FROM classification WHERE run_id = ?", (result.run_id,)
    ).fetchall()
    assert {r["status"] for r in rows} == {"ok"}
    assert all(r["raw_response"] for r in rows), "raw output is kept for debugging"
    conn.close()


def test_run_records_how_it_was_produced(db_path, gmail):
    result = scan(db_path, gmail, FakeLLM(), max_body_chars=1234)

    conn = db.connect(db_path)
    run = db.get_run(conn, result.run_id)
    assert run["model"] == "fake:test"
    assert run["max_body_chars"] == 1234
    assert run["prompt_version"] and run["prompt_text"]
    assert run["status"] == "completed"
    conn.close()


def test_second_identical_scan_calls_no_model(db_path, gmail):
    first = scan(db_path, gmail, FakeLLM())

    llm = FakeLLM()
    second = scan(db_path, gmail, llm)

    assert second.run_id == first.run_id
    assert second.classified == 0
    assert second.skipped == 3
    assert llm.calls == [], "already-persisted work must not be re-sent to the model"


def test_a_run_that_fails_everything_is_not_completed(db_path, gmail):
    result = scan(db_path, gmail, FakeLLM(failures=99))

    conn = db.connect(db_path)
    assert db.get_run(conn, result.run_id)["status"] == "failed"
    conn.close()


def test_a_run_with_some_errors_is_resumable(db_path, gmail):
    llm = FakeLLM({"Weekly newsletter": {"summary": "incomplete"}})
    first = scan(db_path, gmail, llm)
    assert first.failed == 1

    second = scan(db_path, gmail, FakeLLM())

    assert second.run_id == first.run_id
    assert second.classified == 1, "only the failed email is retried"
    assert second.skipped == 2

    conn = db.connect(db_path)
    assert db.get_run(conn, second.run_id)["status"] == "completed"
    conn.close()


def test_new_run_flag_classifies_again_without_touching_the_old_run(db_path, gmail):
    first = scan(db_path, gmail, FakeLLM())
    second = scan(db_path, gmail, FakeLLM(), new_run=True)

    assert second.run_id != first.run_id
    assert second.classified == 3

    conn = db.connect(db_path)
    for run_id in (first.run_id, second.run_id):
        n = conn.execute(
            "SELECT COUNT(*) FROM classification WHERE run_id = ? AND status = 'ok'",
            (run_id,),
        ).fetchone()[0]
        assert n == 3
    conn.close()


def test_changing_the_model_starts_a_new_run(db_path, gmail):
    first = scan(db_path, gmail, FakeLLM())
    second = scan(db_path, gmail, FakeLLM(), model="fake:other")

    assert second.run_id != first.run_id
    assert second.classified == 3


def test_failures_are_recorded_and_retried_on_rerun(db_path, gmail):
    # Every call fails, so the run stays incomplete rather than losing emails.
    failing = scan(db_path, gmail, FakeLLM(failures=99))
    assert failing.classified == 0
    assert failing.failed == 3

    conn = db.connect(db_path)
    rows = conn.execute(
        "SELECT * FROM classification WHERE run_id = ?", (failing.run_id,)
    ).fetchall()
    assert {r["status"] for r in rows} == {"error"}
    assert all(r["error"] for r in rows)
    conn.close()

    recovered = scan(db_path, gmail, FakeLLM())
    assert recovered.run_id == failing.run_id
    assert recovered.classified == 3


def test_one_bad_email_does_not_sink_the_run(db_path, gmail):
    llm = FakeLLM({"Weekly newsletter": {"summary": "incomplete"}})
    result = scan(db_path, gmail, llm)

    assert result.classified == 2
    assert result.failed == 1

    conn = db.connect(db_path)
    row = conn.execute(
        "SELECT * FROM classification WHERE run_id = ? AND status = 'error'",
        (result.run_id,),
    ).fetchone()
    assert "invalid output" in row["error"]
    assert row["attempts"] == 2, "one repair retry before giving up"
    conn.close()


def test_invalid_output_is_repaired_on_the_second_attempt(db_path):
    from conftest import FakeGmail

    gmail = FakeGmail([make_email("m1", "Order receipt")])

    class Flaky(FakeLLM):
        def generate_json(self, **kw):
            self.calls.append(kw)
            if len(self.calls) == 1:
                return json.dumps({"summary": "missing the rest"})
            return json.dumps(default_classification(type="receipt"))

    result = run_scan(options(None), gmail=gmail, llm=Flaky())
    assert result.classified == 1
    assert result.results[0]["type"] == "receipt"


def test_resume_rejects_a_run_from_different_settings(db_path, gmail):
    first = scan(db_path, gmail, FakeLLM(failures=99))

    with pytest.raises(ScanError, match="different settings"):
        scan(db_path, gmail, FakeLLM(), model="fake:other", resume=first.run_id)


def test_resume_rejects_a_completed_run(db_path, gmail):
    first = scan(db_path, gmail, FakeLLM())

    with pytest.raises(ScanError, match="already completed"):
        scan(db_path, gmail, FakeLLM(), resume=first.run_id)


def test_resume_does_not_refetch_from_gmail(db_path, gmail):
    scan(db_path, gmail, FakeLLM(failures=99))
    before = len(gmail.queries)

    scan(db_path, gmail, FakeLLM())

    assert len(gmail.queries) == before, "a resumed run classifies the stored mail"


def test_scan_without_a_database_still_classifies(gmail):
    result = run_scan(options(None), gmail=gmail, llm=FakeLLM())
    assert result.classified == 3
    assert len(result.results) == 3


def test_dry_run_costs_the_work_without_calling_the_model(db_path, gmail):
    llm = FakeLLM()
    result = run_scan(options(db_path, dry_run=True), gmail=gmail, llm=llm)

    assert result.run_id is None
    assert result.estimated_input_tokens > 0
    assert llm.calls == []

    conn = db.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM classification_run").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM email").fetchone()[0] == 3
    conn.close()


def test_dry_run_does_not_advance_the_watermark(db_path, gmail):
    run_scan(options(db_path, dry_run=True, query="in:inbox"), gmail=gmail, llm=FakeLLM())

    conn = db.connect(db_path)
    assert db.get_watermark(conn, "in:inbox") is None
    conn.close()


def test_since_last_narrows_the_query_after_a_scan(db_path, gmail):
    scan(db_path, gmail, FakeLLM(), query="in:inbox")
    scan(db_path, gmail, FakeLLM(), query="in:inbox", since_last=True, new_run=True)

    last_query = gmail.queries[-1][0]
    assert "after:" in last_query


def test_since_last_conflicts_with_after(db_path, gmail):
    with pytest.raises(ScanError, match="mutually exclusive"):
        scan(db_path, gmail, FakeLLM(), since_last=True, after="2026-08-01")


def test_no_store_body_keeps_only_the_snippet(db_path, gmail):
    result = scan(db_path, gmail, FakeLLM(), store_body=False)
    assert result.classified == 3

    conn = db.connect(db_path)
    rows = conn.execute("SELECT body, snippet FROM email").fetchall()
    assert all(r["body"] is None for r in rows)
    assert all(r["snippet"] for r in rows)
    conn.close()


def test_reuse_labels_shows_earlier_labels_to_the_model(db_path, gmail):
    llm = FakeLLM()
    scan(db_path, gmail, llm, reuse_labels=True, concurrency=1)

    assert any("used so far" in call["user"] for call in llm.calls)


def test_reuse_labels_changes_the_prompt_version(db_path, gmail):
    plain = scan(db_path, gmail, FakeLLM())
    reused = scan(db_path, gmail, FakeLLM(), reuse_labels=True)
    assert plain.run_id != reused.run_id

    conn = db.connect(db_path)
    versions = {
        db.get_run(conn, plain.run_id)["prompt_version"],
        db.get_run(conn, reused.run_id)["prompt_version"],
    }
    assert len(versions) == 2
    conn.close()


def test_bodies_are_stored_once_across_runs(db_path, gmail):
    scan(db_path, gmail, FakeLLM())
    scan(db_path, gmail, FakeLLM(), new_run=True)

    conn = db.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM email").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM classification").fetchone()[0] == 6
    conn.close()
