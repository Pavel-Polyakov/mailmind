from __future__ import annotations

import pytest
from conftest import FakeGmail, FakeLLM, default_classification, make_email

from mailmind.diff import DiffError, run_diff
from mailmind.scan import ScanOptions, run_scan

EMAILS = [
    make_email("m1", "Order A"),
    make_email("m2", "Flight moved"),
    make_email("m3", "Weekly digest"),
]


def classify(db_path, model, responses, **kw):
    return run_scan(
        ScanOptions(model=model, db_path=db_path, **kw),
        gmail=FakeGmail(EMAILS),
        llm=FakeLLM(responses),
    ).run_id


@pytest.fixture
def two_runs(db_path):
    left = classify(
        db_path,
        "fake:a",
        {
            "Order A": default_classification(type="order_receipt", importance="low"),
            "Flight moved": default_classification(type="flight_change", importance="high"),
            "Weekly digest": default_classification(type="newsletter", importance="low"),
        },
    )
    right = classify(
        db_path,
        "fake:b",
        {
            "Order A": default_classification(type="receipt", importance="normal"),
            "Flight moved": default_classification(type="flight_change", importance="high"),
            "Weekly digest": default_classification(type="newsletter", importance="low"),
        },
    )
    return db_path, left, right


def test_diff_counts_agreement_per_field(two_runs):
    db_path, left, right = two_runs
    result = run_diff(db_path, left, right)

    assert result.overlap == 3
    fields = {f.field: f for f in result.fields}
    assert fields["type"].agreed == 2
    assert fields["type"].agreement == pytest.approx(2 / 3)
    assert fields["importance"].agreed == 2
    assert fields["category"].agreement == 1.0


def test_diff_records_what_changed_into_what(two_runs):
    db_path, left, right = two_runs
    result = run_diff(db_path, left, right)

    changes = {f.field: f.changes for f in result.fields}
    assert changes["type"][("order_receipt", "receipt")] == 1
    assert changes["importance"][("low", "normal")] == 1


def test_diff_gives_examples_of_disagreement(two_runs):
    db_path, left, right = two_runs
    result = run_diff(db_path, left, right, examples=5)

    assert len(result.examples) == 1
    example = result.examples[0]
    assert example["subject"] == "Order A"
    assert set(example["changed"]) == {"type", "importance"}
    assert example["before"]["type"] == "order_receipt"
    assert example["after"]["type"] == "receipt"


def test_diff_reports_emails_only_one_run_covered(db_path):
    left = classify(db_path, "fake:a", {})
    right = run_scan(
        ScanOptions(model="fake:b", db_path=db_path),
        gmail=FakeGmail(EMAILS[:2]),
        llm=FakeLLM(),
    ).run_id

    result = run_diff(db_path, left, right)
    assert result.overlap == 2
    assert result.only_left == 1
    assert result.only_right == 0


def test_diff_ignores_failed_classifications(db_path):
    left = classify(db_path, "fake:a", {"Weekly digest": {"summary": "broken"}})
    right = classify(db_path, "fake:b", {})

    result = run_diff(db_path, left, right)
    assert result.overlap == 2, "an email that failed in one run cannot be compared"


def test_diff_rejects_comparing_a_run_with_itself(two_runs):
    db_path, left, _ = two_runs
    with pytest.raises(DiffError, match="two different runs"):
        run_diff(db_path, left, left)


def test_diff_rejects_an_unknown_run(two_runs):
    db_path, left, _ = two_runs
    with pytest.raises(DiffError, match="no run 999"):
        run_diff(db_path, left, 999)
