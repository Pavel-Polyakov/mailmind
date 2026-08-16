from __future__ import annotations

import csv
import io
import json

import pytest
from conftest import FakeGmail, FakeLLM, default_classification, make_email

from mailmind import db, output
from mailmind.refine import run_refine
from mailmind.report import ReportError, build
from mailmind.scan import ScanOptions, run_scan

MAPPING = {
    "mappings": [
        {"source_label": "order_receipt", "canonical_label": "receipt"},
        {"source_label": "purchase_confirmation", "canonical_label": "receipt"},
    ]
}


@pytest.fixture
def dataset(db_path):
    emails = [
        make_email("m1", "Order A", "shop@store.com"),
        make_email("m2", "Order B", "sales@store.com"),
        make_email("m3", "Flight moved", "ops@airline.com"),
    ]
    llm = FakeLLM(
        {
            "Order A": default_classification(
                type="order_receipt", category="Shopping", confidence=0.95
            ),
            "Order B": default_classification(
                type="purchase_confirmation", category="Shopping", confidence=0.4
            ),
            "Flight moved": default_classification(
                type="flight_change", category="Travel", importance="high",
                needs_action=True, suggested_action="inbox", confidence=0.9,
            ),
        }
    )
    run = run_scan(
        ScanOptions(model="fake:test", db_path=db_path), gmail=FakeGmail(emails), llm=llm
    )
    return db_path, run.run_id


def test_by_type_without_refinement_shows_raw_labels(dataset):
    db_path, run_id = dataset
    table = build(db_path, view="type", run_id=run_id, refine_ids=[])
    labels = {row[0] for row in table.rows}
    assert labels == {"order_receipt", "purchase_confirmation", "flight_change"}


def test_refinement_is_applied_at_query_time(dataset):
    db_path, run_id = dataset
    refine_id = run_refine(
        db_path, source_run_id=run_id, kinds=("type",), model="fake:test",
        llm=FakeLLM({"label_map": MAPPING}),
    )[0].refine_run_id

    table = build(db_path, view="type", run_id=run_id, refine_ids=[refine_id])
    counts = {row[0]: row[1] for row in table.rows}

    assert counts["receipt"] == 2
    # A label the mapping never saw still appears, unchanged.
    assert counts["flight_change"] == 1
    assert sum(counts.values()) == 3, "no email may be lost by refinement"


def test_a_mapping_made_before_new_labels_appeared_still_works(dataset):
    """The reason mappings are applied as a join instead of being baked in."""
    db_path, run_id = dataset
    refine_id = run_refine(
        db_path, source_run_id=run_id, kinds=("type",), model="fake:test",
        llm=FakeLLM({"label_map": MAPPING}),
    )[0].refine_run_id

    # A later scan discovers a type the refine run has never seen.
    later = run_scan(
        ScanOptions(model="fake:test", db_path=db_path, new_run=True),
        gmail=FakeGmail([make_email("m4", "Login code", "auth@bank.com")]),
        llm=FakeLLM({"Login code": default_classification(type="login_code")}),
    )

    table = build(db_path, view="type", run_id=later.run_id, refine_ids=[refine_id])
    assert {row[0] for row in table.rows} == {"login_code"}


def test_a_category_refine_run_does_not_rewrite_types(dataset):
    db_path, run_id = dataset
    refine_id = run_refine(
        db_path, source_run_id=run_id, kinds=("category",), model="fake:test",
        llm=FakeLLM({"label_map": {"mappings": [
            {"source_label": "Shopping", "canonical_label": "Finance"}
        ]}}),
    )[0].refine_run_id

    types = build(db_path, view="type", run_id=run_id, refine_ids=[refine_id])
    assert "order_receipt" in {row[0] for row in types.rows}

    categories = build(db_path, view="category", run_id=run_id, refine_ids=[refine_id])
    assert {row[0] for row in categories.rows} == {"Finance", "Travel"}


def test_two_refine_runs_of_the_same_kind_are_rejected(dataset):
    db_path, run_id = dataset
    ids = [
        run_refine(
            db_path, source_run_id=run_id, kinds=("type",), model="fake:test",
            llm=FakeLLM({"label_map": MAPPING}),
        )[0].refine_run_id
        for _ in range(2)
    ]
    with pytest.raises(ReportError, match="two refine runs"):
        build(db_path, view="type", run_id=run_id, refine_ids=ids)


def test_sender_view_groups_by_domain(dataset):
    db_path, run_id = dataset
    table = build(db_path, view="sender", run_id=run_id, refine_ids=[])
    counts = {row[0]: row[1] for row in table.rows}
    assert counts["store.com"] == 2
    assert counts["airline.com"] == 1


def test_needs_action_view_lists_only_actionable_mail(dataset):
    db_path, run_id = dataset
    table = build(db_path, view="needs-action", run_id=run_id, refine_ids=[])
    assert len(table.rows) == 1
    assert "airline.com" in table.rows[0][1]


def test_low_confidence_view_respects_the_threshold(dataset):
    db_path, run_id = dataset
    table = build(
        db_path, view="low-confidence", run_id=run_id, refine_ids=[], min_confidence=0.5
    )
    assert len(table.rows) == 1
    assert table.rows[0][0] == 0.4


def test_emails_view_filters_by_type(dataset):
    db_path, run_id = dataset
    table = build(
        db_path, view="emails", run_id=run_id, refine_ids=[], type_filter="flight_change"
    )
    assert len(table.rows) == 1


def test_emails_view_filters_by_refined_type(dataset):
    db_path, run_id = dataset
    refine_id = run_refine(
        db_path, source_run_id=run_id, kinds=("type",), model="fake:test",
        llm=FakeLLM({"label_map": MAPPING}),
    )[0].refine_run_id

    table = build(
        db_path, view="emails", run_id=run_id, refine_ids=[refine_id], type_filter="receipt"
    )
    assert len(table.rows) == 2


def test_report_defaults_to_the_latest_run(dataset):
    db_path, run_id = dataset
    later = run_scan(
        ScanOptions(model="fake:other", db_path=db_path),
        gmail=FakeGmail([make_email("m9", "Something else")]),
        llm=FakeLLM(),
    )
    table = build(db_path, view="type", run_id=None, refine_ids=[])
    assert sum(row[1] for row in table.rows) == 1
    assert later.run_id != run_id


def test_unknown_run_is_reported_clearly(dataset):
    db_path, _ = dataset
    with pytest.raises(ReportError, match="no run 999"):
        build(db_path, view="type", run_id=999, refine_ids=[])


def test_unknown_view_lists_the_known_ones(dataset):
    db_path, run_id = dataset
    with pytest.raises(ReportError, match="Known:"):
        build(db_path, view="nonsense", run_id=run_id, refine_ids=[])


def test_json_output_is_machine_readable(dataset):
    db_path, run_id = dataset
    table = build(db_path, view="type", run_id=run_id, refine_ids=[])
    payload = json.loads(output.to_json(table))
    assert payload["columns"][0] == "label"
    assert {row["label"] for row in payload["rows"]} == {
        "order_receipt", "purchase_confirmation", "flight_change"
    }


def test_csv_output_has_a_header_and_every_row(dataset):
    db_path, run_id = dataset
    table = build(db_path, view="type", run_id=run_id, refine_ids=[])
    rows = list(csv.reader(io.StringIO(output.to_csv(table))))
    assert rows[0] == table.columns
    assert len(rows) == len(table.rows) + 1


def test_errors_view_surfaces_failed_classifications(db_path):
    run = run_scan(
        ScanOptions(model="fake:test", db_path=db_path),
        gmail=FakeGmail([make_email("m1", "Broken")]),
        llm=FakeLLM({"Broken": {"summary": "incomplete"}}),
    )
    table = build(db_path, view="errors", run_id=run.run_id, refine_ids=[])
    assert len(table.rows) == 1
    assert "invalid output" in table.rows[0][3]
