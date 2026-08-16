from __future__ import annotations

import pytest
from conftest import FakeGmail, FakeLLM, default_classification, make_email

from mailmind import db
from mailmind.refine import RefineError, reconcile, run_refine
from mailmind.scan import ScanOptions, run_scan

RECEIPT_LABELS = {
    "mappings": [
        {"source_label": "order_receipt", "canonical_label": "receipt"},
        {"source_label": "purchase_confirmation", "canonical_label": "receipt"},
        {"source_label": "newsletter", "canonical_label": "newsletter"},
    ]
}


@pytest.fixture
def classified(db_path):
    """A run with three near-duplicate types, ready to consolidate."""
    emails = [
        make_email("m1", "Order A"),
        make_email("m2", "Order B"),
        make_email("m3", "Weekly digest"),
    ]
    llm = FakeLLM(
        {
            "Order A": default_classification(type="order_receipt", category="Shopping"),
            "Order B": default_classification(type="purchase_confirmation", category="Shopping"),
            "Weekly digest": default_classification(type="newsletter", category="Personal"),
        }
    )
    result = run_scan(
        ScanOptions(model="fake:test", db_path=db_path),
        gmail=FakeGmail(emails),
        llm=llm,
    )
    return result.run_id


def test_reconcile_drops_invented_labels():
    mapping = {"a": "x", "ghost": "x"}
    assert reconcile(mapping, ["a", "b"]) == {"a": "x", "b": "b"}


def test_reconcile_maps_forgotten_labels_to_themselves():
    # Otherwise a refined report would silently lose those emails.
    assert reconcile({"a": "x"}, ["a", "b", "c"]) == {"a": "x", "b": "b", "c": "c"}


def test_reconcile_ignores_blank_targets():
    assert reconcile({"a": "  "}, ["a"]) == {"a": "a"}


def test_refine_stores_a_mapping_and_leaves_sources_alone(db_path, classified):
    conn = db.connect(db_path)
    before = [dict(r) for r in conn.execute(
        "SELECT message_id, type FROM classification WHERE run_id = ? ORDER BY message_id",
        (classified,),
    )]
    conn.close()

    outcomes = run_refine(
        db_path,
        source_run_id=classified,
        kinds=("type",),
        model="fake:test",
        llm=FakeLLM({"label_map": RECEIPT_LABELS}),
    )

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.source_labels == 3
    assert outcome.canonical_labels == 2
    assert outcome.mapping["order_receipt"] == "receipt"

    conn = db.connect(db_path)
    after = [dict(r) for r in conn.execute(
        "SELECT message_id, type FROM classification WHERE run_id = ? ORDER BY message_id",
        (classified,),
    )]
    assert after == before, "refine must not rewrite classifications"

    stored = db.load_label_map(conn, outcome.refine_run_id, "type")
    assert stored["purchase_confirmation"] == "receipt"
    conn.close()


def test_each_refine_run_is_persisted_separately(db_path, classified):
    first = run_refine(
        db_path, source_run_id=classified, kinds=("type",), model="fake:a",
        llm=FakeLLM({"label_map": RECEIPT_LABELS}),
    )[0]
    second = run_refine(
        db_path, source_run_id=classified, kinds=("type",), model="fake:b",
        llm=FakeLLM({"label_map": {"mappings": []}}),
    )[0]

    assert first.refine_run_id != second.refine_run_id

    conn = db.connect(db_path)
    assert len(db.list_refine_runs(conn)) == 2
    assert db.load_label_map(conn, first.refine_run_id, "type")["order_receipt"] == "receipt"
    # An empty response still maps every label to itself.
    assert db.load_label_map(conn, second.refine_run_id, "type")["order_receipt"] == "order_receipt"
    conn.close()


def test_refine_handles_both_kinds(db_path, classified):
    outcomes = run_refine(
        db_path, source_run_id=classified, model="fake:test",
        llm=FakeLLM({"label_map": RECEIPT_LABELS}),
    )
    assert [o.kind for o in outcomes] == ["type", "category"]


def test_refine_rejects_an_unknown_run(db_path, classified):
    with pytest.raises(RefineError, match="no run 999"):
        run_refine(db_path, source_run_id=999, model="fake:test", llm=FakeLLM())


def test_refine_reports_invalid_output_rather_than_saving_junk(db_path, classified):
    llm = FakeLLM({"label_map": {"wrong_key": []}})
    with pytest.raises(RefineError, match="valid label map"):
        run_refine(db_path, source_run_id=classified, kinds=("type",), model="fake:test", llm=llm)

    conn = db.connect(db_path)
    assert db.list_refine_runs(conn) == []
    conn.close()
