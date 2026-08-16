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
    assert reconcile(mapping, ["a", "b"])[0] == {"a": "x", "b": "b"}


def test_reconcile_maps_omitted_labels_to_themselves():
    # The prompt asks only for labels that change, so most labels are absent.
    # Without the identity fill a refined report would lose those emails.
    assert reconcile({"a": "x"}, ["a", "b", "c"])[0] == {"a": "x", "b": "b", "c": "c"}


def test_reconcile_ignores_blank_targets():
    assert reconcile({"a": "  "}, ["a"])[0] == {"a": "a"}


def test_reconcile_counts_only_real_changes():
    _, returned = reconcile({"a": "x", "b": "y"}, ["a", "b", "c"])
    assert returned == 2, "identity fills are not contributions from the model"


def test_reconcile_does_not_count_echoed_labels():
    # A model that echoes the input back has proposed nothing.
    mapping, returned = reconcile({"a": "a", "b": "b"}, ["a", "b"])
    assert returned == 0
    assert mapping == {"a": "a", "b": "b"}


def test_a_model_that_says_nothing_is_distinguishable_from_no_work_needed(db_path, classified):
    """The bug this whole change exists to fix."""
    outcome = run_refine(
        db_path, source_run_id=classified, kinds=("type",), model="fake:test",
        llm=FakeLLM({"label_map": {"mappings": []}}),
    )[0]

    assert outcome.returned == 0
    assert outcome.merged == 0
    assert outcome.looks_like_a_no_op is True
    # The mapping is still total, so no email is lost by applying it.
    assert set(outcome.mapping) == set(outcome.mapping.values())


def test_a_real_consolidation_is_not_flagged_as_a_no_op(db_path, classified):
    outcome = run_refine(
        db_path, source_run_id=classified, kinds=("type",), model="fake:test",
        llm=FakeLLM({"label_map": RECEIPT_LABELS}),
    )[0]

    assert outcome.returned == 2
    assert outcome.merged == 2
    assert outcome.looks_like_a_no_op is False


def test_raw_response_is_stored_for_debugging(db_path, classified):
    outcome = run_refine(
        db_path, source_run_id=classified, kinds=("type",), model="fake:test",
        llm=FakeLLM({"label_map": RECEIPT_LABELS}),
    )[0]

    conn = db.connect(db_path)
    raw = db.get_refine_run(conn, outcome.refine_run_id)["raw_response"]
    conn.close()
    assert "order_receipt" in raw


def test_examples_are_budgeted_by_label_count():
    """Measured: 3 examples each over 288 labels drops merges from 267 to 1."""
    from mailmind.refine import examples_per_label

    assert examples_per_label(20) == 3, "a short list can afford full examples"
    assert examples_per_label(46) == 3
    assert examples_per_label(100) == 1
    assert examples_per_label(288) == 0, "examples would drown the instruction"


def test_chains_are_collapsed_to_a_single_target():
    # Models readily emit a -> b and b -> c in one response. Reports apply the
    # mapping with a single join, so the chain has to be resolved here.
    mapping, _ = reconcile(
        {
            "event_invitation": "event_notification",
            "event_notification": "marketing_email",
            "marketing_email": "promotion",
        },
        ["event_invitation", "event_notification", "marketing_email", "promotion"],
    )
    assert mapping["event_invitation"] == "promotion"
    assert mapping["event_notification"] == "promotion"
    assert mapping["marketing_email"] == "promotion"
    assert mapping["promotion"] == "promotion"


def test_a_cycle_does_not_hang():
    mapping, _ = reconcile({"a": "b", "b": "a"}, ["a", "b"])
    assert set(mapping) == {"a", "b"}
    assert set(mapping.values()) <= {"a", "b"}


def test_no_label_maps_to_something_that_is_itself_remapped(db_path, classified):
    """The invariant that makes a refined report internally consistent."""
    outcome = run_refine(
        db_path, source_run_id=classified, kinds=("type",), model="fake:test",
        llm=FakeLLM({"label_map": {"mappings": [
            {"source_label": "order_receipt", "canonical_label": "purchase_confirmation"},
            {"source_label": "purchase_confirmation", "canonical_label": "receipt"},
        ]}}),
    )[0]

    remapped = {s for s, d in outcome.mapping.items() if s != d}
    targets = set(outcome.mapping.values())
    assert not (remapped & targets), "a canonical label must not itself be remapped"
    assert outcome.mapping["order_receipt"] == "receipt"


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
