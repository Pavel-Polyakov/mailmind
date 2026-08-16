from __future__ import annotations

import base64

import pytest

from mailmind.gmail import GmailError, build_query, parse_message


def encode(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def message(**kw) -> dict:
    return {
        "id": "m1",
        "threadId": "t1",
        "internalDate": "1786000000000",
        "snippet": "A snippet",
        "labelIds": ["INBOX", "IMPORTANT"],
        "payload": {
            "headers": [
                {"name": "From", "value": "Shop <shop@example.com>"},
                {"name": "Subject", "value": "Your receipt"},
                {"name": "Date", "value": "Mon, 10 Aug 2026 09:00:00 +0000"},
                {"name": "List-Unsubscribe", "value": "<https://x.example/u>"},
            ],
            "mimeType": "text/plain",
            "body": {"data": encode("Total: 42.00 EUR")},
        },
        **kw,
    }


def test_parse_extracts_the_fields_classification_needs():
    mail = parse_message(message(), 4000)

    assert mail.message_id == "m1"
    assert mail.thread_id == "t1"
    assert mail.sender == "Shop <shop@example.com>"
    assert mail.subject == "Your receipt"
    assert mail.snippet == "A snippet"
    assert mail.labels == ["INBOX", "IMPORTANT"]
    assert mail.list_unsubscribe == "<https://x.example/u>"
    assert "42.00 EUR" in mail.body
    assert mail.date.startswith("2026-08-10")


def test_parse_prefers_text_plain_from_a_multipart_tree():
    payload = message()
    payload["payload"] = {
        "mimeType": "multipart/alternative",
        "body": {},
        "parts": [
            {"mimeType": "text/html", "body": {"data": encode("<p>html version</p>")}},
            {"mimeType": "text/plain", "body": {"data": encode("plain version")}},
        ],
    }
    assert parse_message(payload, 4000).body == "plain version"


def test_parse_falls_back_to_html_when_there_is_no_plain_part():
    payload = message()
    payload["payload"] = {
        "mimeType": "multipart/alternative",
        "body": {},
        "parts": [
            {
                "mimeType": "multipart/related",
                "body": {},
                "parts": [{"mimeType": "text/html", "body": {"data": encode("<p>nested html</p>")}}],
            }
        ],
    }
    assert "nested html" in parse_message(payload, 4000).body


def test_parse_truncates_to_the_cap():
    payload = message()
    payload["payload"]["body"] = {"data": encode("x " * 5000)}
    body = parse_message(payload, 100).body
    assert len(body) < 200


def test_parse_survives_a_message_with_no_headers_or_body():
    mail = parse_message({"id": "m9", "payload": {}}, 4000)
    assert mail.message_id == "m9"
    assert mail.sender is None
    assert mail.body == ""


def test_parse_falls_back_to_internal_date_when_the_header_is_broken():
    payload = message()
    payload["payload"]["headers"] = [{"name": "Date", "value": "not a date"}]
    assert parse_message(payload, 4000).date.startswith("2026")


def test_build_query_merges_the_raw_query_with_dates():
    query = build_query("from:bank.com", after="2026-08-01", before="2026-08-16")
    assert query == "from:bank.com after:2026/08/01 before:2026/08/16"


def test_build_query_translates_the_watermark_to_a_unix_bound():
    query = build_query("in:inbox", since_ms=1786000000000)
    assert query == "in:inbox after:1786000001"


def test_build_query_rejects_a_non_iso_date():
    with pytest.raises(GmailError, match="YYYY-MM-DD"):
        build_query(after="16/08/2026")


def test_build_query_is_empty_when_nothing_is_given():
    assert build_query() == ""
