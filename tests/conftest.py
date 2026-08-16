from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mailmind.db import Email  # noqa: E402
from mailmind.providers.base import LLM, ProviderError, RetryableError  # noqa: E402


class FakeLLM(LLM):
    """A scriptable model. Records every call so tests can assert on prompts."""

    def __init__(self, responses=None, *, failures=0, retryable=False, model="fake"):
        super().__init__(model)
        self.responses = responses or {}
        self.calls: list[dict] = []
        self.failures = failures
        self.retryable = retryable

    def generate_json(self, *, system, user, schema, schema_name):
        self.calls.append({"system": system, "user": user, "schema_name": schema_name})
        if self.failures > 0:
            self.failures -= 1
            raise (RetryableError if self.retryable else ProviderError)("scripted failure")

        if schema_name == "label_map":
            return json.dumps(self.responses.get("label_map", {"mappings": []}))

        for marker, payload in self.responses.items():
            if marker != "label_map" and marker in user:
                return json.dumps(payload) if isinstance(payload, dict) else payload
        return json.dumps(default_classification())


def default_classification(**overrides):
    base = {
        "summary": "A thing happened.",
        "type": "newsletter",
        "category": "Personal",
        "importance": "low",
        "needs_action": False,
        "suggested_action": "archive",
        "confidence": 0.8,
        "reasoning_short": "looks routine",
    }
    base.update(overrides)
    return base


class FakeGmail:
    """Stands in for GmailClient: same two methods, no network."""

    def __init__(self, emails: list[Email]):
        self.emails = {e.message_id: e for e in emails}
        self.queries: list[tuple[str | None, int | None]] = []

    def search(self, query, limit=None):
        self.queries.append((query, limit))
        ids = list(self.emails)
        return ids[:limit] if limit else ids

    def fetch(self, message_id):
        return self.emails[message_id]


def make_email(message_id: str, subject: str, sender: str = "shop@example.com", **kw) -> Email:
    return Email(
        message_id=message_id,
        thread_id=kw.get("thread_id", f"t-{message_id}"),
        date=kw.get("date", "2026-08-10T09:00:00+00:00"),
        internal_date=kw.get("internal_date", 1786000000000),
        sender=sender,
        subject=subject,
        snippet=kw.get("snippet", subject),
        body=kw.get("body", f"Body of {subject}"),
        labels=kw.get("labels", ["INBOX"]),
        list_unsubscribe=kw.get("list_unsubscribe"),
    )


@pytest.fixture
def emails():
    return [
        make_email("m1", "Your order receipt", "shop@example.com"),
        make_email("m2", "Flight BA281 changed", "alerts@airline.com"),
        make_email("m3", "Weekly newsletter", "news@blog.com"),
    ]


@pytest.fixture
def gmail(emails):
    return FakeGmail(emails)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"
