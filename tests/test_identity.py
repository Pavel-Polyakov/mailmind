from mailmind.identity import prompt_version, run_fingerprint
from mailmind.models import Classification, json_schema

SCHEMA = json_schema(Classification)


def fingerprint(**overrides):
    args = {
        "model": "openai:gpt-4o-mini",
        "base_url": None,
        "prompt_version": "abc123",
        "schema_version": "1",
        "query": "in:inbox",
        "limit": 100,
        "max_body_chars": 4000,
        "reuse_labels": False,
    }
    args.update(overrides)
    return run_fingerprint(**args)


def test_prompt_version_is_stable_for_identical_input():
    assert prompt_version("hello", SCHEMA) == prompt_version("hello", SCHEMA)


def test_prompt_version_changes_when_the_prompt_changes():
    assert prompt_version("hello", SCHEMA) != prompt_version("hello.", SCHEMA)


def test_prompt_version_changes_when_the_schema_changes():
    altered = {**SCHEMA, "title": "Different"}
    assert prompt_version("hello", SCHEMA) != prompt_version("hello", altered)


def test_prompt_version_ignores_schema_key_order():
    reordered = dict(reversed(list(SCHEMA.items())))
    assert prompt_version("hello", SCHEMA) == prompt_version("hello", reordered)


def test_identical_settings_share_a_fingerprint():
    assert fingerprint() == fingerprint()


def test_every_component_changes_the_fingerprint():
    baseline = fingerprint()
    for change in (
        {"model": "openai:gpt-4o"},
        {"base_url": "http://localhost:8000/v1"},
        {"prompt_version": "def456"},
        {"schema_version": "2"},
        {"query": "in:spam"},
        {"limit": 50},
        {"max_body_chars": 2000},
        {"reuse_labels": True},
    ):
        assert fingerprint(**change) != baseline, change
