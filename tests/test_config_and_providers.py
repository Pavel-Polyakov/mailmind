from __future__ import annotations

import importlib

import pytest

from mailmind import config
from mailmind.providers import PROVIDERS, ProviderError, resolve
from mailmind.providers.base import RetryableError, with_retry


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MAILMIND_CONFIG_DIR", str(tmp_path))
    importlib.reload(config)
    yield tmp_path
    monkeypatch.delenv("MAILMIND_CONFIG_DIR", raising=False)
    importlib.reload(config)


def test_defaults_apply_when_there_is_no_config_file(config_dir):
    cfg = config.load()
    assert cfg.concurrency == 8
    assert cfg.max_body_chars == 4000
    assert cfg.db is None


def test_config_file_overrides_defaults(config_dir):
    (config_dir / "config.toml").write_text('model = "openai:gpt-4o"\nconcurrency = 3\n')
    cfg = config.load()
    assert cfg.model == "openai:gpt-4o"
    assert cfg.concurrency == 3


def test_flags_override_the_config_file(config_dir):
    (config_dir / "config.toml").write_text('model = "openai:gpt-4o"\n')
    assert config.load(model="anthropic:claude-sonnet-5").model == "anthropic:claude-sonnet-5"


def test_an_absent_flag_does_not_shadow_the_file(config_dir):
    (config_dir / "config.toml").write_text('concurrency = 3\n')
    assert config.load(concurrency=None).concurrency == 3


def test_a_typo_in_the_config_file_is_reported(config_dir):
    (config_dir / "config.toml").write_text('modle = "openai:gpt-4o"\n')
    with pytest.raises(config.ConfigError, match="unknown keys"):
        config.load()


def test_malformed_toml_is_reported(config_dir):
    (config_dir / "config.toml").write_text("model = [unclosed\n")
    with pytest.raises(config.ConfigError, match="could not read"):
        config.load()


def test_config_dir_is_owner_only(config_dir):
    path = config.ensure_config_dir()
    assert path.stat().st_mode & 0o777 == 0o700


def test_db_path_is_expanded(config_dir):
    assert not str(config.load(db="~/mail.db").db).startswith("~")


def test_every_provider_is_reachable_by_name():
    assert set(PROVIDERS) == {"openai", "openai-compatible", "ollama", "anthropic"}


def test_a_bare_model_name_is_rejected_with_guidance():
    with pytest.raises(ProviderError, match="provider:model"):
        resolve("gpt-4o-mini")


def test_an_unknown_provider_lists_the_known_ones():
    with pytest.raises(ProviderError, match="Known:"):
        resolve("acme:model-1")


def test_an_empty_model_name_is_rejected():
    with pytest.raises(ProviderError, match="no model named"):
        resolve("openai:")


def test_openai_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        resolve("openai:gpt-4o-mini")


def test_openai_compatible_requires_a_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="requires --base-url"):
        resolve("openai-compatible:llama3")


def test_retry_gives_up_and_reports_the_last_failure():
    def always_fails():
        raise RetryableError("429 rate limited")

    with pytest.raises(ProviderError, match="429 rate limited"):
        with_retry(always_fails, attempts=3, sleep=lambda _: None)


def test_retry_succeeds_once_the_transient_failure_clears():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("503")
        return "ok"

    assert with_retry(flaky, attempts=4, sleep=lambda _: None) == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_a_terminal_error():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise ProviderError("400 invalid model")

    with pytest.raises(ProviderError, match="400"):
        with_retry(bad_request, attempts=4, sleep=lambda _: None)
    assert calls["n"] == 1, "a 400 will not fix itself"


def test_retry_backs_off_between_attempts():
    delays: list[float] = []

    def always_fails():
        raise RetryableError("429")

    with pytest.raises(ProviderError):
        with_retry(always_fails, attempts=4, base_delay=1.0, sleep=delays.append)

    assert len(delays) == 3
    assert delays[0] < delays[-1], "backoff must grow"
