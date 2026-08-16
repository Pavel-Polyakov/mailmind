"""Configuration: file, defaults, and flag precedence.

Flags override the config file, which overrides the defaults below. API keys are
never read from the config file -- only from the environment.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("MAILMIND_CONFIG_DIR", Path.home() / ".config" / "mailmind"))
CONFIG_FILE = CONFIG_DIR / "config.toml"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
TOKEN_FILE = CONFIG_DIR / "token.json"

DEFAULTS = {
    "db": None,
    "model": "openai:gpt-4o-mini",
    "base_url": None,
    "concurrency": 8,
    "max_body_chars": 4000,
}

# Keys accepted in config.toml. Anything else is a typo worth reporting.
KNOWN_KEYS = frozenset(DEFAULTS)


@dataclass(frozen=True)
class Config:
    db: Path | None
    model: str
    base_url: str | None
    concurrency: int
    max_body_chars: int


class ConfigError(Exception):
    pass


def _read_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    unknown = set(data) - KNOWN_KEYS
    if unknown:
        raise ConfigError(
            f"unknown keys in {path}: {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(KNOWN_KEYS))}"
        )
    return data


def load(**overrides) -> Config:
    """Merge defaults, config file, and explicit flag overrides.

    An override of None means "flag not given", so it does not shadow the file.
    """
    values = dict(DEFAULTS)
    values.update(_read_file(CONFIG_FILE))
    values.update({k: v for k, v in overrides.items() if v is not None})

    db = values["db"]
    return Config(
        db=Path(db).expanduser() if db else None,
        model=str(values["model"]),
        base_url=values["base_url"],
        concurrency=int(values["concurrency"]),
        max_body_chars=int(values["max_body_chars"]),
    )


def ensure_config_dir() -> Path:
    """Create the config directory with owner-only permissions."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.chmod(0o700)
    return CONFIG_DIR


def secure_file(path: Path) -> None:
    """Restrict a file that holds credentials or mail content to the owner."""
    if path.exists():
        path.chmod(0o600)
