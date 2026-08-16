"""Provider interface.

Everything above this layer sees one method: given a system prompt, a user
prompt, and a JSON schema, return raw JSON text. Keeping the surface this small
is what makes pointing the tool at a local model a configuration change rather
than a rewrite.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

T = TypeVar("T")


class ProviderError(Exception):
    """A call failed in a way that will not improve on its own."""


class RetryableError(ProviderError):
    """Rate limiting, a 5xx, or a transport hiccup: worth trying again."""


class LLM(ABC):
    """A model that can be asked for schema-constrained JSON."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = base_url

    @abstractmethod
    def generate_json(
        self, *, system: str, user: str, schema: dict, schema_name: str
    ) -> str:
        """Return the model's raw JSON response text, unparsed."""

    def estimate_tokens(self, text: str) -> int:
        """Rough count for --dry-run. Deliberately approximate, not billing."""
        return max(1, len(text) // 4)

    def __str__(self) -> str:
        return f"{type(self).__name__.removesuffix('LLM').lower()}:{self.model}"


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Exponential backoff with jitter over RetryableError."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            last = exc
            if attempt == attempts - 1:
                break
            delay = min(base_delay * (2**attempt), max_delay)
            sleep(delay * (0.5 + random.random() / 2))
    raise ProviderError(f"giving up after {attempts} attempts: {last}") from last
