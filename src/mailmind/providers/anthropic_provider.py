"""Anthropic models, via forced tool use for schema-constrained output.

This provider exists mainly to keep the abstraction honest: a second real
backend with a different structured-output mechanism proves the interface is
not an OpenAI wrapper in disguise.
"""

from __future__ import annotations

import json
import os

from .base import LLM, ProviderError, RetryableError


class AnthropicLLM(LLM):
    def __init__(self, model: str, base_url: str | None = None) -> None:
        super().__init__(model, base_url)
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderError(
                "the anthropic package is not installed. Install it with: uv add anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self._client = Anthropic(api_key=api_key, base_url=base_url, max_retries=0)

    def generate_json(self, *, system: str, user: str, schema: dict, schema_name: str) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[
                    {
                        "name": schema_name,
                        "description": "Record the structured result.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": schema_name},
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001 - translated below
            raise _translate(exc) from exc

        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return json.dumps(block.input)
        raise ProviderError("model did not call the result tool")


def _translate(exc: Exception) -> ProviderError:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if name in ("APIConnectionError", "APITimeoutError", "RateLimitError", "OverloadedError"):
        return RetryableError(str(exc) or name)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return RetryableError(f"HTTP {status}: {exc}")
    return ProviderError(str(exc) or name)
