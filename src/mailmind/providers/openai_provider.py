"""OpenAI and OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, ...).

Local servers vary in how much of the OpenAI API they implement. Strict
json_schema is tried first and the call degrades to json_object with the schema
inlined in the prompt, which every compatible server does support.
"""

from __future__ import annotations

import json
import os

from .base import LLM, ProviderError, RetryableError


class OpenAILLM(LLM):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        requires_key: bool = True,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(model, base_url)
        self.reasoning_effort = reasoning_effort
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderError(
                "the openai package is not installed. Install it with: uv add openai"
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            if requires_key:
                raise ProviderError(
                    f"{api_key_env} is not set. Export it, or point --base-url at a "
                    f"local server that does not need a key."
                )
            # Local servers ignore the value but the client insists on one.
            api_key = "not-needed"
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self._supports_json_schema = True

    def generate_json(self, *, system: str, user: str, schema: dict, schema_name: str) -> str:
        if self._supports_json_schema:
            try:
                return self._call(system, user, self._schema_format(schema, schema_name))
            except ProviderError as exc:
                if not _is_unsupported_format(exc):
                    raise
                # Remember, so we degrade once per run rather than once per email.
                self._supports_json_schema = False

        system = f"{system}\nRespond with JSON matching this schema:\n{json.dumps(schema)}"
        return self._call(system, user, {"type": "json_object"})

    @staticmethod
    def _schema_format(schema: dict, schema_name: str) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }

    def _call(self, system: str, user: str, response_format: dict) -> str:
        extra: dict = {}
        if self.reasoning_effort is not None:
            extra["reasoning_effort"] = self.reasoning_effort
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=response_format,
                temperature=0,
                extra_body=extra or None,
            )
        except Exception as exc:  # noqa: BLE001 - translated below
            raise _translate(exc) from exc

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ProviderError("response truncated by the model's output limit")
        content = choice.message.content
        if not content:
            raise ProviderError("model returned an empty response")
        return content


class OllamaLLM(OpenAILLM):
    """Ollama's OpenAI-compatible endpoint, with its usual local address.

    Thinking is switched off by default. Classifying an email is a
    read-and-label task, not a reasoning one, and on a local model the
    thinking tokens cost seconds per email while the schema forces the answer
    into fixed fields anyway. `reasoning_short` already carries the model's
    justification.

    Note this is `reasoning_effort`, not the `think` field: `think` belongs to
    Ollama's native /api/chat, and the OpenAI-compatible /v1 endpoint silently
    ignores fields it does not know, so `think` there would look like it
    worked while doing nothing. Set MAILMIND_REASONING_EFFORT to override
    (e.g. "low", "medium", "high", or "" to send nothing at all).
    """

    def __init__(self, model: str, base_url: str | None = None) -> None:
        effort = os.environ.get("MAILMIND_REASONING_EFFORT", "none")
        super().__init__(
            model,
            base_url or "http://localhost:11434/v1",
            api_key_env="OLLAMA_API_KEY",
            requires_key=False,
            reasoning_effort=effort or None,
        )


class CompatibleLLM(OpenAILLM):
    """Any other OpenAI-compatible server; --base-url is mandatory."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        if not base_url:
            raise ProviderError(
                "openai-compatible requires --base-url, e.g. "
                "--model openai-compatible:llama3 --base-url http://localhost:8000/v1"
            )
        super().__init__(
            model, base_url, api_key_env="OPENAI_COMPATIBLE_API_KEY", requires_key=False
        )


def _translate(exc: Exception) -> ProviderError:
    """Map SDK exceptions onto retryable / terminal."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if name in ("APIConnectionError", "APITimeoutError", "RateLimitError"):
        return RetryableError(str(exc) or name)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return RetryableError(f"HTTP {status}: {exc}")
    return ProviderError(str(exc) or name)


def _is_unsupported_format(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json_schema" in text
