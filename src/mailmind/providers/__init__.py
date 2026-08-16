"""Provider registry: `provider:model` in, an LLM out."""

from __future__ import annotations

from typing import Callable

from .base import LLM, ProviderError, RetryableError, with_retry

__all__ = ["LLM", "ProviderError", "RetryableError", "with_retry", "resolve", "PROVIDERS"]


def _openai(model: str, base_url: str | None) -> LLM:
    from .openai_provider import OpenAILLM

    return OpenAILLM(model, base_url)


def _compatible(model: str, base_url: str | None) -> LLM:
    from .openai_provider import CompatibleLLM

    return CompatibleLLM(model, base_url)


def _ollama(model: str, base_url: str | None) -> LLM:
    from .openai_provider import OllamaLLM

    return OllamaLLM(model, base_url)


def _anthropic(model: str, base_url: str | None) -> LLM:
    from .anthropic_provider import AnthropicLLM

    return AnthropicLLM(model, base_url)


PROVIDERS: dict[str, Callable[[str, str | None], LLM]] = {
    "openai": _openai,
    "openai-compatible": _compatible,
    "ollama": _ollama,
    "anthropic": _anthropic,
}


def resolve(spec: str, base_url: str | None = None) -> LLM:
    """Build the model named by a `provider:model` spec.

    Imports are deferred so a missing optional SDK only matters to whoever
    actually asked for that provider.
    """
    if ":" not in spec:
        raise ProviderError(
            f"model must be given as provider:model, got {spec!r}. "
            f"Known providers: {', '.join(sorted(PROVIDERS))}"
        )
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise ProviderError(
            f"unknown provider {provider!r}. Known: {', '.join(sorted(PROVIDERS))}"
        )
    if not model.strip():
        raise ProviderError(f"no model named in {spec!r}")
    return PROVIDERS[provider](model.strip(), base_url)
