"""Central registry for LLM provider strategies."""

from __future__ import annotations

import logging
from typing import Type

from deep_researcher.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_registry: dict[str, Type[BaseLLMProvider]] = {}


def register_provider(name: str):
    """Class decorator to register a provider strategy under a lowercase identifier."""
    def decorator(cls: Type[BaseLLMProvider]) -> Type[BaseLLMProvider]:
        key = name.strip().lower()
        if key in _registry:
            logger.debug(f"Overwriting provider registration for '{key}'")
        _registry[key] = cls
        return cls

    return decorator


def list_available_providers() -> list[str]:
    """Return a sorted list of registered provider identifiers."""
    return sorted(_registry.keys())


def get_provider_class(name: str) -> Type[BaseLLMProvider]:
    """Retrieve the provider strategy class for the given provider identifier.

    Args:
        name: The provider name (e.g. 'ollama', 'openai', 'azure_openai', 'bedrock').

    Returns:
        The provider strategy class.

    Raises:
        ValueError: If the provider is not registered.
    """
    key = name.strip().lower()
    if key not in _registry:
        available = ", ".join(list_available_providers())
        raise ValueError(
            f"Unknown LLM provider: '{name}'. "
            f"Available providers: [{available}]. "
            "Please check the LLM_PROVIDER or role-specific provider variable in your .env file."
        )
    return _registry[key]


def get_provider(name: str) -> BaseLLMProvider:
    """Retrieve and instantiate the provider strategy for the given provider identifier."""
    cls = get_provider_class(name)
    return cls()
