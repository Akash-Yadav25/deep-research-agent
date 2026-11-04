"""Ollama local provider strategy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langchain_ollama import ChatOllama

from deep_researcher.providers.base import BaseLLMProvider
from deep_researcher.providers.registry import register_provider

if TYPE_CHECKING:
    from deep_researcher.config.settings import RoleModelConfig


@register_provider("ollama")
class OllamaProvider(BaseLLMProvider):
    """Strategy for local Ollama models via langchain-ollama."""

    def validate_environment(self) -> None:
        """Validate Ollama environment.

        Ollama runs locally; no API credentials required.
        """
        pass

    def build_model(self, config: RoleModelConfig) -> ChatOllama:
        """Build and return a ChatOllama instance from RoleModelConfig."""
        base_url = (
            config.extra_params.get("base_url")
            or getattr(config, "base_url", None)
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        keep_alive = (
            config.extra_params.get("keep_alive")
            or getattr(config, "keep_alive", None)
            or os.getenv("OLLAMA_KEEP_ALIVE", "15m")
        )

        kwargs: dict[str, Any] = {
            "model": config.model,
            "base_url": base_url,
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
            "keep_alive": keep_alive,
        }

        # num_predict in Ollama controls max output tokens
        tokens = config.max_tokens if config.max_tokens > 0 else getattr(config, "num_predict", -1)
        if tokens > 0:
            kwargs["num_predict"] = tokens

        if config.reasoning:
            kwargs["reasoning"] = True

        return ChatOllama(**kwargs)
