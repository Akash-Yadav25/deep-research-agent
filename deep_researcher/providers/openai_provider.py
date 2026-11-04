"""OpenAI provider strategy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel

from deep_researcher.providers.base import BaseLLMProvider
from deep_researcher.providers.registry import register_provider

if TYPE_CHECKING:
    from deep_researcher.config.settings import RoleModelConfig


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """Strategy for OpenAI models via langchain-openai."""

    def validate_environment(self) -> None:
        """Validate OpenAI credentials in environment."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing or empty. "
                "Please configure OPENAI_API_KEY in your .env file to use the OpenAI provider."
            )

    def build_model(self, config: RoleModelConfig) -> BaseChatModel:
        """Build and return a ChatOpenAI instance from RoleModelConfig."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'langchain-openai' package is required to use the OpenAI provider. "
                "Install it using: pip install langchain-openai"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = (
            config.extra_params.get("base_url")
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or None
        )
        organization = (
            config.extra_params.get("organization")
            or os.getenv("OPENAI_ORGANIZATION", "").strip()
            or None
        )

        kwargs: dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
            "api_key": api_key,
        }

        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization

        tokens = config.max_tokens if config.max_tokens > 0 else getattr(config, "num_predict", -1)
        if tokens > 0:
            # Handles reasoning models (o1, o3-mini) vs standard completion models
            if config.model.startswith(("o1", "o3")):
                kwargs["max_completion_tokens"] = tokens
            else:
                kwargs["max_tokens"] = tokens

        kwargs.update(config.extra_params)
        return ChatOpenAI(**kwargs)
