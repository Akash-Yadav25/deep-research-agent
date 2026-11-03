"""Factory for instantiating configured LLM models across providers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

# Import all provider modules to trigger their @register_provider decorators
import deep_researcher.providers.ollama_provider  # noqa: F401
import deep_researcher.providers.openai_provider  # noqa: F401
import deep_researcher.providers.azure_openai_provider  # noqa: F401
import deep_researcher.providers.bedrock_provider  # noqa: F401
from deep_researcher.providers.registry import get_provider

if TYPE_CHECKING:
    from deep_researcher.config.settings import RoleModelConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory to create BaseChatModel instances based on RoleModelConfig."""

    @staticmethod
    def create_model(config: RoleModelConfig) -> BaseChatModel:
        """Create a validated BaseChatModel instance for the specified role configuration.

        Args:
            config: RoleModelConfig with provider, model, temperature, and tokens.

        Returns:
            Configured BaseChatModel ready for invocation.
        """
        provider_name = config.provider.strip().lower()
        logger.info(
            f"[ModelFactory] Creating model for provider='{provider_name}', model='{config.model}'"
        )
        provider = get_provider(provider_name)
        provider.validate_environment()
        return provider.build_model(config)
