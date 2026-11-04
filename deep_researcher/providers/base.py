"""Abstract base class for all LLM provider strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:
    from deep_researcher.config.settings import RoleModelConfig


class BaseLLMProvider(ABC):
    """Abstract strategy contract for LLM provider implementations."""

    @abstractmethod
    def validate_environment(self) -> None:
        """Validate that all required environment variables or credentials are set.

        Raises:
            ValueError: If required credentials or configurations are missing.
        """
        ...

    @abstractmethod
    def build_model(self, config: RoleModelConfig) -> BaseChatModel:
        """Instantiate and configure the BaseChatModel client.

        Args:
            config: RoleModelConfig specifying model name, temperature, tokens, etc.

        Returns:
            Configured BaseChatModel instance.
        """
        ...
