"""Azure OpenAI provider strategy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel

from deep_researcher.providers.base import BaseLLMProvider
from deep_researcher.providers.registry import register_provider

if TYPE_CHECKING:
    from deep_researcher.config.settings import RoleModelConfig


@register_provider("azure_openai")
class AzureOpenAIProvider(BaseLLMProvider):
    """Strategy for Azure OpenAI models via langchain-openai."""

    def validate_environment(self) -> None:
        """Validate Azure OpenAI credentials in environment."""
        missing = []
        if not os.getenv("AZURE_OPENAI_API_KEY", "").strip():
            missing.append("AZURE_OPENAI_API_KEY")
        if not os.getenv("AZURE_OPENAI_ENDPOINT", "").strip():
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not os.getenv("AZURE_OPENAI_API_VERSION", "").strip():
            missing.append("AZURE_OPENAI_API_VERSION")

        if missing:
            raise ValueError(
                f"Missing required Azure OpenAI environment variable(s): {', '.join(missing)}. "
                "Please configure them in your .env file to use the Azure OpenAI provider."
            )

    def build_model(self, config: RoleModelConfig) -> BaseChatModel:
        """Build and return an AzureChatOpenAI instance from RoleModelConfig."""
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'langchain-openai' package is required to use the Azure OpenAI provider. "
                "Install it using: pip install langchain-openai"
            ) from exc

        deployment_name = (
            config.extra_params.get("azure_deployment")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
            or config.model
        )

        kwargs: dict[str, Any] = {
            "azure_deployment": deployment_name,
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview").strip(),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
            "temperature": config.temperature,
        }

        tokens = config.max_tokens if config.max_tokens > 0 else getattr(config, "num_predict", -1)
        if tokens > 0:
            kwargs["max_tokens"] = tokens

        kwargs.update(config.extra_params)
        return AzureChatOpenAI(**kwargs)
