"""AWS Bedrock provider strategy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel

from deep_researcher.providers.base import BaseLLMProvider
from deep_researcher.providers.registry import register_provider

if TYPE_CHECKING:
    from deep_researcher.config.settings import RoleModelConfig


@register_provider("bedrock")
class BedrockProvider(BaseLLMProvider):
    """Strategy for AWS Bedrock models via langchain-aws ChatBedrockConverse."""

    def validate_environment(self) -> None:
        """Validate AWS Bedrock configuration in environment."""
        region = (
            os.getenv("AWS_REGION", "").strip()
            or os.getenv("AWS_DEFAULT_REGION", "").strip()
        )
        if not region:
            raise ValueError(
                "AWS_REGION (or AWS_DEFAULT_REGION) is missing or empty. "
                "Please configure AWS_REGION in your .env file to use the AWS Bedrock provider."
            )

    def build_model(self, config: RoleModelConfig) -> BaseChatModel:
        """Build and return a ChatBedrockConverse instance from RoleModelConfig."""
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError as exc:
            raise ImportError(
                "The 'langchain-aws' and 'boto3' packages are required to use the AWS Bedrock provider. "
                "Install them using: pip install langchain-aws boto3"
            ) from exc

        region = (
            config.extra_params.get("region_name")
            or os.getenv("AWS_REGION", "").strip()
            or os.getenv("AWS_DEFAULT_REGION", "us-east-1").strip()
        )

        kwargs: dict[str, Any] = {
            "model": config.model,
            "region_name": region,
            "temperature": config.temperature,
        }

        tokens = config.max_tokens if config.max_tokens > 0 else getattr(config, "num_predict", -1)
        if tokens > 0:
            kwargs["max_tokens"] = tokens

        profile = (
            config.extra_params.get("credentials_profile_name")
            or os.getenv("AWS_PROFILE", "").strip()
        )
        if profile:
            kwargs["credentials_profile_name"] = profile

        # Explicit credentials if provided in env
        access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
        session_token = os.getenv("AWS_SESSION_TOKEN", "").strip()
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
            if session_token:
                kwargs["aws_session_token"] = session_token

        kwargs.update(config.extra_params)
        return ChatBedrockConverse(**kwargs)
