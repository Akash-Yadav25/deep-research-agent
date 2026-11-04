"""Unit tests for the LLM provider subsystem, registry, and factory."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

from deep_researcher.config.settings import RoleModelConfig
from deep_researcher.providers.factory import ModelFactory
from deep_researcher.providers.registry import (
    get_provider,
    get_provider_class,
    list_available_providers,
    register_provider,
)
from deep_researcher.providers.base import BaseLLMProvider
from deep_researcher.providers.ollama_provider import OllamaProvider
from deep_researcher.providers.openai_provider import OpenAIProvider
from deep_researcher.providers.azure_openai_provider import AzureOpenAIProvider
from deep_researcher.providers.bedrock_provider import BedrockProvider


def test_registry_contains_expected_providers():
    """Verify that all four core providers are registered."""
    providers = list_available_providers()
    assert "ollama" in providers
    assert "openai" in providers
    assert "azure_openai" in providers
    assert "bedrock" in providers


def test_custom_provider_registration():
    """Verify that the @register_provider decorator works for new providers."""
    @register_provider("mock_provider")
    class MockProvider(BaseLLMProvider):
        def validate_environment(self) -> None:
            pass

        def build_model(self, config: RoleModelConfig):
            mock = MagicMock()
            mock.model = config.model
            return mock

    assert "mock_provider" in list_available_providers()
    instance = get_provider("mock_provider")
    assert isinstance(instance, MockProvider)


def test_unknown_provider_raises_informative_error():
    """Verify that requesting an unregistered provider raises ValueError with available list."""
    with pytest.raises(ValueError) as exc_info:
        get_provider("non_existent_cloud")
    err_msg = str(exc_info.value)
    assert "Unknown LLM provider: 'non_existent_cloud'" in err_msg
    assert "Available providers:" in err_msg
    assert "ollama" in err_msg


def test_ollama_provider_build():
    """Verify OllamaProvider parameter mapping from RoleModelConfig."""
    cfg = RoleModelConfig(
        provider="ollama",
        model="deepseek-r1:14b",
        temperature=0.6,
        max_tokens=4096,
        num_ctx=32768,
        reasoning=True,
    )
    provider = OllamaProvider()
    provider.validate_environment()
    llm = provider.build_model(cfg)
    assert llm.model == "deepseek-r1:14b"
    assert llm.temperature == 0.6
    assert llm.num_ctx == 32768


def test_openai_provider_missing_key_validation():
    """Verify OpenAIProvider raises ValueError when OPENAI_API_KEY is missing."""
    provider = OpenAIProvider()
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError) as exc_info:
            provider.validate_environment()
        assert "OPENAI_API_KEY is missing" in str(exc_info.value)


def test_openai_provider_build_model_mocked():
    """Verify OpenAIProvider parameter mapping when dependencies are available."""
    mock_chat_openai_cls = MagicMock()
    mock_chat_openai_cls.return_value = MagicMock()

    cfg = RoleModelConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=2048,
    )

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-123"}):
        with patch.dict(sys.modules, {"langchain_openai": MagicMock(ChatOpenAI=mock_chat_openai_cls)}):
            provider = OpenAIProvider()
            provider.validate_environment()
            provider.build_model(cfg)

            mock_chat_openai_cls.assert_called_once_with(
                model="gpt-4o",
                temperature=0.7,
                api_key="sk-test-key-123",
                max_tokens=2048,
            )


def test_openai_reasoning_model_uses_max_completion_tokens():
    """Verify OpenAIProvider uses max_completion_tokens for o1/o3 series."""
    mock_chat_openai_cls = MagicMock()

    cfg = RoleModelConfig(
        provider="openai",
        model="o3-mini",
        temperature=1.0,
        max_tokens=8192,
    )

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-123"}):
        with patch.dict(sys.modules, {"langchain_openai": MagicMock(ChatOpenAI=mock_chat_openai_cls)}):
            provider = OpenAIProvider()
            provider.build_model(cfg)

            mock_chat_openai_cls.assert_called_once_with(
                model="o3-mini",
                temperature=1.0,
                api_key="sk-test-key-123",
                max_completion_tokens=8192,
            )


def test_azure_openai_missing_env_validation():
    """Verify AzureOpenAIProvider reports missing environment variables."""
    provider = AzureOpenAIProvider()
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            provider.validate_environment()
        err_msg = str(exc_info.value)
        assert "AZURE_OPENAI_API_KEY" in err_msg
        assert "AZURE_OPENAI_ENDPOINT" in err_msg
        assert "AZURE_OPENAI_API_VERSION" in err_msg


def test_azure_openai_build_model_mocked():
    """Verify AzureOpenAIProvider parameter mapping."""
    mock_azure_chat_cls = MagicMock()

    cfg = RoleModelConfig(
        provider="azure_openai",
        model="gpt-4o",
        temperature=0.2,
        max_tokens=4096,
    )

    azure_env = {
        "AZURE_OPENAI_API_KEY": "azure-key-xyz",
        "AZURE_OPENAI_ENDPOINT": "https://my-resource.openai.azure.com/",
        "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "my-gpt4o-deployment",
    }

    with patch.dict(os.environ, azure_env):
        with patch.dict(sys.modules, {"langchain_openai": MagicMock(AzureChatOpenAI=mock_azure_chat_cls)}):
            provider = AzureOpenAIProvider()
            provider.validate_environment()
            provider.build_model(cfg)

            mock_azure_chat_cls.assert_called_once_with(
                azure_deployment="my-gpt4o-deployment",
                azure_endpoint="https://my-resource.openai.azure.com/",
                api_version="2024-08-01-preview",
                api_key="azure-key-xyz",
                temperature=0.2,
                max_tokens=4096,
            )


def test_bedrock_missing_region_validation():
    """Verify BedrockProvider requires AWS_REGION."""
    provider = BedrockProvider()
    with patch.dict(os.environ, {"AWS_REGION": "", "AWS_DEFAULT_REGION": ""}, clear=False):
        with pytest.raises(ValueError) as exc_info:
            provider.validate_environment()
        assert "AWS_REGION" in str(exc_info.value)


def test_bedrock_build_model_mocked():
    """Verify BedrockProvider parameter mapping with ChatBedrockConverse."""
    mock_bedrock_cls = MagicMock()

    cfg = RoleModelConfig(
        provider="bedrock",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0.0,
        max_tokens=4096,
    )

    bedrock_env = {
        "AWS_REGION": "us-west-2",
        "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    }

    with patch.dict(os.environ, bedrock_env):
        with patch.dict(sys.modules, {"langchain_aws": MagicMock(ChatBedrockConverse=mock_bedrock_cls)}):
            provider = BedrockProvider()
            provider.validate_environment()
            provider.build_model(cfg)

            mock_bedrock_cls.assert_called_once_with(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                region_name="us-west-2",
                temperature=0.0,
                max_tokens=4096,
                aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            )


def test_model_factory_dispatch():
    """Verify ModelFactory creates and validates model according to RoleModelConfig."""
    cfg = RoleModelConfig(
        provider="ollama",
        model="qwen3.5:9b",
        temperature=0.0,
        max_tokens=2048,
        num_ctx=16384,
    )
    llm = ModelFactory.create_model(cfg)
    assert llm is not None
    assert llm.model == "qwen3.5:9b"
    assert llm.temperature == 0.0
