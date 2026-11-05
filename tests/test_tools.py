"""Unit tests for tool wrappers, circuit breakers, and fallbacks."""

from unittest.mock import MagicMock, patch
import pytest

from deep_researcher.tools.arxiv_search import search_arxiv
from deep_researcher.tools.tavily_search import extract_webpage_content, search_web
from deep_researcher.utils.fallbacks import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    build_chat_model,
)
from deep_researcher.config.settings import ModelConfig


def test_tavily_missing_key_graceful_message():
    """Verify that search_web handles missing API key without raising an exception."""
    with patch("deep_researcher.tools.tavily_search.TAVILY_API_KEY", ""):
        result = search_web.invoke({"query": "quantum computing"})
        assert "Tavily API key is missing or not set" in result


def test_arxiv_search_mock():
    """Verify search_arxiv parsing with a mock result."""
    mock_paper = MagicMock()
    mock_paper.title = "Advances in Reasoning Models"
    mock_paper.get_short_id.return_value = "2401.99999"
    mock_paper.published.strftime.return_value = "2024-01-15"
    author_mock = MagicMock()
    author_mock.name = "Jane Doe"
    mock_paper.authors = [author_mock]
    mock_paper.categories = ["cs.AI", "cs.CL"]
    mock_paper.entry_id = "https://arxiv.org/abs/2401.99999"
    mock_paper.pdf_url = "https://arxiv.org/pdf/2401.99999.pdf"
    mock_paper.summary = "This paper explores reasoning models and reinforcement learning."

    with patch("deep_researcher.tools.arxiv_search._arxiv_client.results", return_value=[mock_paper]):
        output = search_arxiv.invoke({"query": "reasoning models", "max_results": 1})
        assert "Advances in Reasoning Models" in output
        assert "Jane Doe" in output
        assert "2401.99999" in output


def test_circuit_breaker_tripping():
    """Verify that CircuitBreaker trips after exceeding failure threshold."""
    cb = CircuitBreaker("test_breaker", failure_threshold=2, recovery_timeout_seconds=10.0)

    def failing_call():
        raise ConnectionError("Network unreachable")

    # Call 1 fails
    with pytest.raises(ConnectionError):
        cb.call_with_breaker(failing_call)
    assert cb.state == "CLOSED"

    # Call 2 fails -> trips to OPEN
    with pytest.raises(ConnectionError):
        cb.call_with_breaker(failing_call)
    assert cb.state == "OPEN"

    # Call 3 is immediately blocked by circuit breaker
    with pytest.raises(CircuitBreakerOpenError):
        cb.call_with_breaker(failing_call)


def test_build_chat_model_config():
    """Verify ChatOllama initialization with ModelConfig."""
    cfg = ModelConfig(
        model="deepseek-r1:14b",
        temperature=0.6,
        num_ctx=32768,
        num_predict=4096,
        reasoning=True,
    )
    llm = build_chat_model(cfg)
    assert llm.model == "deepseek-r1:14b"
    assert llm.temperature == 0.6


def test_writer_config_and_llm():
    """Verify WRITER_CONFIG defaults and get_writer_llm initialization."""
    from deep_researcher.config.settings import WRITER_CONFIG
    from deep_researcher.utils.fallbacks import get_writer_llm

    assert WRITER_CONFIG.max_tokens >= 8192
    assert WRITER_CONFIG.num_ctx >= 32768
    writer_llm = get_writer_llm()
    assert writer_llm is not None
    assert writer_llm.num_ctx >= 32768
