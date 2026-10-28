"""Utilities package for citations, model fallbacks, and output serialization."""

from deep_researcher.utils.citations import CitationManager
from deep_researcher.utils.fallbacks import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    arxiv_circuit_breaker,
    build_chat_model,
    get_supervisor_llm,
    get_worker_llm,
    get_worker_llm_with_fallbacks,
    web_circuit_breaker,
)
from deep_researcher.utils.output import print_step_header, save_research_artifacts

__all__ = [
    "CitationManager",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "arxiv_circuit_breaker",
    "build_chat_model",
    "get_supervisor_llm",
    "get_worker_llm",
    "get_worker_llm_with_fallbacks",
    "print_step_header",
    "save_research_artifacts",
    "web_circuit_breaker",
]
