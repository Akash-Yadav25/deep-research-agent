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
from deep_researcher.utils.structured import extract_and_validate_json, invoke_structured

__all__ = [
    "CitationManager",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "arxiv_circuit_breaker",
    "build_chat_model",
    "extract_and_validate_json",
    "get_supervisor_llm",
    "get_worker_llm",
    "get_worker_llm_with_fallbacks",
    "invoke_structured",
    "print_step_header",
    "save_research_artifacts",
    "web_circuit_breaker",
]
