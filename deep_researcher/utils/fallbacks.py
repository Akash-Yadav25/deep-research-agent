"""Model construction with resilient fallback chains, retries, and circuit breakers."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel

from deep_researcher.config.settings import (
    SUPERVISOR_CONFIG,
    WORKER_CONFIG,
    WRITER_CONFIG,
    RoleModelConfig,
)
from deep_researcher.providers.factory import ModelFactory

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is in OPEN state."""
    pass


class CircuitBreaker:
    """Automated circuit breaker to prevent repeated cascading failures."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def record_success(self) -> None:
        """Record successful invocation, resetting failure counters."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def record_failure(self, exc: Exception) -> None:
        """Record invocation failure, potentially tripping the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.warning(
            f"CircuitBreaker '{self.name}' recorded failure ({self.failure_count}/{self.failure_threshold}): {exc}"
        )
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"CircuitBreaker '{self.name}' TRIPPED to OPEN state. Will pause requests for {self.recovery_timeout_seconds}s."
            )

    def allow_request(self) -> bool:
        """Check if request is allowed by current circuit state."""
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout_seconds:
                self.state = "HALF_OPEN"
                logger.info(f"CircuitBreaker '{self.name}' transitioning to HALF_OPEN (probe attempt).")
                return True
            return False
        if self.state == "HALF_OPEN":
            return True
        return True

    def call_with_breaker(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function under circuit breaker protection."""
        if not self.allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. Requests blocked to prevent thrashing."
            )
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure(exc)
            raise


# Global circuit breakers for research subsystems
web_circuit_breaker = CircuitBreaker("tavily_web_search", failure_threshold=4, recovery_timeout_seconds=20.0)
arxiv_circuit_breaker = CircuitBreaker("arxiv_paper_search", failure_threshold=3, recovery_timeout_seconds=30.0)


def build_chat_model(config: RoleModelConfig) -> BaseChatModel:
    """Instantiate a chat model client via the provider factory."""
    return ModelFactory.create_model(config)


def get_supervisor_llm() -> BaseChatModel:
    """Retrieve supervisor LLM instance for reasoning and structured outputs."""
    return build_chat_model(SUPERVISOR_CONFIG)


def get_worker_llm() -> BaseChatModel:
    """Retrieve worker LLM instance for tool-calling execution."""
    return build_chat_model(WORKER_CONFIG)


def get_writer_llm() -> BaseChatModel:
    """Retrieve writer LLM instance with expanded token capacity for final report synthesis."""
    return build_chat_model(WRITER_CONFIG)


def get_worker_llm_with_fallbacks() -> BaseChatModel:
    """Retrieve worker model directly. Multi-tier fallbacks deferred per design."""
    return get_worker_llm()
