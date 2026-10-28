"""Model construction with resilient fallback chains, retries, and circuit breakers."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

from langchain_core.runnables import RunnableWithFallbacks
from langchain_ollama import ChatOllama

from deep_researcher.config.settings import (
    SUPERVISOR_CONFIG,
    WORKER_CONFIG,
    ModelConfig,
)

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


def build_chat_model(config: ModelConfig) -> ChatOllama:
    """Instantiate a ChatOllama model client with optimized parameters."""
    kwargs: dict[str, Any] = {
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "num_ctx": config.num_ctx,
        "keep_alive": config.keep_alive,
    }
    if config.num_predict > 0:
        kwargs["num_predict"] = config.num_predict
    if config.reasoning:
        kwargs["reasoning"] = True

    return ChatOllama(**kwargs)


def get_supervisor_llm() -> ChatOllama:
    """Retrieve supervisor LLM instance (deepseek-r1:14b).
    
    Operates strictly via structured outputs for planning and writing.
    """
    return build_chat_model(SUPERVISOR_CONFIG)


def get_worker_llm() -> ChatOllama:
    """Retrieve worker LLM instance (qwen3:8b) for tool-calling execution."""
    return build_chat_model(WORKER_CONFIG)


def get_worker_llm_with_fallbacks() -> RunnableWithFallbacks:
    """Construct worker model with multi-tier fallback configuration.
    
    Primary: qwen3:8b (context 16k, temp 0.0)
    Tier 1 Fallback: qwen3:8b (expanded context 32k to handle large document scrapes)
    """
    primary = get_worker_llm()

    fallback_tier1 = ChatOllama(
        model=WORKER_CONFIG.model,
        base_url=WORKER_CONFIG.base_url,
        temperature=0.0,
        num_ctx=32768,  # Expanded context for larger payloads
        keep_alive=WORKER_CONFIG.keep_alive,
    )

    return primary.with_fallbacks([fallback_tier1])
