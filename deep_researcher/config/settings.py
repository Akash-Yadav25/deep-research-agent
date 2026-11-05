"""Central configuration for the Deep Researcher system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class RoleModelConfig:
    """Provider-agnostic configuration for a model role (supervisor or worker)."""

    model: str
    temperature: float
    num_ctx: int = 16384
    provider: str = "ollama"
    max_tokens: int = -1
    num_predict: int = -1
    keep_alive: str = "15m"
    base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    reasoning: bool = False  # Supported for reasoning models such as deepseek-r1
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_tokens == -1 and self.num_predict != -1:
            object.__setattr__(self, "max_tokens", self.num_predict)
        elif self.num_predict == -1 and self.max_tokens != -1:
            object.__setattr__(self, "num_predict", self.max_tokens)


# Backward-compatibility alias
ModelConfig = RoleModelConfig


# --- Provider & Model Configurations ---

def resolve_reasoning_default(model: str, env_val: str | None = None) -> bool:
    """Resolve reasoning mode: obey explicit env var, else enable only for reasoning architectures.

    - If env_val is non-empty: parses 'true'/'1'/'yes' vs 'false'/'0'/'no'.
    - If env_val is None or empty: returns True only for models containing 'deepseek-r1',
      'o1', or 'o3' in their names; False for general models (e.g. 'qwen3.5', 'llama3', 'gpt-4o').
    """
    if env_val is not None and env_val.strip():
        return env_val.strip().lower() in ("true", "1", "yes")
    model_lower = model.lower()
    return any(r in model_lower for r in ("deepseek-r1", "o1", "o3"))


# Global provider default: "ollama" | "openai" | "azure_openai" | "bedrock"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

_sup_model = os.getenv("SUPERVISOR_MODEL", "deepseek-r1:14b")
SUPERVISOR_CONFIG = RoleModelConfig(
    provider=os.getenv("SUPERVISOR_PROVIDER", LLM_PROVIDER).lower(),
    model=_sup_model,
    temperature=float(os.getenv("SUPERVISOR_TEMPERATURE", "0.6")),  # Optimal for reasoning/planning
    num_ctx=int(os.getenv("SUPERVISOR_NUM_CTX", "32768")),          # Expanded context for briefs/drafts
    max_tokens=int(os.getenv("SUPERVISOR_MAX_TOKENS", os.getenv("SUPERVISOR_NUM_PREDICT", "4096"))),
    reasoning=resolve_reasoning_default(_sup_model, os.getenv("SUPERVISOR_REASONING")),
)

WORKER_CONFIG = RoleModelConfig(
    provider=os.getenv("WORKER_PROVIDER", LLM_PROVIDER).lower(),
    model=os.getenv("WORKER_MODEL", "qwen3.5:9b"),
    temperature=float(os.getenv("WORKER_TEMPERATURE", "0.0")),     # Strict 0.0 for deterministic tool calls
    num_ctx=int(os.getenv("WORKER_NUM_CTX", "16384")),             # Fits search results + conversation
    max_tokens=int(os.getenv("WORKER_MAX_TOKENS", os.getenv("WORKER_NUM_PREDICT", "2048"))),
)

_writer_model = os.getenv("WRITER_MODEL", _sup_model)
WRITER_CONFIG = RoleModelConfig(
    provider=os.getenv("WRITER_PROVIDER", os.getenv("SUPERVISOR_PROVIDER", LLM_PROVIDER)).lower(),
    model=_writer_model,
    temperature=float(os.getenv("WRITER_TEMPERATURE", "0.4")),     # Balanced for clear, authoritative exposition
    num_ctx=int(os.getenv("WRITER_NUM_CTX", os.getenv("SUPERVISOR_NUM_CTX", "32768"))),
    max_tokens=int(os.getenv("WRITER_MAX_TOKENS", os.getenv("FINAL_REPORT_MAX_TOKENS", "8192"))),
    reasoning=resolve_reasoning_default(_writer_model, os.getenv("WRITER_REASONING", "false")),
)

# --- Provider Credentials & Settings ---

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION", "")

# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

# AWS Bedrock
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "")
AWS_PROFILE = os.getenv("AWS_PROFILE", "")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")

# --- External API Keys ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Core Execution Settings ---
MAX_RESEARCH_ITERATIONS = int(os.getenv("MAX_RESEARCH_ITERATIONS", "5"))
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "true").lower() in ("true", "1", "yes")

# --- Tavily Search Settings ---
TAVILY_SEARCH_DEPTH_EXPLORATORY = "basic"      # 1 credit
TAVILY_SEARCH_DEPTH_DEEP = "advanced"           # 2 credits
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# --- ArXiv Search Settings ---
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
ARXIV_DELAY_SECONDS = float(os.getenv("ARXIV_DELAY_SECONDS", "3.0"))  # Strict 3s rate-limit per ArXiv ToS
