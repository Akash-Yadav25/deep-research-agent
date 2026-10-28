"""Central configuration for the Deep Researcher system."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific Ollama model."""

    model: str
    temperature: float
    num_ctx: int
    num_predict: int = -1
    keep_alive: str = "15m"
    base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    reasoning: bool = False  # Supported for reasoning models such as deepseek-r1


# --- Model Configurations ---

SUPERVISOR_CONFIG = ModelConfig(
    model=os.getenv("SUPERVISOR_MODEL", "deepseek-r1:14b"),
    temperature=float(os.getenv("SUPERVISOR_TEMPERATURE", "0.6")),  # Optimal for reasoning/planning
    num_ctx=int(os.getenv("SUPERVISOR_NUM_CTX", "32768")),          # Expanded context for briefs/drafts
    num_predict=4096,
    reasoning=True,                                                 # Strips <think> tokens from response
)

WORKER_CONFIG = ModelConfig(
    model=os.getenv("WORKER_MODEL", "qwen3:8b"),
    temperature=float(os.getenv("WORKER_TEMPERATURE", "0.0")),     # Strict 0.0 for deterministic tool calls
    num_ctx=int(os.getenv("WORKER_NUM_CTX", "16384")),             # Fits search results + conversation
    num_predict=2048,
)

# --- External API Keys ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Core Execution Settings ---
MAX_RESEARCH_ITERATIONS = int(os.getenv("MAX_RESEARCH_ITERATIONS", "5"))
ENABLE_SUBTOPICS = os.getenv("ENABLE_SUBTOPICS", "false").lower() in ("true", "1", "yes")
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "true").lower() in ("true", "1", "yes")

# --- Tavily Search Settings ---
TAVILY_SEARCH_DEPTH_EXPLORATORY = "basic"      # 1 credit
TAVILY_SEARCH_DEPTH_DEEP = "advanced"           # 2 credits
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# --- ArXiv Search Settings ---
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
ARXIV_DELAY_SECONDS = float(os.getenv("ARXIV_DELAY_SECONDS", "3.0"))  # Strict 3s rate-limit per ArXiv ToS
