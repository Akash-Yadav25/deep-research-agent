# Deep Researcher 🔬

A production-grade, hierarchical multi-agent research system built with **LangGraph**, locally deployed **Ollama** models, **Tavily Search**, and the **arXiv** research paper repository.

---

## 🏛️ Core Architectural Pillars

1. **Hierarchical Delegation**: A central supervisor (`deepseek-r1:14b`) strategically plans, evaluates, and delegates targeted sub-inquiries to domain execution agents rather than executing monolithic web queries.
2. **Phase-Ordered Sub-Agent Execution**: Domain workers (`qwen3.5:9b`) strictly enforce **Search/Read first**, and **Batch Save second** to maximize key-value cache utilization, prevent context fragmentation, and eliminate LLM thrashing.
3. **Unified Citations**: Research findings use local source tags (`[web-1]`, `[arxiv-2]`) that are dynamically remapped into a globally unified sequential namespace (`[1]`, `[2]`, `[3]`) and published alongside an academic reference bibliography and structured JSON source catalog.
4. **Resilient Fallbacks**: Multi-tier model fallback chains, tool-level retry policies, and an automated circuit breaker prevent execution stalls.

---

## 🧠 Model Allocation Strategy

| Role | Model | Deployment | Invocation Pattern | Rationale |
|:---|:---|:---|:---|:---|
| **Supervisor & Planner** | `deepseek-r1:14b` | Local Ollama | `.with_structured_output(..., method="json_schema")` | Deep reasoning and planning capabilities. Operating via grammar-constrained structured decoding avoids Ollama tool-calling limitations. Reasoning traces `<think>` are cleanly captured. |
| **Domain Workers** | `qwen3.5:9b` | Local Ollama | Native Tool Calling (`.bind_tools()` / `create_react_agent`) | High token throughput with native support for tool/function calling and search execution. |

> [!IMPORTANT]
> **Hardware & Memory Optimization for Laptops and Low-End Devices**
>
> Swapping between different large models (e.g., alternating between a 14B supervisor and a 9B worker) requires Ollama to continually unload and reload model weights into VRAM/system RAM, which takes noticeable time. Furthermore, retaining several large models in memory can saturate available RAM, leading to OS memory swapping and making the laptop sluggish.
>
> **Recommendation for Lower-End / Memory-Constrained Devices**:
> Configure both roles to use the **same model** (such as `qwen3.5:9b`):
> ```env
> SUPERVISOR_MODEL=qwen3.5:9b
> WORKER_MODEL=qwen3.5:9b
> ```
> This keeps a single model warm in memory, eliminates model swapping latency, and ensures responsive, smooth local execution.

---

## 🔄 Execution Pipeline

```
[Start]
   │
   ▼
1. clarify_with_user         (Disambiguates topic; adopts explicit assumptions in headless mode)
   │
   ▼
2. write_research_brief      (Detects language; generates prioritized questions & search queries)
   │
   ▼
3. write_draft_report        (Generates skeleton report with section hypotheses & open questions)
   │
   ▼
4. supervisor_subgraph       ◄──┐ (Iterative research loop: coordinates platform agents,
   ├─ supervisor_node           │  evaluates evidence coverage, updates blackboard)
   ├─ web_researcher            │
   ├─ paper_researcher          │
   └─ evaluate_progress ────────┘
   │
   ▼
5. final_report_generation   (Synthesizes report, remaps local citations to [1],[2], generates references)
   │
   ▼
6. subtopics (optional)      (Evaluates spin-off areas & fans out parallel mini-briefs via Send)
   │
   ▼
[Publish Deliverables: report.md + sources.json]
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python >= 3.11
- [Ollama](https://ollama.com/) running locally:
  ```bash
  ollama pull deepseek-r1:14b   # High-capacity reasoning planner
  ollama pull qwen3.5:9b        # Worker & tool-calling model (or unified model)
  ```
- (Optional) Tavily API key from [tavily.com](https://tavily.com) (1,000 free requests/month).

### 2. Installation
```bash
git clone <repo-url>
cd "Deep Researcher"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e ".[dev]"
```

### 3. Configuration
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```
Edit `.env`:
```env
TAVILY_API_KEY=tvly-your-api-key-here
OLLAMA_BASE_URL=http://localhost:11434

# Standard Dual-Model Setup (Requires ~16GB+ VRAM/RAM)
SUPERVISOR_MODEL=deepseek-r1:14b
WORKER_MODEL=qwen3.5:9b

# Low-End Devices / Single-Model Recommendation (Zero model-swap overhead):
# SUPERVISOR_MODEL=qwen3.5:9b
# WORKER_MODEL=qwen3.5:9b

MAX_RESEARCH_ITERATIONS=5
ENABLE_SUBTOPICS=false
HEADLESS_MODE=true
```

### 4. Running Research

```bash
# Basic run
deep-researcher "Recent breakthroughs in solid-state EV batteries"

# Custom iterations and follow-up subtopics enabled
deep-researcher "Transformer architectural advances in low-resource reasoning" --max-iterations 4 --subtopics

# Output directory override
deep-researcher "Quantum error mitigation techniques" -o ./my_reports
```

---

## 🧪 Testing

Run the full automated test suite:
```bash
pytest tests/ -v
```

All 18 unit and integration tests verify:
- Structured Pydantic schema validation & bounds
- Deduplication and metadata enrichment in state reducers
- Local-to-global sequential citation remapping (`[web-1]` → `[1]`)
- Composite and multi-token citation brackets
- ArXiv rate-limit compliant wrapper and Tavily tool fallbacks
- Circuit breaker state transitions (CLOSED → OPEN)
- Hierarchical LangGraph compilation and conditional routing

---

## 📦 Output Deliverables

Each research session creates a timestamped folder inside `output/`:
- **`report.md`**: Publication-grade research report with inline sequential citations `[1]`, `[2]`, executive summary, technical sections, and full references bibliography.
- **`sources.json`**: Complete machine-readable catalog of all retrieved web and arXiv sources with titles, authors, URLs, arXiv IDs, and relevance metrics.
