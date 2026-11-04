# Deep Researcher 🔬

A production-grade, hierarchical multi-agent research system built with **LangGraph**, dynamic multi-provider LLM support (**Ollama**, **OpenAI**, **Azure OpenAI**, and **AWS Bedrock**), **Tavily Search**, and the **arXiv** research paper repository.

---

## 🏛️ Core Architectural Pillars

1. **Multi-Provider Strategy & Factory Architecture**: Dynamic, provider-agnostic model selection via `.env` using the Strategy, Factory Method, and Dynamic Registry design patterns. Seamlessly toggle between local Ollama inference, OpenAI, Azure OpenAI, and AWS Bedrock without modifying pipeline code.
2. **Hierarchical Delegation**: A central supervisor strategically plans, evaluates, and delegates targeted sub-inquiries to domain execution agents rather than executing monolithic web queries.
3. **Phase-Ordered Sub-Agent Execution**: Domain workers strictly enforce **Search/Read first**, and **Batch Save second** to maximize key-value cache utilization, prevent context fragmentation, and eliminate LLM thrashing.
4. **Unified Citations**: Research findings use local source tags (`[web-1]`, `[arxiv-2]`) that are dynamically remapped into a globally unified sequential namespace (`[1]`, `[2]`, `[3]`) and published alongside an academic reference bibliography and structured JSON source catalog.
5. **Fault Isolation & Tool Circuit Breakers**: Automated circuit breakers and tool-level retry policies prevent execution stalls during external search outages.

---

## 🔌 Supported Model Providers

The system uses a unified `BaseLLMProvider` strategy interface with automatic provider discovery. Configure the active provider via `LLM_PROVIDER` in your `.env` file:

| Provider | `LLM_PROVIDER` | Underlying Client | Recommended Supervisor Model | Recommended Worker Model | Key Features |
|:---|:---|:---|:---|:---|:---|
| **Ollama** *(Default)* | `ollama` | `ChatOllama` | `deepseek-r1:14b` | `qwen3.5:9b` | 100% local and private, zero API costs, `<think>` token capture. |
| **OpenAI** | `openai` | `ChatOpenAI` | `gpt-4o` | `gpt-4o-mini` | High throughput, strict structured outputs, native tool calling. |
| **Azure OpenAI** | `azure_openai` | `AzureChatOpenAI` | `gpt-4o` *(deployment)* | `gpt-4o-mini` *(deployment)* | Enterprise compliance, private endpoints, regional data sovereignty. |
| **AWS Bedrock** | `bedrock` | `ChatBedrockConverse` | `anthropic.claude-3-5-sonnet-20241022-v2:0` | `anthropic.claude-3-5-haiku-20241022-v1:0` | Bedrock Converse API, IAM role/profile credential chains, Converse tool calling. |

---

## 🧠 Model Allocation Strategy

| Role | Responsibility | Invocation Pattern | Description |
|:---|:---|:---|:---|
| **Supervisor & Planner** | Topic clarification, brief generation, skeleton drafting, iterative evaluation, final report synthesis. | `.with_structured_output(Schema)` | High-reasoning model optimized for planning and Pydantic schema generation. |
| **Domain Workers** | Web search & content extraction (Tavily), academic preprint analysis (arXiv). | Tool Calling (`create_react_agent`) | High token throughput with native support for tool/function calling and iterative execution. |

> [!IMPORTANT]
> **Hardware & Memory Optimization for Local Ollama Users**
>
> Swapping between different large models (e.g., alternating between a 14B supervisor and a 9B worker) requires Ollama to continually reload weights in VRAM/system RAM.
>
> **Recommendation for Lower-End / Memory-Constrained Laptops**:
> Configure both roles to use the same model (e.g. `qwen3.5:9b`):
> ```env
> LLM_PROVIDER=ollama
> SUPERVISOR_MODEL=qwen3.5:9b
> WORKER_MODEL=qwen3.5:9b
> ```

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
[Publish Deliverables: report.md + sources.json]
```

---

## 🚀 Quickstart

### 1. Prerequisites
- Python >= 3.11
- A valid search API key from [tavily.com](https://tavily.com) (1,000 free requests/month).
- Access to at least one supported LLM provider:
  - **Ollama**: [ollama.com](https://ollama.com/) running locally with pulled models:
    ```bash
    ollama pull deepseek-r1:14b
    ollama pull qwen3.5:9b
    ```
  - **OpenAI**: An active API key with access to GPT-4o models.
  - **Azure OpenAI**: An Azure resource with model deployments and endpoint.
  - **AWS Bedrock**: AWS credentials with Bedrock model access permissions.

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

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your chosen provider:

#### Option A: Local Ollama (Default)
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
SUPERVISOR_MODEL=deepseek-r1:14b
WORKER_MODEL=qwen3.5:9b
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```

#### Option B: OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SUPERVISOR_MODEL=gpt-4o
WORKER_MODEL=gpt-4o-mini
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```

#### Option C: Azure OpenAI
```env
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
SUPERVISOR_MODEL=gpt-4o
WORKER_MODEL=gpt-4o-mini
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```

#### Option D: AWS Bedrock
```env
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
# Credentials can use AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,
# AWS_PROFILE, or default EC2/EKS IAM instance roles
SUPERVISOR_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
WORKER_MODEL=anthropic.claude-3-5-haiku-20241022-v1:0
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```

---

### 4. Running Research

```bash
# Basic run (uses models defined in .env)
deep-researcher "Recent breakthroughs in solid-state EV batteries"

# Custom iterations
deep-researcher "Transformer architectural advances in low-resource reasoning" --max-iterations 4

# Output directory override
deep-researcher "Quantum error mitigation techniques" -o ./my_reports
```

---

## 🧪 Testing

Run the full automated test suite:
```bash
pytest tests/ -v
```

All 30 unit and integration tests verify:
- **Provider Registry & Strategies**: Validation, parameter mappings, and dynamic factory dispatch across Ollama, OpenAI, Azure OpenAI, and AWS Bedrock.
- **Structured Pydantic Schemas**: Validation, bounds, and deserialization across all pipeline steps.
- **State Deduplication**: Deduplication and metadata enrichment in state reducers.
- **Citation Engine**: Local-to-global sequential citation remapping (`[web-1]` → `[1]`) and composite bracket handling.
- **Circuit Breakers**: Trip conditions (CLOSED → OPEN) and failure isolation for web and arXiv searches.
- **LangGraph Compilation**: Structural integrity and conditional routing across subgraphs.

---

## 📦 Output Deliverables

Each research session creates a timestamped folder inside `output/`:
- **`report.md`**: Publication-grade research report with inline sequential citations `[1]`, `[2]`, executive summary, technical sections, and full references bibliography.
- **`sources.json`**: Complete machine-readable catalog of all retrieved web and arXiv sources with titles, authors, URLs, arXiv IDs, and relevance metrics.
