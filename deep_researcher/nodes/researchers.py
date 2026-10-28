"""Domain researcher nodes: Phase-ordered execution for Web and ArXiv agents."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from deep_researcher.models.state import ResearchNote, ResearchState, Source
from deep_researcher.tools.arxiv_search import search_arxiv
from deep_researcher.tools.tavily_search import extract_webpage_content, search_web
from deep_researcher.utils.fallbacks import (
    arxiv_circuit_breaker,
    get_worker_llm,
    web_circuit_breaker,
)

logger = logging.getLogger(__name__)

WEB_WORKER_PROMPT = """You are a Web Intelligence Specialist with access to real-time search tools.

Strict Phase Protocol:
Phase 1 (Search & Read):
- Execute the search queries provided by the supervisor using `search_web`.
- If a result appears uniquely vital or requires full text, use `extract_webpage_content`.
Phase 2 (Synthesize):
- Synthesize all discovered findings into an analytical summary.
- Explicitly cite findings using local source markers like [web-1], [web-2] corresponding to the search results you uncovered.
- Be precise, factual, and include empirical numbers, release dates, or architecture details where found."""

ARXIV_WORKER_PROMPT = """You are an Academic Literature Specialist with access to the arXiv preprint repository.

Strict Phase Protocol:
Phase 1 (Search & Read):
- Execute the academic search queries provided by the supervisor using `search_arxiv`.
- Analyze paper abstracts, theoretical claims, novel architectures, and empirical findings.
Phase 2 (Synthesize):
- Synthesize the technical contributions of the papers.
- Explicitly cite findings using local source markers like [arxiv-1], [arxiv-2] corresponding to the retrieved papers.
- Highlight core theorems, benchmark results, mathematical formalisms, and author contributions."""


def _parse_web_sources(messages: list[Any], existing_count: int) -> tuple[list[Source], int]:
    """Phase 2 helper: Extract structured Source objects from Tavily ToolMessages."""
    sources: list[Source] = []
    counter = existing_count

    for msg in messages:
        if isinstance(msg, ToolMessage) or (hasattr(msg, "name") and msg.name in ("search_web", "extract_webpage_content")):
            content_str = str(msg.content)
            # Find all Title / URL pairs
            # Pattern: Title: <title>\n    URL: <url>
            blocks = re.split(r"\[\d+\]\s*Title:", content_str)
            for block in blocks[1:]:
                title_match = re.search(r"^(.*?)(?:\n\s*URL:)", block, re.DOTALL)
                url_match = re.search(r"URL:\s*([^\s\n]+)", block)
                score_match = re.search(r"Score:\s*([^\s\n]+)", block)
                content_match = re.search(r"Content:\s*(.*?)(?=$|\n\s*\[\d+\])", block, re.DOTALL)

                url = url_match.group(1).strip() if url_match else ""
                if url:
                    counter += 1
                    title = title_match.group(1).strip() if title_match else url
                    snippet = content_match.group(1).strip()[:500] if content_match else ""
                    try:
                        score = float(score_match.group(1).strip()) if score_match else 0.0
                    except ValueError:
                        score = 0.0

                    sources.append(
                        Source(
                            local_id=f"web-{counter}",
                            url=url,
                            title=title,
                            content_snippet=snippet,
                            source_type="web",
                            relevance_score=score,
                        )
                    )

    return sources, counter


def _parse_arxiv_sources(messages: list[Any], existing_count: int) -> tuple[list[Source], int]:
    """Phase 2 helper: Extract structured Source objects from ArXiv ToolMessages."""
    sources: list[Source] = []
    counter = existing_count

    for msg in messages:
        if isinstance(msg, ToolMessage) or (hasattr(msg, "name") and msg.name == "search_arxiv"):
            content_str = str(msg.content)
            entries = content_str.split("\n---\n")
            for entry in entries:
                title_match = re.search(r"Title:\s*(.*?)(?=\nArXiv ID:|$)", entry, re.DOTALL)
                arxiv_id_match = re.search(r"ArXiv ID:\s*([^\s\n]+)", entry)
                pub_match = re.search(r"Published:\s*([^\s\n]+)", entry)
                authors_match = re.search(r"Authors:\s*(.*?)(?=\nCategories:|$)", entry, re.DOTALL)
                url_match = re.search(r"URL:\s*([^\s\n]+)", entry)
                abstract_match = re.search(r"Abstract:\s*(.*?)(?=$)", entry, re.DOTALL)

                arxiv_id = arxiv_id_match.group(1).strip() if arxiv_id_match else ""
                url = url_match.group(1).strip() if url_match else (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")

                if arxiv_id or url:
                    counter += 1
                    title = title_match.group(1).strip() if title_match else f"arXiv:{arxiv_id}"
                    authors_raw = authors_match.group(1).strip() if authors_match else ""
                    authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
                    pub_date = pub_match.group(1).strip() if pub_match else ""
                    snippet = abstract_match.group(1).strip()[:500] if abstract_match else ""

                    sources.append(
                        Source(
                            local_id=f"arxiv-{counter}",
                            url=url,
                            title=title,
                            content_snippet=snippet,
                            source_type="arxiv",
                            authors=authors,
                            published_date=pub_date,
                            arxiv_id=arxiv_id,
                            relevance_score=1.0,
                        )
                    )

    return sources, counter


def web_researcher(state: ResearchState) -> dict:
    """Execute phase-ordered Web research worker (Search & Read -> Batch Save)."""
    logger.info("[web_researcher] Starting Web research phase...")

    # Phase 1: Search & Read
    # Build tool-enabled ReAct agent with qwen3:8b
    worker_llm = get_worker_llm()
    agent = create_react_agent(
        model=worker_llm,
        tools=[search_web, extract_webpage_content],
        prompt=WEB_WORKER_PROMPT,
    )

    last_instruction = state["messages"][-1].content if state.get("messages") else state.get("topic", "")

    try:
        # Protect tool execution with circuit breaker
        response = web_circuit_breaker.call_with_breaker(
            agent.invoke,
            {"messages": [HumanMessage(content=last_instruction)]},
        )
        agent_messages = response.get("messages", [])
    except Exception as exc:
        logger.error(f"[web_researcher] Execution error or circuit breaker open: {exc}")
        return {
            "research_notes": [
                ResearchNote(
                    content=f"Web search encountered temporary difficulty: {exc}",
                    source_ids=[],
                    confidence=0.1,
                )
            ],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_phase": "evaluation",
        }

    # Phase 2: Batch Save Phase
    existing_sources = state.get("sources", [])
    existing_web_count = sum(1 for s in existing_sources if s.source_type == "web")
    extracted_sources, _ = _parse_web_sources(agent_messages, existing_web_count)

    # Extract final analytical synthesis from last AIMessage
    final_text = ""
    for msg in reversed(agent_messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = str(msg.content).strip()
            break

    notes = []
    if final_text:
        source_ids = [s.local_id for s in extracted_sources]
        notes.append(
            ResearchNote(
                content=final_text,
                source_ids=source_ids,
                research_question="Web Domain Investigation",
                confidence=0.9 if extracted_sources else 0.5,
            )
        )

    logger.info(
        f"[web_researcher] Batch save complete: {len(extracted_sources)} sources, {len(notes)} notes."
    )

    return {
        "sources": extracted_sources,
        "research_notes": notes,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "current_phase": "evaluation",
    }


def paper_researcher(state: ResearchState) -> dict:
    """Execute phase-ordered Academic ArXiv research worker (Search & Read -> Batch Save)."""
    logger.info("[paper_researcher] Starting Academic ArXiv research phase...")

    # Phase 1: Search & Read
    worker_llm = get_worker_llm()
    agent = create_react_agent(
        model=worker_llm,
        tools=[search_arxiv],
        prompt=ARXIV_WORKER_PROMPT,
    )

    last_instruction = state["messages"][-1].content if state.get("messages") else state.get("topic", "")

    try:
        response = arxiv_circuit_breaker.call_with_breaker(
            agent.invoke,
            {"messages": [HumanMessage(content=last_instruction)]},
        )
        agent_messages = response.get("messages", [])
    except Exception as exc:
        logger.error(f"[paper_researcher] Execution error or circuit breaker open: {exc}")
        return {
            "research_notes": [
                ResearchNote(
                    content=f"ArXiv literature search encountered temporary difficulty: {exc}",
                    source_ids=[],
                    confidence=0.1,
                )
            ],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_phase": "evaluation",
        }

    # Phase 2: Batch Save Phase
    existing_sources = state.get("sources", [])
    existing_arxiv_count = sum(1 for s in existing_sources if s.source_type == "arxiv")
    extracted_sources, _ = _parse_arxiv_sources(agent_messages, existing_arxiv_count)

    final_text = ""
    for msg in reversed(agent_messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = str(msg.content).strip()
            break

    notes = []
    if final_text:
        source_ids = [s.local_id for s in extracted_sources]
        notes.append(
            ResearchNote(
                content=final_text,
                source_ids=source_ids,
                research_question="Academic Literature Investigation",
                confidence=0.95 if extracted_sources else 0.5,
            )
        )

    logger.info(
        f"[paper_researcher] Batch save complete: {len(extracted_sources)} sources, {len(notes)} notes."
    )

    return {
        "sources": extracted_sources,
        "research_notes": notes,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "current_phase": "evaluation",
    }
