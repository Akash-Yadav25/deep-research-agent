"""LangGraph state schemas and custom reducers for Deep Researcher."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from deep_researcher.models.schemas import DraftReport, ResearchBrief


@dataclass
class Source:
    """A research source record with metadata and citation tracking."""

    local_id: str                          # e.g., "web-1", "arxiv-2"
    global_id: int | None = None           # Sequential integer [1], [2], ... assigned at final report
    url: str = ""
    title: str = ""
    content_snippet: str = ""
    source_type: str = ""                  # "web" | "arxiv"
    relevance_score: float = 0.0
    # Academic/arXiv specific
    authors: list[str] = field(default_factory=list)
    published_date: str = ""
    arxiv_id: str = ""

    def to_dict(self) -> dict:
        """Convert Source instance to dictionary."""
        return {
            "local_id": self.local_id,
            "global_id": self.global_id,
            "url": self.url,
            "title": self.title,
            "content_snippet": self.content_snippet,
            "source_type": self.source_type,
            "relevance_score": self.relevance_score,
            "authors": self.authors,
            "published_date": self.published_date,
            "arxiv_id": self.arxiv_id,
        }


@dataclass
class ResearchNote:
    """An analytical note extracted by a domain researcher agent."""

    content: str                           # Synthesized finding or empirical data point
    source_ids: list[str] = field(default_factory=list)  # Referenced local source IDs
    research_question: str = ""            # Target question addressed
    confidence: float = 1.0                # Worker confidence score (0.0 to 1.0)


def merge_sources(current: list[Source], update: list[Source]) -> list[Source]:
    """Reducer that merges sources while preventing duplicates.
    
    Deduplicates based on normalized URL or arXiv ID.
    Preserves rich metadata when duplicates are updated.
    """
    merged: list[Source] = list(current)
    seen_keys: set[str] = set()

    for s in merged:
        key = (s.url or s.arxiv_id or s.local_id).strip().lower()
        if key:
            seen_keys.add(key)

    for s in update:
        key = (s.url or s.arxiv_id or s.local_id).strip().lower()
        if not key or key not in seen_keys:
            merged.append(s)
            if key:
                seen_keys.add(key)
        else:
            # Update existing source if the incoming source has richer details
            for idx, existing in enumerate(merged):
                existing_key = (existing.url or existing.arxiv_id or existing.local_id).strip().lower()
                if existing_key == key:
                    if s.title and (not existing.title or len(s.title) > len(existing.title)):
                        existing.title = s.title
                    if s.content_snippet and (not existing.content_snippet or len(s.content_snippet) > len(existing.content_snippet)):
                        existing.content_snippet = s.content_snippet
                    if not existing.authors and s.authors:
                        existing.authors = s.authors
                    if not existing.published_date and s.published_date:
                        existing.published_date = s.published_date
                    break

    return merged


class ResearchState(TypedDict):
    """Global blackboard state for the Deep Researcher graph."""

    # --- Communication Blackboard ---
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str

    # --- Planning Artifacts ---
    research_brief: ResearchBrief | None
    draft_report: DraftReport | None
    detected_language: str

    # --- Research Accumulation ---
    research_notes: Annotated[list[ResearchNote], operator.add]
    sources: Annotated[list[Source], merge_sources]
    iteration_count: int
    max_iterations: int
    current_phase: str  # "clarify" | "brief" | "draft" | "research" | "evaluation" | "final" | "complete"

    # --- Output Deliverables ---
    final_report: str
    sources_json: str


class WorkerState(TypedDict):
    """State passed into individual domain worker executions."""

    messages: Annotated[list[BaseMessage], add_messages]
    search_queries: list[str]
    worker_type: str  # "web" | "arxiv"
    research_question: str
    research_notes: Annotated[list[ResearchNote], operator.add]
    sources: Annotated[list[Source], merge_sources]
