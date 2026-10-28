"""Unit tests for Pydantic schemas and data validation."""

import pytest
from pydantic import ValidationError

from deep_researcher.models.schemas import (
    ClarificationResult,
    DraftReport,
    DraftSection,
    EvaluationResult,
    ResearchBrief,
    ResearchQuestion,
    SubtopicEvaluation,
    SupervisorDecision,
)
from deep_researcher.models.state import ResearchNote, Source, merge_sources


def test_clarification_result_schema():
    """Verify ClarificationResult instantiation and defaults."""
    res = ClarificationResult(
        needs_clarification=False,
        refined_topic="Quantum Computing NISQ Algorithms",
        assumptions=["Focus on gate-based quantum systems"],
    )
    assert not res.needs_clarification
    assert res.refined_topic == "Quantum Computing NISQ Algorithms"
    assert len(res.assumptions) == 1
    assert res.clarification_question is None


def test_research_brief_and_questions():
    """Verify ResearchBrief and nested ResearchQuestion validation."""
    q1 = ResearchQuestion(
        question="What is the quantum volume of contemporary processors?",
        search_queries_web=["quantum volume IBM 2024", "quantum volume benchmarks"],
        search_queries_arxiv=["ti:\"quantum volume\" AND cat:quant-ph"],
        priority=1,
    )
    brief = ResearchBrief(
        original_topic="quantum computing benchmarks",
        refined_topic="Empirical Quantum Computing Benchmarks and Metrics",
        detected_language="en",
        research_questions=[q1],
        report_structure=["Introduction", "Metrics", "Discussion"],
        key_terms=["quantum volume", "fidelity", "coherence"],
    )

    assert brief.detected_language == "en"
    assert len(brief.research_questions) == 1
    assert brief.research_questions[0].priority == 1

    # Verify priority constraints (1 <= priority <= 5)
    with pytest.raises(ValidationError):
        ResearchQuestion(question="Invalid priority", priority=10)


def test_draft_report_schema():
    """Verify DraftReport with DraftSections."""
    sec = DraftSection(
        heading="Algorithmic Complexity",
        hypothesis="Polynomial overhead under error mitigation",
        open_questions=["What is the sampling overhead?"],
    )
    draft = DraftReport(
        title="Theoretical Bounds",
        abstract="Working skeleton",
        sections=[sec],
    )
    assert len(draft.sections) == 1
    assert draft.sections[0].heading == "Algorithmic Complexity"


def test_supervisor_decision_schema():
    """Verify SupervisorDecision actions constraint."""
    decision = SupervisorDecision(
        reasoning="Need academic literature on theorems",
        action="paper_research",
        target_questions=["Question 1"],
        search_queries=["cat:cs.AI AND ti:reasoning"],
    )
    assert decision.action == "paper_research"

    # Invalid action should fail validation
    with pytest.raises(ValidationError):
        SupervisorDecision(
            reasoning="Invalid",
            action="invalid_action",  # type: ignore
        )


def test_evaluation_result_schema():
    """Verify EvaluationResult score bounds and fields."""
    ev = EvaluationResult(
        coverage_score=8,
        gaps=["Hardware implementations"],
        sufficient=True,
    )
    assert ev.coverage_score == 8
    assert ev.sufficient

    # Score outside 1-10 should fail
    with pytest.raises(ValidationError):
        EvaluationResult(
            coverage_score=15,
            gaps=[],
            sufficient=False,
        )


def test_merge_sources_reducer():
    """Verify merge_sources deduplication and metadata enrichment."""
    s1 = Source(
        local_id="web-1",
        url="https://example.com/quantum",
        title="Quantum Intro",
        source_type="web",
    )
    s2 = Source(
        local_id="web-2",
        url="https://example.com/quantum",  # Duplicate URL
        title="Quantum Intro Expanded",     # Richer title
        content_snippet="Detailed content snippet",
        source_type="web",
    )
    s3 = Source(
        local_id="arxiv-1",
        url="https://arxiv.org/abs/2301.00001",
        title="arXiv Quantum Paper",
        source_type="arxiv",
        arxiv_id="2301.00001",
    )

    merged = merge_sources([s1], [s2, s3])
    # Should only have 2 distinct sources (example.com and arxiv)
    assert len(merged) == 2
    # Metadata should be enriched
    assert merged[0].title == "Quantum Intro Expanded"
    assert merged[0].content_snippet == "Detailed content snippet"
    assert merged[1].arxiv_id == "2301.00001"
