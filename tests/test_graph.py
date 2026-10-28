"""Unit and integration tests for graph structures and node logic."""

from unittest.mock import MagicMock, patch
import pytest

from deep_researcher.graph.main_graph import build_main_graph
from deep_researcher.graph.research_loop import build_research_loop
from deep_researcher.models.schemas import ClarificationResult
from deep_researcher.models.state import ResearchNote, ResearchState, Source
from deep_researcher.nodes.clarify import clarify_with_user
from deep_researcher.nodes.final_report import final_report_node


def test_research_loop_compilation():
    """Verify research loop subgraph compiles without structural errors."""
    subgraph_builder = build_research_loop()
    subgraph = subgraph_builder.compile()
    assert subgraph is not None
    assert "supervisor_node" in subgraph.nodes
    assert "web_researcher" in subgraph.nodes
    assert "paper_researcher" in subgraph.nodes
    assert "evaluate_progress" in subgraph.nodes


def test_main_graph_compilation():
    """Verify main graph compiles with all planned nodes."""
    graph = build_main_graph()
    assert graph is not None
    expected_nodes = [
        "clarify_with_user",
        "write_research_brief",
        "write_draft_report",
        "research_loop",
        "final_report_node",
        "subtopic_evaluation",
        "subtopic_research_worker",
    ]
    for node in expected_nodes:
        assert node in graph.nodes


def test_clarify_node_headless():
    """Verify clarify_with_user proceeds autonomously in headless mode."""
    mock_res = ClarificationResult(
        needs_clarification=False,
        refined_topic="Deep Learning Optimization",
        assumptions=["Focus on SGD and Adam variants"],
    )

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_res
    mock_llm.with_structured_output.return_value = mock_structured

    with patch("deep_researcher.nodes.clarify.get_supervisor_llm", return_value=mock_llm):
        initial_state: ResearchState = {
            "messages": [],
            "topic": "deep learning optimization",
            "research_brief": None,
            "draft_report": None,
            "detected_language": "en",
            "research_notes": [],
            "sources": [],
            "iteration_count": 0,
            "max_iterations": 5,
            "current_phase": "clarify",
            "final_report": "",
            "sources_json": "",
            "enable_subtopics": False,
            "subtopics": [],
            "subtopic_reports": [],
        }
        output = clarify_with_user(initial_state)
        assert output["topic"] == "Deep Learning Optimization"
        assert output["current_phase"] == "brief"
        assert len(output["messages"]) == 1


def test_final_report_node_synthesis_and_citation():
    """Verify final_report_node synthesis produces remapped report and valid sources JSON."""
    mock_llm = MagicMock()
    # LLM outputs raw markdown with local citations [web-1] and [arxiv-1]
    mock_response = MagicMock()
    mock_response.content = (
        "# Optimization Survey\n\n"
        "Adam optimizer converges faster [web-1], but stochastic methods offer generalization [arxiv-1]."
    )
    mock_llm.invoke.return_value = mock_response

    sources = [
        Source(local_id="web-1", title="Adam Optimization Overview", url="https://example.com/adam", source_type="web"),
        Source(local_id="arxiv-1", title="Generalization of SGD", url="https://arxiv.org/abs/1701.00001", source_type="arxiv", arxiv_id="1701.00001"),
    ]
    notes = [
        ResearchNote(content="Adam is an adaptive learning rate optimizer.", source_ids=["web-1"]),
        ResearchNote(content="SGD generalizes well in overparameterized regimes.", source_ids=["arxiv-1"]),
    ]

    state: ResearchState = {
        "messages": [],
        "topic": "optimization",
        "research_brief": None,
        "draft_report": None,
        "detected_language": "en",
        "research_notes": notes,
        "sources": sources,
        "iteration_count": 2,
        "max_iterations": 5,
        "current_phase": "final",
        "final_report": "",
        "sources_json": "",
        "enable_subtopics": False,
        "subtopics": [],
        "subtopic_reports": [],
    }

    with patch("deep_researcher.nodes.final_report.get_supervisor_llm", return_value=mock_llm):
        result = final_report_node(state)
        assert "[1]" in result["final_report"]
        assert "[2]" in result["final_report"]
        assert "## References" in result["final_report"]
        assert "Adam Optimization Overview" in result["final_report"]
        assert result["current_phase"] == "complete"
        assert len(result["sources_json"]) > 0
