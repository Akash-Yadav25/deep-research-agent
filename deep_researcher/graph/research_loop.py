"""Supervisor research loop subgraph."""

from langgraph.graph import START, StateGraph

from deep_researcher.models.state import ResearchState
from deep_researcher.nodes.evaluate import evaluate_progress
from deep_researcher.nodes.researchers import paper_researcher, web_researcher
from deep_researcher.nodes.supervisor import supervisor_node


def build_research_loop() -> StateGraph:
    """Construct the supervisor research loop subgraph.

    Flow:
        START -> supervisor_node
        supervisor_node -> [Command(goto="web_researcher" | "paper_researcher" | "evaluate_progress")]
        web_researcher -> evaluate_progress
        paper_researcher -> evaluate_progress
        evaluate_progress -> [Command(goto="supervisor_node" | END)]
    """
    builder = StateGraph(ResearchState)

    # Register nodes
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("web_researcher", web_researcher)
    builder.add_node("paper_researcher", paper_researcher)
    builder.add_node("evaluate_progress", evaluate_progress)

    # Entry edge
    builder.add_edge(START, "supervisor_node")

    # Worker edges returning to evaluate_progress
    builder.add_edge("web_researcher", "evaluate_progress")
    builder.add_edge("paper_researcher", "evaluate_progress")

    return builder
