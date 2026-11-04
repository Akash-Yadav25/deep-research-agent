"""Top-level research graph orchestration."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from deep_researcher.graph.research_loop import build_research_loop
from deep_researcher.models.state import ResearchState
from deep_researcher.nodes.brief import write_research_brief
from deep_researcher.nodes.clarify import clarify_with_user
from deep_researcher.nodes.draft import write_draft_report
from deep_researcher.nodes.final_report import final_report_node


def build_main_graph(checkpointer=None):
    """Build and compile the main hierarchical research graph.

    Execution Pipeline:
        START -> clarify_with_user -> write_research_brief -> write_draft_report
              -> research_loop (supervisor subgraph)
              -> final_report_node
              -> END
    """
    builder = StateGraph(ResearchState)

    # 1. Pipeline planning nodes
    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("write_research_brief", write_research_brief)
    builder.add_node("write_draft_report", write_draft_report)

    # 2. Iterative research loop subgraph
    research_subgraph = build_research_loop().compile()
    builder.add_node("research_loop", research_subgraph)

    # 3. Report synthesis & publication node
    builder.add_node("final_report_node", final_report_node)

    # Edges - Main spine
    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("clarify_with_user", "write_research_brief")
    builder.add_edge("write_research_brief", "write_draft_report")
    builder.add_edge("write_draft_report", "research_loop")
    builder.add_edge("research_loop", "final_report_node")
    builder.add_edge("final_report_node", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)
