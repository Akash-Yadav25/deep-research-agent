"""Supervisor node: orchestrates iterative delegation to specialized domain agents."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from deep_researcher.models.schemas import SupervisorDecision
from deep_researcher.models.state import ResearchState
from deep_researcher.utils.fallbacks import get_supervisor_llm

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """You are the Central Research Supervisor coordinating an autonomous research team.

Team Members Available:
1. 'web_research': Specialized in real-world web search via Tavily (industry announcements, benchmarks, documentation, technical blog posts).
2. 'paper_research': Specialized in academic literature search via arXiv (foundational papers, theoretical models, formal evaluations).

Your Mission:
Inspect the current Research Brief, Skeleton Draft Report, and all collected Research Notes so far.
Determine the most urgent, high-leverage research action for this iteration:
- If key practical benchmarks, industry trends, or broad context are missing: select 'web_research' and provide 2-3 specific search queries.
- If foundational theory, academic citations, formal algorithms, or arXiv papers are missing: select 'paper_research' and provide 2-3 specific academic search queries.
- If all core research questions have solid, diverse, cited evidence, or if max iterations are approaching: select 'DONE'.

Guidelines:
- Balance coverage between web and arXiv literature.
- Avoid repeating search queries that have already been executed.
- Target specific unresolved open questions from the draft report.
- Respond strictly using the SupervisorDecision structured schema."""


def supervisor_node(
    state: ResearchState,
) -> Command[Literal["web_researcher", "paper_researcher", "evaluate_progress"]]:
    """Analyze current evidence coverage and delegate the next task or conclude."""
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)

    logger.info(f"[supervisor_node] Iteration {iteration}/{max_iter}")

    # If maximum iterations reached, immediately move to evaluation
    if iteration >= max_iter:
        logger.info("[supervisor_node] Max iterations reached, routing to evaluate_progress.")
        return Command(
            goto="evaluate_progress",
            update={"current_phase": "evaluation"},
        )

    llm = get_supervisor_llm()
    structured_llm = llm.with_structured_output(SupervisorDecision, method="json_schema")

    # Context distillation to keep prompt clean
    brief = state.get("research_brief")
    brief_summary = brief.model_dump_json(indent=2) if brief else "N/A"

    draft = state.get("draft_report")
    draft_summary = draft.model_dump_json(indent=2) if draft else "N/A"

    existing_notes = state.get("research_notes", [])
    notes_summary = "\n".join(
        f"- [Q: {n.research_question or 'general'}] {n.content[:250]}... (sources: {', '.join(n.source_ids)})"
        for n in existing_notes[-15:]  # Recent 15 notes to prevent context bloat
    ) or "No research notes gathered yet."

    sources_count = len(state.get("sources", []))

    user_msg_content = (
        f"## Research Brief\n{brief_summary}\n\n"
        f"## Draft Report Skeleton\n{draft_summary}\n\n"
        f"## Current Evidence ({len(existing_notes)} notes, {sources_count} sources)\n{notes_summary}\n\n"
        f"## Iteration Status: {iteration} / {max_iter}\n"
        "Please evaluate what evidence is still missing and issue your decision."
    )

    try:
        decision: SupervisorDecision = structured_llm.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=user_msg_content),
        ])
    except Exception as exc:
        logger.warning(f"Supervisor structured output failed ({exc}), falling back to alternating heuristic.")
        # Heuristic fallback: alternate web vs paper
        action: Literal["web_research", "paper_research"] = "web_research" if iteration % 2 == 0 else "paper_research"
        query = state.get("topic", "overview")
        decision = SupervisorDecision(
            reasoning=f"Fallback heuristic for iteration {iteration}",
            action=action,
            target_questions=["Core investigation"],
            search_queries=[f"{query} latest analysis" if action == "web_research" else f"all:{query}"],
        )

    logger.info(f"[supervisor_node] Decision: {decision.action} | Reason: {decision.reasoning}")

    if decision.action == "DONE":
        return Command(
            goto="evaluate_progress",
            update={"current_phase": "evaluation"},
        )

    target_worker = (
        "web_researcher" if decision.action == "web_research" else "paper_researcher"
    )
    instruction = (
        f"[Supervisor Directive]\n"
        f"Target questions: {', '.join(decision.target_questions)}\n"
        f"Queries to execute:\n"
        + "\n".join(f"- {q}" for q in decision.search_queries)
        + f"\nReasoning: {decision.reasoning}"
    )

    return Command(
        goto=target_worker,
        update={
            "messages": [HumanMessage(content=instruction)],
            "current_phase": "research",
        },
    )
