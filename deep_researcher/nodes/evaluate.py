"""Evaluate research progress: checks coverage, identifies gaps, and decides termination."""

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from deep_researcher.models.schemas import EvaluationResult
from deep_researcher.models.state import ResearchState
from deep_researcher.utils.fallbacks import get_supervisor_llm

logger = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = """You are an independent Research Quality Assessor and Peer Reviewer.
Inspect the Research Brief, Skeleton Draft, and all collected empirical Research Notes.

Evaluate:
1. `coverage_score`: Grade evidence thoroughness from 1 to 10 across all research questions.
2. `gaps`: List specific topics or questions that are still missing clear evidence or empirical backing.
3. `sufficient`: Set to true if evidence is sufficiently comprehensive, authoritative, and cited to author a definitive, top-tier research report.
4. `next_priority`: The single most important missing aspect to investigate if not yet sufficient.

Note: If at least 3 high-quality notes with valid sources exist across key facets, or if coverage score >= 7, consider it sufficient to prevent unnecessary token consumption."""


def evaluate_progress(
    state: ResearchState,
) -> Command[Literal["supervisor_node", "__end__"]]:
    """Assess evidence coverage and decide whether to loop for another iteration or terminate."""
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)

    logger.info(f"[evaluate_progress] Assessing coverage at iteration {iteration}/{max_iter}")

    # Safety termination if maximum iterations reached
    if iteration >= max_iter:
        logger.info("[evaluate_progress] Max iteration threshold met. Concluding research loop.")
        return Command(
            goto=END,
            update={"current_phase": "final"},
        )

    notes = state.get("research_notes", [])
    sources = state.get("sources", [])

    # If we have no notes at all, loop back immediately
    if not notes:
        logger.info("[evaluate_progress] No notes gathered yet. Looping to supervisor.")
        return Command(goto="supervisor_node")

    llm = get_supervisor_llm()
    structured_llm = llm.with_structured_output(EvaluationResult)

    brief = state.get("research_brief")
    brief_text = brief.model_dump_json(indent=2) if brief else state.get("topic", "")

    notes_summary = "\n\n".join(
        f"Note {idx} (Sources: {', '.join(n.source_ids)}):\n{n.content[:300]}..."
        for idx, n in enumerate(notes[-10:], start=1)
    )

    prompt = (
        f"## Research Brief\n{brief_text}\n\n"
        f"## Gathered Notes ({len(notes)} total notes, {len(sources)} sources)\n{notes_summary}\n\n"
        f"Iteration {iteration} of maximum {max_iter} allowed."
    )

    try:
        eval_result: EvaluationResult = structured_llm.invoke([
            SystemMessage(content=EVAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
    except Exception as exc:
        logger.warning(f"Evaluation structured output failed ({exc}), using heuristic.")
        # If we have at least 2 notes and sources, consider sufficient
        sufficient = len(notes) >= 3 and len(sources) >= 3
        eval_result = EvaluationResult(
            coverage_score=7 if sufficient else 5,
            gaps=[],
            sufficient=sufficient,
            next_priority=None if sufficient else "Additional literature search",
        )

    logger.info(
        f"[evaluate_progress] Score: {eval_result.coverage_score}/10 | Sufficient: {eval_result.sufficient}"
    )

    if eval_result.sufficient:
        return Command(
            goto=END,
            update={"current_phase": "final"},
        )

    return Command(
        goto="supervisor_node",
        update={"current_phase": "research"},
    )
