"""Optional subtopic evaluation and fan-out generation for deep-dive spin-off reports."""

from __future__ import annotations

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Send

from deep_researcher.models.schemas import SubtopicEvaluation
from deep_researcher.models.state import ResearchState
from deep_researcher.utils.fallbacks import get_supervisor_llm

logger = logging.getLogger(__name__)

SUBTOPIC_SYSTEM_PROMPT = """You are a Research Director evaluating follow-up inquiries.
Review the completed research report and identify 2 to 3 high-impact, specific spin-off subtopics that:
1. Warrant dedicated deep-dive investigations of their own.
2. Address questions that were only briefly introduced or identified as open frontiers in the main report.

You must respond with valid structured output conforming to SubtopicEvaluation."""


def subtopic_evaluation(state: ResearchState) -> dict:
    """Evaluate whether follow-up spin-off subtopics should be generated."""
    if not state.get("enable_subtopics", False):
        logger.info("[subtopic_evaluation] Subtopics feature is disabled. Skipping.")
        return {"subtopics": []}

    report = state.get("final_report", "")
    if not report:
        return {"subtopics": []}

    llm = get_supervisor_llm()
    structured_llm = llm.with_structured_output(SubtopicEvaluation)

    logger.info("[subtopic_evaluation] Evaluating candidate spin-off subtopics...")

    try:
        result: SubtopicEvaluation = structured_llm.invoke([
            SystemMessage(content=SUBTOPIC_SYSTEM_PROMPT),
            HumanMessage(content=f"Report excerpt:\n{report[:4000]}"),
        ])
        subtopics = result.subtopics if result.has_subtopics else []
    except Exception as exc:
        logger.warning(f"Subtopic evaluation failed ({exc}), setting empty subtopics.")
        subtopics = []

    logger.info(f"[subtopic_evaluation] Identified {len(subtopics)} spin-off subtopics.")
    return {"subtopics": subtopics}


def route_subtopics(state: ResearchState) -> list[Send] | str:
    """Routing function: fan out to subtopic workers if present, else terminate."""
    subtopics = state.get("subtopics", [])
    if not subtopics or not state.get("enable_subtopics", False):
        return END

    logger.info(f"[route_subtopics] Fanning out {len(subtopics)} subtopic investigations via Send...")
    return [
        Send(
            "subtopic_research_worker",
            {
                "topic": st,
                "messages": [],
                "research_brief": None,
                "draft_report": None,
                "detected_language": state.get("detected_language", "en"),
                "research_notes": [],
                "sources": [],
                "iteration_count": 0,
                "max_iterations": 2,  # Compact iterations for spin-off subtopics
                "current_phase": "brief",
                "final_report": "",
                "sources_json": "",
                "enable_subtopics": False,  # Prevent recursive runaway
                "subtopics": [],
                "subtopic_reports": [],
            },
        )
        for st in subtopics
    ]


def subtopic_research_worker(subtopic_state: ResearchState) -> dict:
    """Compact research executor for a spin-off subtopic."""
    subtopic_name = subtopic_state.get("topic", "Subtopic")
    logger.info(f"[subtopic_research_worker] Processing spin-off subtopic: '{subtopic_name}'")

    llm = get_supervisor_llm()
    prompt = (
        f"Generate a concise, targeted executive brief (400-600 words) on this spin-off subtopic: \"{subtopic_name}\".\n"
        "Provide background, key mechanisms, and relevant research directions."
    )

    try:
        response = llm.invoke([
            SystemMessage(content="You are an expert technical researcher producing an executive mini-brief."),
            HumanMessage(content=prompt),
        ])
        content = str(response.content).strip()
    except Exception as exc:
        content = f"Brief for '{subtopic_name}' could not be generated: {exc}"

    report = f"### Spin-off Brief: {subtopic_name}\n\n{content}\n"
    return {"subtopic_reports": [report]}
