"""Clarify-with-user node: disambiguates prompt and establishes baseline assumptions."""

import logging
from langchain_core.messages import HumanMessage, SystemMessage

from deep_researcher.config.settings import HEADLESS_MODE
from deep_researcher.models.schemas import ClarificationResult
from deep_researcher.models.state import ResearchState
from deep_researcher.utils.fallbacks import get_supervisor_llm
from deep_researcher.utils.structured import invoke_structured

logger = logging.getLogger(__name__)

CLARIFY_SYSTEM_PROMPT = """You are a senior research coordinator.
Analyze the user's research topic prompt to determine:
1. Whether the topic is ambiguous, excessively broad, or requires domain disambiguation.
2. Formulate explicit, reasonable assumptions that ground the research into a coherent, high-value investigation.
3. Formulate a refined, specific topic title that incorporates these assumptions.

If in headless mode (headless=true), always provide a well-grounded refined topic and a list of working assumptions so the pipeline can proceed autonomously.

You must respond with valid structured output."""


def clarify_with_user(state: ResearchState) -> dict:
    """Analyze the prompt, determine if clarification is required, and formulate assumptions."""
    llm = get_supervisor_llm()

    topic = state.get("topic", "").strip()
    logger.info(f"[clarify_with_user] Analyzing topic: '{topic}' (headless={HEADLESS_MODE})")

    user_prompt = (
        f"Research topic: \"{topic}\"\n"
        f"Execution mode: {'Headless (autonomous, make reasonable assumptions)' if HEADLESS_MODE else 'Interactive'}"
    )

    try:
        result: ClarificationResult = invoke_structured(
            llm=llm,
            schema=ClarificationResult,
            messages=[
                SystemMessage(content=CLARIFY_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ],
        )
        refined = result.refined_topic.strip() if result.refined_topic else topic
        assumptions = result.assumptions
    except Exception as exc:
        logger.warning(f"Clarification structured LLM failed ({exc}), falling back to direct topic.")
        refined = topic
        assumptions = []

    announcement = f"Starting research on: '{refined}'"
    if assumptions:
        announcement += f" (Assumptions: {'; '.join(assumptions)})"

    return {
        "topic": refined,
        "messages": [HumanMessage(content=announcement)],
        "current_phase": "brief",
    }
