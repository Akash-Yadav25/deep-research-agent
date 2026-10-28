"""Write research brief node: generates structured research plan."""

import logging
from langchain_core.messages import HumanMessage, SystemMessage

from deep_researcher.models.schemas import ResearchBrief, ResearchQuestion
from deep_researcher.models.state import ResearchState
from deep_researcher.utils.fallbacks import get_supervisor_llm

logger = logging.getLogger(__name__)

BRIEF_SYSTEM_PROMPT = """You are a Principal AI Scientist and research strategist.
Given a research topic, develop an exhaustive, highly structured research brief.

Your brief must include:
1. `original_topic`: The user's input topic.
2. `refined_topic`: A clear, scholarly research title.
3. `detected_language`: The ISO 639-1 language code of the topic (e.g., 'en', 'es', 'zh', 'ja', 'de').
4. `research_questions`: 3 to 6 deep, non-trivial research questions. For each question:
   - Provide 2 to 3 targeted web search queries tailored for search engines.
   - Provide 1 to 2 academic search queries targeting arXiv preprints (utilizing terms like 'ti:...', 'cat:...', etc.).
   - Assign priority (1=critical core concept, 2=substantive analysis, 3=edge cases/benchmarks).
5. `report_structure`: Recommended section headings for the comprehensive final publication.
6. `key_terms`: Specific domain keywords, acronyms, and technical phrases.

You must respond with valid structured output adhering to the schema."""


def write_research_brief(state: ResearchState) -> dict:
    """Analyze the refined topic and construct a comprehensive ResearchBrief."""
    llm = get_supervisor_llm()
    structured_llm = llm.with_structured_output(ResearchBrief, method="json_schema")

    topic = state.get("topic", "")
    logger.info(f"[write_research_brief] Formulating research brief for: '{topic}'")

    try:
        brief: ResearchBrief = structured_llm.invoke([
            SystemMessage(content=BRIEF_SYSTEM_PROMPT),
            HumanMessage(content=f"Research topic: \"{topic}\""),
        ])
    except Exception as exc:
        logger.warning(f"Structured output failed for research brief ({exc}), creating baseline brief.")
        brief = ResearchBrief(
            original_topic=topic,
            refined_topic=topic,
            detected_language="en",
            research_questions=[
                ResearchQuestion(
                    question=f"What are the foundational concepts and latest developments in {topic}?",
                    search_queries_web=[f"{topic} overview advances", f"{topic} architecture benchmarks"],
                    search_queries_arxiv=[f"all:{topic}"],
                    priority=1,
                )
            ],
            report_structure=[
                "Executive Summary",
                "Theoretical Foundations",
                "Current Methodologies and Benchmarks",
                "Open Challenges and Future Directions",
            ],
            key_terms=[topic],
        )

    logger.info(
        f"[write_research_brief] Generated {len(brief.research_questions)} research questions in language '{brief.detected_language}'."
    )

    return {
        "research_brief": brief,
        "detected_language": brief.detected_language,
        "current_phase": "draft",
    }
