"""Write draft report node: generates skeleton report with hypotheses and open questions."""

import logging
from langchain_core.messages import HumanMessage, SystemMessage

from deep_researcher.models.schemas import DraftReport, DraftSection
from deep_researcher.models.state import ResearchState
from deep_researcher.utils.fallbacks import get_supervisor_llm

logger = logging.getLogger(__name__)

DRAFT_SYSTEM_PROMPT = """You are an academic monograph editor and lead author.
Given a detailed Research Brief, construct a skeleton draft report.

For each section in the report:
1. Provide a scholarly section heading.
2. Articulate a clear hypothesis or thesis statement for what the evidence is anticipated to show.
3. Formulate specific open empirical questions that domain researchers must gather evidence to answer or substantiate.

You must respond with valid structured output conforming to the DraftReport schema."""


def write_draft_report(state: ResearchState) -> dict:
    """Generate a skeleton draft report with section hypotheses and open investigation questions."""
    llm = get_supervisor_llm()
    structured_llm = llm.with_structured_output(DraftReport)

    brief = state.get("research_brief")
    if brief:
        brief_json = brief.model_dump_json(indent=2)
    else:
        brief_json = f'{{"topic": "{state.get("topic", "")}"}}'

    logger.info("[write_draft_report] Synthesizing skeleton draft report...")

    try:
        draft: DraftReport = structured_llm.invoke([
            SystemMessage(content=DRAFT_SYSTEM_PROMPT),
            HumanMessage(content=f"Research Brief:\n{brief_json}"),
        ])
    except Exception as exc:
        logger.warning(f"Structured output failed for draft report ({exc}), generating fallback draft.")
        sections = []
        headings = brief.report_structure if brief and brief.report_structure else [
            "Executive Overview",
            "State of the Art Analysis",
            "Empirical Evidence and Comparisons",
            "Conclusion and Future Roadmap",
        ]
        for heading in headings:
            sections.append(
                DraftSection(
                    heading=heading,
                    hypothesis=f"Key insights regarding {heading} will be supported by retrieved literature.",
                    open_questions=[
                        f"What are the latest empirical benchmarks for {heading}?",
                        f"What are the primary tradeoffs and limitations in {heading}?",
                    ],
                )
            )
        draft = DraftReport(
            title=brief.refined_topic if brief else state.get("topic", "Research Report"),
            abstract="Initial working draft synthesizing findings across web literature and academic preprints.",
            sections=sections,
        )

    logger.info(f"[write_draft_report] Skeleton report created with {len(draft.sections)} sections.")

    return {
        "draft_report": draft,
        "current_phase": "research",
    }
