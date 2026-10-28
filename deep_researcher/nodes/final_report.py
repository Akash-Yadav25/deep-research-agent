"""Final report generation node: synthesizes evidence, verifies citations, and produces deliverables."""

import logging
from langchain_core.messages import HumanMessage, SystemMessage

from deep_researcher.models.state import ResearchState
from deep_researcher.utils.citations import CitationManager
from deep_researcher.utils.fallbacks import get_supervisor_llm

logger = logging.getLogger(__name__)

FINAL_REPORT_SYSTEM_PROMPT = """You are an elite academic writer and technical analyst.
Your objective is to author an exhaustive, publication-grade research report based entirely on the gathered evidence notes and draft outline.

Strict Publication Standards:
1. Architecture: Follow the section headings outlined in the Draft Report or Research Brief.
2. In-text Citations: Whenever you state an empirical fact, statistic, theorem, benchmark, or finding, you MUST include the local source tag (e.g., [web-1], [arxiv-2]).
3. Accuracy: Synthesize factual information strictly from the provided notes. Never fabricate citations or URLs.
4. Depth: Provide an in-depth, rigorous treatise, not a superficial summary. Include an Executive Summary, Detailed Analysis per section, and Strategic Implications.
5. Language: Write the report in the requested language (ISO code: {language}).

Do NOT generate the final references section manually; our automated citation engine will remap local codes into sequential global citations ([1], [2], ...) and append the formal bibliography."""


def final_report_node(state: ResearchState) -> dict:
    """Synthesize evidence into a comprehensive report and remap local citations to global numbers."""
    logger.info("[final_report_node] Synthesizing final research report...")

    brief = state.get("research_brief")
    draft = state.get("draft_report")
    notes = state.get("research_notes", [])
    sources = state.get("sources", [])
    language = state.get("detected_language", "en")

    brief_str = brief.model_dump_json(indent=2) if brief else state.get("topic", "")
    draft_str = draft.model_dump_json(indent=2) if draft else ""

    # Compile all notes with their associated local source ids
    compiled_notes = []
    for idx, n in enumerate(notes, start=1):
        sources_tag = f"Sources: {', '.join(n.source_ids)}" if n.source_ids else "No explicit source"
        compiled_notes.append(f"### Finding {idx} ({sources_tag}):\n{n.content}")

    notes_text = "\n\n".join(compiled_notes) or "No research notes gathered."

    # Provide sources catalog so the LLM knows which local ID maps to what
    sources_catalog = []
    for s in sources:
        if s.source_type == "arxiv":
            sources_catalog.append(f"- [{s.local_id}] arXiv:{s.arxiv_id} \"{s.title}\" ({s.published_date})")
        else:
            sources_catalog.append(f"- [{s.local_id}] \"{s.title}\" ({s.url})")
    catalog_text = "\n".join(sources_catalog) if sources_catalog else "No external sources cataloged."

    user_prompt = (
        f"## Research Brief\n{brief_str}\n\n"
        f"## Skeleton Outline\n{draft_str}\n\n"
        f"## Available Source Catalog\n{catalog_text}\n\n"
        f"## Empirical Research Notes\n{notes_text}\n\n"
        "Please write the exhaustive research report now."
    )

    llm = get_supervisor_llm()
    system_prompt = FINAL_REPORT_SYSTEM_PROMPT.format(language=language)

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        raw_report = str(response.content).strip()
    except Exception as exc:
        logger.error(f"[final_report_node] LLM generation error: {exc}")
        raw_report = (
            f"# Research Report: {state.get('topic', '')}\n\n"
            f"## Executive Summary\n\nSynthesized findings based on {len(notes)} notes and {len(sources)} sources.\n\n"
            f"### Evidence Summary\n\n{notes_text}\n"
        )

    # Unified Citations Remapping Phase
    logger.info("[final_report_node] Remapping local citations to unified sequential global namespace...")
    citation_mgr = CitationManager(sources=sources)
    remapped_report = citation_mgr.remap_citations(raw_report)
    full_report = remapped_report + citation_mgr.build_reference_list()
    sources_json = citation_mgr.to_json()

    logger.info("[final_report_node] Report synthesis and bibliography generation complete.")

    return {
        "final_report": full_report,
        "sources_json": sources_json,
        "current_phase": "complete",
    }
