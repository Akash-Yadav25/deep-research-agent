"""ArXiv academic research paper search tool for research workers."""

import logging
import arxiv
from langchain_core.tools import tool

from deep_researcher.config.settings import ARXIV_DELAY_SECONDS, ARXIV_MAX_RESULTS

logger = logging.getLogger(__name__)

# Respectful ArXiv client honoring rate-limit of >= 3s per call
_arxiv_client = arxiv.Client(
    page_size=20,
    delay_seconds=ARXIV_DELAY_SECONDS,
    num_retries=3,
)


@tool
def search_arxiv(
    query: str,
    max_results: int = 5,
) -> str:
    """Search the arXiv preprint repository for academic papers, preprints, and theorems.

    Supports advanced query syntax such as:
    - ti:"title text" (title match)
    - au:"author name" (author match)
    - abs:"abstract text" (abstract match)
    - cat:cs.AI or cat:cs.CL or cat:stat.ML (category match)
    - Boolean operators: AND, OR, ANDNOT (must be uppercase)

    Args:
        query: Academic search query (e.g., 'ti:reasoning AND cat:cs.AI').
        max_results: Maximum number of papers to retrieve (1-10).

    Returns:
        Structured text with titles, arXiv IDs, authors, publication dates, and abstracts.
    """
    try:
        limit = min(max(1, max_results), ARXIV_MAX_RESULTS)

        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )

        entries = []
        for paper in _arxiv_client.results(search):
            authors_list = [a.name for a in paper.authors]
            authors_str = ", ".join(authors_list[:4])
            if len(authors_list) > 4:
                authors_str += f" (+{len(authors_list) - 4} others)"

            pub_date = (
                paper.published.strftime("%Y-%m-%d")
                if paper.published
                else "Unknown date"
            )

            short_id = paper.get_short_id()
            pdf_url = paper.pdf_url or f"https://arxiv.org/pdf/{short_id}.pdf"
            abs_url = paper.entry_id or f"https://arxiv.org/abs/{short_id}"

            # Keep abstract snippet bounded to prevent overflowing worker LLM context
            abstract_snippet = paper.summary.replace("\n", " ").strip()[:600]

            entries.append(
                f"Title: {paper.title}\n"
                f"ArXiv ID: {short_id}\n"
                f"Published: {pub_date}\n"
                f"Authors: {authors_str}\n"
                f"Categories: {', '.join(paper.categories)}\n"
                f"URL: {abs_url}\n"
                f"PDF URL: {pdf_url}\n"
                f"Abstract: {abstract_snippet}...\n"
            )

        if not entries:
            return f"No arXiv papers found matching query: '{query}'"

        return "\n---\n".join(entries)
    except Exception as exc:
        logger.warning(f"ArXiv search error: {exc}")
        return f"ArXiv search failed: {str(exc)}"
