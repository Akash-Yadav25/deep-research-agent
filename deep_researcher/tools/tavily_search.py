"""Tavily web search and content extraction tools for research workers."""

import logging
from langchain_core.tools import tool
from tavily import TavilyClient

from deep_researcher.config.settings import (
    TAVILY_API_KEY,
    TAVILY_MAX_RESULTS,
    TAVILY_SEARCH_DEPTH_DEEP,
    TAVILY_SEARCH_DEPTH_EXPLORATORY,
)

logger = logging.getLogger(__name__)


def _get_client() -> TavilyClient | None:
    """Retrieve Tavily client if API key is present."""
    if not TAVILY_API_KEY or TAVILY_API_KEY.startswith("tvly-xxxx"):
        return None
    try:
        return TavilyClient(api_key=TAVILY_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize TavilyClient: {e}")
        return None


@tool
def search_web(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
) -> str:
    """Search the web for up-to-date facts, analysis, and primary sources on a topic.

    Args:
        query: The search query string.
        search_depth: 'basic' (fast discovery) or 'advanced' (in-depth analysis).
        max_results: Maximum number of links/summaries to return (1-10).

    Returns:
        Structured text with titles, URLs, relevance scores, and content snippets.
    """
    client = _get_client()
    if client is None:
        return (
            "Error: Tavily API key is missing or not set. "
            "Please provide a valid TAVILY_API_KEY in the environment or .env file to enable web search."
        )

    try:
        depth = (
            TAVILY_SEARCH_DEPTH_DEEP
            if search_depth == "advanced"
            else TAVILY_SEARCH_DEPTH_EXPLORATORY
        )
        limit = min(max(1, max_results), TAVILY_MAX_RESULTS)

        response = client.search(
            query=query,
            search_depth=depth,
            max_results=limit,
            include_answer=True,
            include_raw_content=False,
        )

        formatted_items = []
        if response.get("answer"):
            formatted_items.append(f"Synthesized Answer: {response['answer']}\n")

        for idx, item in enumerate(response.get("results", []), start=1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            score = item.get("score", "N/A")
            content = item.get("content", "").strip()

            formatted_items.append(
                f"[{idx}] Title: {title}\n"
                f"    URL: {url}\n"
                f"    Score: {score}\n"
                f"    Content: {content}\n"
            )

        if not formatted_items:
            return f"No web search results found for query: '{query}'"

        return "\n".join(formatted_items)
    except Exception as exc:
        logger.warning(f"Tavily search error: {exc}")
        return f"Web search failed: {str(exc)}"


@tool
def extract_webpage_content(urls: list[str]) -> str:
    """Extract clean, markdown-formatted text content from target URLs.

    Args:
        urls: List of web URLs to extract text from (max 5 URLs).

    Returns:
        Aggregated markdown content extracted from each URL.
    """
    client = _get_client()
    if client is None:
        return (
            "Error: Tavily API key is missing or not set. "
            "Please provide TAVILY_API_KEY to enable webpage extraction."
        )

    try:
        clean_urls = [u for u in urls if u and isinstance(u, str)][:5]
        if not clean_urls:
            return "No valid URLs provided for extraction."

        response = client.extract(urls=clean_urls)

        results = []
        for item in response.get("results", []):
            url = item.get("url", "")
            raw_content = item.get("raw_content", "")
            # Truncate to 3000 chars to avoid overflowing worker model context
            snippet = raw_content[:3000].strip() if raw_content else "No content extracted."
            results.append(f"URL: {url}\nContent:\n{snippet}\n")

        for fail in response.get("failed_results", []):
            url = fail.get("url", "")
            err = fail.get("error", "Extraction failed")
            results.append(f"URL: {url}\nStatus: Failed ({err})\n")

        return "\n---\n".join(results) if results else "No content returned from extraction."
    except Exception as exc:
        logger.warning(f"Tavily extraction error: {exc}")
        return f"Web extraction failed: {str(exc)}"
