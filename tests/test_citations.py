"""Unit tests for Unified CitationManager."""

import json
from deep_researcher.models.state import Source
from deep_researcher.utils.citations import CitationManager


def test_citation_remapping_single():
    """Test remapping of simple single citations."""
    s1 = Source(local_id="web-1", title="Web Source 1", url="https://web1.com", source_type="web")
    s2 = Source(local_id="arxiv-1", title="Arxiv Paper 1", url="https://arxiv.org/abs/2301", source_type="arxiv", arxiv_id="2301")

    mgr = CitationManager(sources=[s1, s2])
    text = "Recent benchmarks show massive gains [web-1] and theoretical proof is provided in [arxiv-1]."

    remapped = mgr.remap_citations(text)
    assert "[1]" in remapped
    assert "[2]" in remapped
    assert "[web-1]" not in remapped
    assert "[arxiv-1]" not in remapped


def test_citation_remapping_order_of_appearance():
    """Verify that citations are numbered strictly in order of appearance in text."""
    s1 = Source(local_id="web-1", title="Web 1", url="https://web1.com")
    s2 = Source(local_id="web-2", title="Web 2", url="https://web2.com")

    mgr = CitationManager(sources=[s1, s2])
    # web-2 appears BEFORE web-1 in the text
    text = "First we cite [web-2] and then later we cite [web-1]."

    remapped = mgr.remap_citations(text)
    # [web-2] should be [1], [web-1] should be [2]
    assert remapped == "First we cite [1] and then later we cite [2]."


def test_citation_composite_brackets():
    """Verify composite brackets such as [web-1, arxiv-2]."""
    s1 = Source(local_id="web-1", title="Web 1", url="https://web1.com")
    s2 = Source(local_id="arxiv-2", title="Arxiv 2", url="https://arxiv.org/abs/2")

    mgr = CitationManager(sources=[s1, s2])
    text = "Combined empirical and theoretical evidence [web-1, arxiv-2]."

    remapped = mgr.remap_citations(text)
    assert remapped == "Combined empirical and theoretical evidence [1, 2]."


def test_citation_remapping_unclosed_trailing_tag():
    """Verify auto-repair when text terminates with an unclosed citation marker (e.g. [web-18)."""
    s1 = Source(local_id="web-18", title="CoMIC Framework", url="https://arxiv.org/html/2606.00756v1")
    mgr = CitationManager(sources=[s1])

    # Truncated string missing the closing bracket
    text = "Aggregates cross-agent guidance keyed by semantic subgoal identifiers [web-18"
    remapped = mgr.remap_citations(text)
    assert remapped == "Aggregates cross-agent guidance keyed by semantic subgoal identifiers [1]"


def test_bibliography_and_json_export():
    """Verify formatted Reference List and JSON export."""
    s1 = Source(
        local_id="arxiv-1",
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        source_type="arxiv",
        arxiv_id="1706.03762",
        authors=["Vaswani", "Shazeer", "Parmar", "Uszkoreit"],
        published_date="2017-06-12",
    )
    s2 = Source(
        local_id="web-1",
        title="Tavily AI Search",
        url="https://tavily.com",
        source_type="web",
    )

    mgr = CitationManager(sources=[s1, s2])
    _ = mgr.remap_citations("Text citing [arxiv-1] and [web-1].")

    bib = mgr.build_reference_list()
    assert "## References" in bib
    assert "Vaswani" in bib
    assert "Attention Is All You Need" in bib
    assert "Tavily AI Search" in bib

    json_str = mgr.to_json()
    catalog = json.loads(json_str)
    assert len(catalog) == 2
    assert catalog[0]["global_id"] == 1
    assert catalog[1]["global_id"] == 2
