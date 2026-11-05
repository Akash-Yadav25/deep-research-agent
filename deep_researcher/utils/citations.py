"""Unified citation manager for cross-agent source remapping and publication."""

from __future__ import annotations

import json
import re
from typing import Sequence
from deep_researcher.models.state import Source


class CitationManager:
    """Manages local-to-global citation remapping and bibliography generation.

    Platform agents reference sources using local scoped identifiers:
    - [web-1], [web-2]
    - [arxiv-1], [arxiv-2]

    During final report publishing, this manager:
    1. Identifies all citations cited in the report text.
    2. Remaps cited local identifiers into sequentially ordered global numbers ([1], [2], ...).
    3. Handles composite tags such as [web-1, arxiv-2] or [web-1][web-2].
    4. Formats a structured markdown bibliography.
    5. Exports complete source metadata as JSON.
    """

    def __init__(self, sources: Sequence[Source] | None = None) -> None:
        self._sources_by_local_id: dict[str, Source] = {}
        self._local_to_global: dict[str, int] = {}
        self._global_to_source: dict[int, Source] = {}
        self._next_global_id = 1

        if sources:
            self.register_sources(sources)

    def register_sources(self, sources: Sequence[Source]) -> None:
        """Register sources into the manager's catalog."""
        for s in sources:
            clean_local_id = s.local_id.strip()
            if clean_local_id:
                # If already registered, update with richer metadata
                if clean_local_id in self._sources_by_local_id:
                    existing = self._sources_by_local_id[clean_local_id]
                    if not existing.title and s.title:
                        existing.title = s.title
                    if not existing.url and s.url:
                        existing.url = s.url
                    if not existing.content_snippet and s.content_snippet:
                        existing.content_snippet = s.content_snippet
                else:
                    self._sources_by_local_id[clean_local_id] = s

    def remap_citations(self, text: str) -> str:
        """Scan text, assign sequential global IDs in order of first appearance, and rewrite markers.

        Rewrites:
        - Single citations: [web-1] -> [1]
        - Composite citations: [web-1, arxiv-2] -> [1, 2]
        - Adjacent citations: [web-1][arxiv-2] -> [1][2]
        """
        # Auto-repair unclosed trailing citation tags (e.g. "[web-18" at end of line/string)
        text = re.sub(r"\[([a-zA-Z0-9_\-]+)(?=\s*\n|$)", r"[\1]", text)

        # Step 1: Discover all local tokens in order of appearance
        # Pattern matches bracketed strings that contain local prefixes like web-1 or arxiv-2
        pattern = r"\[([a-zA-Z0-9_\-\s,;]+)\]"

        def _tokenize(bracket_content: str) -> list[str]:
            tokens = re.split(r"[,;]+", bracket_content)
            return [t.strip() for t in tokens if t.strip()]

        for match in re.finditer(pattern, text):
            tokens = _tokenize(match.group(1))
            for tok in tokens:
                # Check if this token matches a known local_id or looks like a local id (prefix-digits)
                if tok in self._sources_by_local_id or re.match(r"^(web|arxiv|source)-\d+$", tok, re.IGNORECASE):
                    if tok not in self._local_to_global:
                        gid = self._next_global_id
                        self._next_global_id += 1
                        self._local_to_global[tok] = gid

                        # Update the Source object if registered
                        src = self._sources_by_local_id.get(tok)
                        if src:
                            src.global_id = gid
                            self._global_to_source[gid] = src
                        else:
                            # Synthesize a fallback source placeholder
                            placeholder = Source(
                                local_id=tok,
                                global_id=gid,
                                title=f"Source {tok}",
                                source_type="arxiv" if tok.startswith("arxiv") else "web",
                            )
                            self._sources_by_local_id[tok] = placeholder
                            self._global_to_source[gid] = placeholder

        # Step 2: Ensure any registered sources not cited yet also get a global ID (for completeness)
        for local_id, src in self._sources_by_local_id.items():
            if local_id not in self._local_to_global:
                gid = self._next_global_id
                self._next_global_id += 1
                self._local_to_global[local_id] = gid
                src.global_id = gid
                self._global_to_source[gid] = src

        # Step 3: Replace bracketed occurrences in text
        def _replace_match(match: re.Match) -> str:
            content = match.group(1)
            tokens = _tokenize(content)
            remapped_tokens: list[str] = []
            has_remapped = False

            for tok in tokens:
                if tok in self._local_to_global:
                    remapped_tokens.append(str(self._local_to_global[tok]))
                    has_remapped = True
                else:
                    remapped_tokens.append(tok)

            if has_remapped:
                # Format as [1, 2] or sorted unique list of integers if all were remapped
                try:
                    int_tokens = sorted(set(int(t) for t in remapped_tokens))
                    return f"[{', '.join(str(i) for i in int_tokens)}]"
                except ValueError:
                    return f"[{', '.join(remapped_tokens)}]"

            return match.group(0)

        return re.sub(pattern, _replace_match, text)

    def build_reference_list(self) -> str:
        """Construct the formatted Markdown References section."""
        if not self._global_to_source:
            return ""

        lines = ["\n\n---\n\n## References\n"]
        for gid in sorted(self._global_to_source.keys()):
            source = self._global_to_source[gid]
            if source.source_type == "arxiv":
                authors_str = (
                    ", ".join(source.authors[:3]) if source.authors else "Unknown Authors"
                )
                if len(source.authors) > 3:
                    authors_str += " et al."

                title = source.title or f"arXiv paper {source.arxiv_id}"
                arxiv_str = f"arXiv:{source.arxiv_id}" if source.arxiv_id else ""
                date_str = f" ({source.published_date})" if source.published_date else ""
                url_str = f" [{source.url}]({source.url})" if source.url else ""

                ref_entry = f"- [{gid}] {authors_str}. \"{title}\". {arxiv_str}{date_str}.{url_str}"
            else:
                title = source.title or source.url or f"Web Source {gid}"
                url_part = f" [{source.url}]({source.url})" if source.url else ""
                ref_entry = f"- [{gid}] \"{title}\".{url_part}"

            lines.append(ref_entry)

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize registered sources to JSON format."""
        sources_list = []
        for gid in sorted(self._global_to_source.keys()):
            src = self._global_to_source[gid]
            sources_list.append(src.to_dict())

        return json.dumps(sources_list, indent=2)
