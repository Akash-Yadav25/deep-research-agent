"""Output management and artifact serialization utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()


def save_research_artifacts(
    report_md: str,
    sources_json_str: str,
    topic: str,
    base_dir: str | Path = "output",
) -> tuple[Path, Path]:
    """Save the final research report and source bibliography to disk.

    Args:
        report_md: Polished research report in Markdown.
        sources_json_str: JSON string of sources catalog.
        topic: The research topic string.
        base_dir: Root directory for output artifacts.

    Returns:
        Tuple of (report_path, sources_path).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean topic for directory naming
    safe_topic = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in topic)[:40].strip("_")
    folder_name = f"{timestamp}_{safe_topic}" if safe_topic else timestamp

    output_dir = Path(base_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    sources_path = output_dir / "sources.json"
    try:
        # Validate and re-format JSON
        parsed = json.loads(sources_json_str)
        formatted_json = json.dumps(parsed, indent=2)
    except Exception:
        formatted_json = sources_json_str
    sources_path.write_text(formatted_json, encoding="utf-8")

    return report_path, sources_path


def print_step_header(title: str, subtitle: str = "") -> None:
    """Print an attractive step banner to the console."""
    msg = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        msg += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(msg, border_style="cyan"))
