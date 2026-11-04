"""Command Line Interface (CLI) for Deep Researcher."""

from __future__ import annotations

import argparse
import sys
from uuid import uuid4

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from deep_researcher.config.settings import (
    HEADLESS_MODE,
    LLM_PROVIDER,
    MAX_RESEARCH_ITERATIONS,
    SUPERVISOR_CONFIG,
    WORKER_CONFIG,
)
from deep_researcher.graph.main_graph import build_main_graph
from deep_researcher.utils.output import print_step_header, save_research_artifacts

console = Console()


def print_system_banner(topic: str, max_iterations: int) -> None:
    """Print the startup banner and operational parameters."""
    grid = Table.grid(padding=1)
    grid.add_column(style="bold cyan")
    grid.add_column()

    grid.add_row("Provider:", f"[bold green]{LLM_PROVIDER.upper()}[/bold green]")
    grid.add_row("Supervisor Model:", f"{SUPERVISOR_CONFIG.model} (structured output & reasoning)")
    grid.add_row("Worker Model:", f"{WORKER_CONFIG.model} (tool calling)")
    grid.add_row("Max Iterations:", str(max_iterations))
    grid.add_row("Topic:", f"[bold white]{topic}[/bold white]")

    console.print(
        Panel(
            grid,
            title="[bold yellow]🔬 DEEP RESEARCHER — MULTI-AGENT ENGINE[/bold yellow]",
            subtitle=f"[dim]Hierarchical Delegation • LangGraph • {LLM_PROVIDER.title()} • Tavily • arXiv[/dim]",
            border_style="cyan",
        )
    )


def run_research(
    topic: str,
    max_iterations: int = MAX_RESEARCH_ITERATIONS,
    output_dir: str = "output",
) -> int:
    """Execute end-to-end research workflow."""
    print_system_banner(topic, max_iterations)

    graph = build_main_graph()
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [],
        "topic": topic,
        "research_brief": None,
        "draft_report": None,
        "detected_language": "en",
        "research_notes": [],
        "sources": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "current_phase": "clarify",
        "final_report": "",
        "sources_json": "",
    }

    console.print("\n[bold green]🚀 Initiating agent pipeline execution...[/bold green]\n")

    current_phase_seen = ""
    try:
        # Stream updates from the graph
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, state_update in event.items():
                phase = state_update.get("current_phase", "")
                if phase and phase != current_phase_seen:
                    current_phase_seen = phase
                    print_step_header(
                        title=f"Phase Transition: {phase.upper()}",
                        subtitle=f"Executed by: [{node_name}]",
                    )

                # Specialized terminal logs per node
                if node_name == "clarify_with_user":
                    refined = state_update.get("topic", topic)
                    console.print(f"  [cyan]Refined Topic:[/cyan] {refined}")

                elif node_name == "write_research_brief":
                    brief = state_update.get("research_brief")
                    if brief:
                        console.print(f"  [cyan]Research Brief Formulated:[/cyan] {len(brief.research_questions)} questions")
                        for q in brief.research_questions:
                            console.print(f"    • [P{q.priority}] {q.question}")

                elif node_name == "write_draft_report":
                    draft = state_update.get("draft_report")
                    if draft:
                        console.print(f"  [cyan]Draft Skeleton Created:[/cyan] {len(draft.sections)} sections planned")

                elif node_name == "supervisor_node":
                    # Check messages from supervisor
                    msgs = state_update.get("messages", [])
                    if msgs:
                        last_msg = msgs[-1].content
                        first_line = last_msg.split("\n")[0] if "\n" in last_msg else last_msg
                        console.print(f"  [yellow]Supervisor Directive:[/yellow] {first_line}")

                elif node_name in ("web_researcher", "paper_researcher"):
                    notes = state_update.get("research_notes", [])
                    sources = state_update.get("sources", [])
                    iter_count = state_update.get("iteration_count", 0)
                    console.print(
                        f"  [magenta]Evidence Saved (Iteration {iter_count}):[/magenta] +{len(notes)} notes, +{len(sources)} sources"
                    )

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user.[/bold red]")
        return 130
    except Exception as exc:
        console.print(f"\n[bold red]Execution error:[/bold red] {exc}", highlight=True)
        return 1

    # Retrieve terminal state
    final_snapshot = graph.get_state(config)
    final_values = final_snapshot.values if final_snapshot else {}

    report_md = final_values.get("final_report", "")
    sources_json = final_values.get("sources_json", "[]")

    if not report_md:
        console.print("[bold red]Error: No final report was generated.[/bold red]")
        return 1

    # Save deliverables
    report_path, sources_path = save_research_artifacts(
        report_md=report_md,
        sources_json_str=sources_json,
        topic=topic,
        base_dir=output_dir,
    )

    console.print("\n" + "=" * 70)
    console.print(f"[bold green]✓ Research Report Published:[/bold green] [underline]{report_path}[/underline]")
    console.print(f"[bold green]✓ Source Catalog Saved:[/bold green]     [underline]{sources_path}[/underline]")
    console.print("=" * 70 + "\n")

    # Display preview
    preview_length = min(len(report_md), 1500)
    console.print(
        Panel(
            Markdown(report_md[:preview_length] + ("\n\n*(Preview truncated — see full report file)*" if len(report_md) > 1500 else "")),
            title="📑 Report Preview",
            border_style="green",
        )
    )

    return 0


def main() -> None:
    """CLI argument parsing and execution."""
    parser = argparse.ArgumentParser(
        description="Deep Researcher: Autonomous hierarchical multi-agent research system."
    )
    parser.add_argument(
        "topic",
        nargs="?",
        type=str,
        help="Research topic or question to investigate.",
    )
    parser.add_argument(
        "--max-iterations",
        "-n",
        type=int,
        default=MAX_RESEARCH_ITERATIONS,
        help=f"Maximum research loop iterations (default: {MAX_RESEARCH_ITERATIONS})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Directory to save final markdown and JSON artifacts (default: output)",
    )

    args = parser.parse_args()

    if not args.topic:
        console.print(
            Panel(
                "[bold red]Missing research topic.[/bold red]\n\n"
                "Usage:\n"
                "  [cyan]deep-researcher \"Quantum computing error mitigation techniques\"[/cyan]\n"
                "  [cyan]deep-researcher \"Solid state battery commercialization\" --max-iterations 4[/cyan]\n",
                title="Deep Researcher",
                border_style="red",
            )
        )
        sys.exit(1)

    exit_code = run_research(
        topic=args.topic,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
