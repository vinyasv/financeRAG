"""
Rich-based terminal UI for Finance RAG.

Claude Code-inspired design: minimal, clean, no emojis.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.status import Status
from rich.text import Text
from rich.style import Style
from rich.theme import Theme
from typing import Any

# Custom theme - Claude Code inspired colors
# Muted palette with blue/cyan accents
THEME = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "green",
    "muted": "dim",
    "accent": "cyan",
    "highlight": "bold white",
    "panel.border": "dim blue",
})

# Create the main console instance
console = Console(theme=THEME, highlight=False)


def print_header(
    doc_count: int = 0,
    table_count: int = 0, 
    chunk_count: int = 0,
    model: str | None = None,
    provider: str | None = None
) -> None:
    """Print the application header with knowledge base stats."""
    lines = []
    
    if doc_count > 0:
        lines.append(f"Knowledge base: {doc_count} documents, {table_count} tables, {chunk_count:,} chunks")
    else:
        lines.append("Knowledge base: empty")
    
    if model:
        model_display = model if len(model) < 40 else model[:37] + "..."
        provider_str = f" ({provider})" if provider else ""
        lines.append(f"Model: {model_display}{provider_str}")
    
    content = "\n".join(lines)
    
    panel = Panel(
        content,
        title="Finance RAG",
        title_align="left",
        border_style="dim blue",
        padding=(0, 1),
    )
    console.print(panel)
    console.print()


def print_answer(answer: str, execution_time_ms: float | None = None) -> None:
    """Print a query answer in a panel."""
    panel = Panel(
        answer,
        title="Answer",
        title_align="left",
        border_style="dim blue",
        padding=(0, 1),
    )
    console.print(panel)
    
    if execution_time_ms is not None:
        console.print(f"  [muted]Completed in {execution_time_ms / 1000:.1f}s[/muted]")
    console.print()


def print_citations(citations: list[Any], max_citations: int = 5) -> None:
    """Print citations in a panel."""
    if not citations:
        return
    
    lines = []
    for cite in citations[:max_citations]:
        ref = cite.format_reference() if hasattr(cite, 'format_reference') else str(cite)
        lines.append(f"  - {ref}")
    
    if len(citations) > max_citations:
        lines.append(f"  [muted]... and {len(citations) - max_citations} more[/muted]")
    
    content = "\n".join(lines)
    
    panel = Panel(
        content,
        title="Sources",
        title_align="left",
        border_style="dim",
        padding=(0, 1),
    )
    console.print(panel)


def print_stats_table(stats: dict) -> None:
    """Print knowledge base statistics as a formatted table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="muted")
    table.add_column("Value", style="highlight")
    
    table.add_row("Documents", str(stats.get('document_count', 0)))
    table.add_row("  PDFs", str(stats.get('pdf_count', 0)))
    table.add_row("  Spreadsheets", str(stats.get('spreadsheet_sheet_count', 0)))
    table.add_row("SQL Tables", str(stats.get('table_count', 0)))
    table.add_row("Text Chunks", f"{stats.get('chunk_count', 0):,}")
    
    console.print()
    console.print(table)
    
    # List documents if available and not too many
    docs = stats.get('documents', [])
    if docs and len(docs) <= 10:
        console.print()
        console.print("[muted]Documents:[/muted]")
        for doc in docs:
            name = doc.get('filename', 'Unknown')
            pages = doc.get('pages', 0)
            console.print(f"  [dim]-[/dim] {name} [muted]({pages} pages)[/muted]")
    
    console.print()


def print_models_table(models: dict[str, str], category_order: list[tuple[str, list[str]]] | None = None) -> None:
    """Print available models as a formatted table."""
    if category_order:
        for category_name, model_names in category_order:
            console.print(f"\n[accent]{category_name}[/accent]")
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Shortcut", style="highlight", min_width=15)
            table.add_column("Full Name", style="muted")
            
            for name in model_names:
                if name in models:
                    table.add_row(name, models[name])
            
            console.print(table)
    else:
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("Shortcut", style="highlight")
        table.add_column("Full Model Name", style="muted")
        
        for short, full in sorted(models.items()):
            table.add_row(short, full)
        
        console.print(table)
    
    console.print()


def print_ingestion_summary(
    ingested: list[tuple[str, str]],
    skipped: list[tuple[str, str]],
    failed: list[tuple[str, str]],
    stats: dict
) -> None:
    """Print ingestion results summary."""
    console.print()
    
    # Summary counts
    total = len(ingested) + len(skipped) + len(failed)
    console.print(Panel(
        f"Processed {total} files: {len(ingested)} ingested, {len(skipped)} skipped, {len(failed)} failed",
        title="Ingestion Summary",
        title_align="left",
        border_style="dim blue",
        padding=(0, 1),
    ))
    
    # Ingested files
    if ingested:
        console.print()
        if len(ingested) <= 10:
            console.print("[success]Ingested:[/success]")
            for name, detail in ingested:
                console.print(f"  [dim]-[/dim] {name} [muted]({detail})[/muted]")
        else:
            console.print(f"[success]Ingested {len(ingested)} files[/success]")
    
    # Skipped files
    if skipped:
        console.print()
        console.print("[warning]Skipped:[/warning]")
        for name, reason in skipped[:5]:
            console.print(f"  [dim]-[/dim] {name} [muted]({reason})[/muted]")
        if len(skipped) > 5:
            console.print(f"  [muted]... and {len(skipped) - 5} more[/muted]")
    
    # Failed files
    if failed:
        console.print()
        console.print("[error]Failed:[/error]")
        for name, error in failed[:5]:
            console.print(f"  [dim]-[/dim] {name}: {error}")
        if len(failed) > 5:
            console.print(f"  [muted]... and {len(failed) - 5} more[/muted]")
    
    # Knowledge base stats
    console.print()
    table = Table(
        title="Knowledge Base",
        show_header=False,
        box=None,
        padding=(0, 2),
        title_style="dim",
    )
    table.add_column("Metric", style="muted")
    table.add_column("Value", style="highlight")
    
    table.add_row("Total Documents", str(stats.get('document_count', 0)))
    table.add_row("SQL Tables", str(stats.get('table_count', 0)))
    table.add_row("Text Chunks", f"{stats.get('chunk_count', 0):,}")
    
    console.print(table)
    console.print()


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[error]Error:[/error] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]Warning:[/warning] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[success]{message}[/success]")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[info]{message}[/info]")


def create_progress() -> Progress:
    """Create a progress bar for file processing."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def create_status(message: str) -> Status:
    """Create a status spinner for long operations."""
    return console.status(message, spinner="dots")
