"""Common CLI bootstrap and export utilities."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..config import config
from ..ui import console, print_error


def configure_cli_logging(log_name: str) -> logging.Logger:
    """Configure logging for a CLI command and return its logger."""
    log_dir = config.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / log_name)],
        force=True,
    )

    return logging.getLogger(__name__)


def silence_third_party_loggers(names: Iterable[str] = ("httpx", "chromadb")) -> None:
    """Silence noisy third-party loggers for CLI commands."""
    for name in names:
        logging.getLogger(name).setLevel(logging.WARNING)


def format_citations(citations) -> str:
    """Format citations as a semicolon-delimited string."""
    if not citations:
        return ""
    return "; ".join(cite.format_reference() for cite in citations[:5])


def _print_export_success(output_path: Path) -> None:
    console.print(f"\n[info]Results exported to:[/info] {output_path}")


def export_results_to_csv(output_path: Path, results: list[dict]) -> bool:
    """Export query results to CSV."""
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["query", "answer", "citations", "response_time_ms", "timestamp"],
            )
            writer.writeheader()
            writer.writerows(results)
    except (IOError, OSError) as exc:
        print_error(f"Failed to export CSV: {exc}")
        return False

    _print_export_success(output_path)
    return True


def export_results_to_json(output_path: Path, results: list[dict]) -> bool:
    """Export query results to JSON."""
    try:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
    except (IOError, OSError) as exc:
        print_error(f"Failed to export JSON: {exc}")
        return False

    _print_export_success(output_path)
    return True


def export_results_to_pdf(output_path: Path, results: list[dict]) -> bool:
    """Export query results to a PDF report."""
    try:
        from fpdf import FPDF
    except ImportError:
        print_error("PDF export requires fpdf2: pip install fpdf2")
        return False

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Finance RAG Query Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(10)

    for index, result in enumerate(results, 1):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f"Query {index}", ln=True, fill=True)

        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(0, 6, result.get("query", ""))
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "Answer:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        answer_safe = result.get("answer", "").encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 5, answer_safe)
        pdf.ln(3)

        if result.get("citations"):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Sources:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            citations_safe = result["citations"].encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, citations_safe)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 5, f"Response time: {result.get('response_time_ms', 0):.1f}ms", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(8)

    try:
        pdf.output(str(output_path))
    except (IOError, OSError) as exc:
        print_error(f"Failed to export PDF: {exc}")
        return False

    _print_export_success(output_path)
    return True
