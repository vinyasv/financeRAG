"""Regression tests for Phase 3/4 cleanup."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.ingestion.chunker import ChunkingConfig, SemanticChunker
from src.ingestion.exceptions import ExtractionFailed
from src.ingestion.pdf_parser import ParsedPage, ParsedPDF
from src.rag_agent import RAGAgent


def test_pdf_extraction_stops_after_docling_failure(monkeypatch):
    """The removed rule-based extractor must not be called after Docling fails."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    agent = RAGAgent.__new__(RAGAgent)
    agent.vlm_extractor = SimpleNamespace(
        extract_tables_from_pdf=AsyncMock(side_effect=ExtractionFailed("vlm down"))
    )
    agent.docling_extractor = SimpleNamespace(
        extract_tables_from_pdf=AsyncMock(side_effect=ExtractionFailed("docling down"))
    )

    result = asyncio.run(agent._extract_pdf_tables(object(), MagicMock(name="report.pdf"), "doc"))

    assert result == []
    assert not hasattr(agent, "table_extractor")


def test_docs_do_not_advertise_removed_rule_based_or_comparability():
    readme = Path("README.md").read_text()

    assert "rule-based" not in readme.lower()
    assert "three-tier" not in readme.lower()
    assert "comparability checker" not in readme.lower()


def test_large_paragraph_chunk_line_ranges_cover_trailing_lines():
    text = "\n".join(
        [
            "One two three four five six.",
            "Seven eight nine ten eleven twelve.",
            "Thirteen fourteen fifteen sixteen seventeen eighteen.",
            "Nineteen twenty twentyone twentytwo twentythree twentyfour.",
        ]
    )
    parsed = ParsedPDF(
        filename="sample.pdf",
        page_count=1,
        pages=[
            ParsedPage(
                page_number=1,
                text=text,
                tables=[],
                width=100,
                height=100,
                line_count=len(text.splitlines()),
                start_line_offset=0,
            )
        ],
        metadata={},
    )
    chunker = SemanticChunker(ChunkingConfig(max_chunk_size=12, chunk_overlap=0, min_chunk_size=1))

    chunks = chunker.chunk_document(parsed, "doc")

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 4
    for current, following in zip(chunks, chunks[1:]):
        assert current.end_line <= following.start_line


def test_splitlines_does_not_count_trailing_newline():
    text = "line one\nline two\n"

    assert len(text.splitlines()) == 2
