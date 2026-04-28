"""Tests for refactored RAGAgent ingestion helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.models import Document
from src.rag_agent import RAGAgent


def make_agent() -> RAGAgent:
    """Create a minimally initialized RAGAgent for helper tests."""
    agent = RAGAgent.__new__(RAGAgent)
    agent.sqlite_store = MagicMock()
    agent.document_store = MagicMock()
    agent.chroma_store = MagicMock()
    agent.schema_cluster_manager = MagicMock()
    return agent


def test_get_or_create_document_persists_new_document():
    """Shared document creation helper should persist new documents once."""
    agent = make_agent()
    agent.sqlite_store.get_document.return_value = None
    agent._persist_document_payload = MagicMock()

    document, created = agent._get_or_create_document(
        "doc_123",
        filename="report.pdf",
        title="Report",
        page_count=5,
        metadata={"source": "pdf"},
        source_path=MagicMock(),
        full_text="hello",
        kind="Document",
        display_name="report.pdf",
    )

    assert created is True
    assert isinstance(document, Document)
    agent._persist_document_payload.assert_called_once()


def test_get_or_create_document_reuses_existing_document():
    """Shared document creation helper should preserve dedupe behavior."""
    agent = make_agent()
    existing = Document(id="doc_123", filename="report.pdf")
    agent.sqlite_store.get_document.return_value = existing
    agent.document_store.get_full_text.return_value = "hello"
    agent.chroma_store.count.return_value = 1
    agent._persist_document_payload = MagicMock()

    document, created = agent._get_or_create_document(
        "doc_123",
        filename="report.pdf",
        title="Report",
        page_count=5,
        metadata={},
        source_path=MagicMock(),
        full_text="hello",
        kind="Document",
        display_name="report.pdf",
    )

    assert created is False
    assert document is existing
    agent._persist_document_payload.assert_not_called()


def test_save_spreadsheet_table_routes_native_and_fallback_paths():
    """Spreadsheet table persistence should still support both storage paths."""
    agent = make_agent()
    agent._assign_cluster_safe = AsyncMock()

    native_sheet = SimpleNamespace(
        dataframe=object(),
        sheet_name="Revenue",
        headers=["revenue"],
        row_count=1,
        col_count=1,
        rows=[{"revenue": 1}],
    )
    asyncio.run(
        agent._save_spreadsheet_table(
            doc_id="doc_native",
            source_document="report.xlsx",
            sheet=native_sheet,
            table_name="sheet_revenue",
        )
    )
    agent.sqlite_store.save_spreadsheet_native.assert_called_once()

    fallback_sheet = SimpleNamespace(
        dataframe=None,
        sheet_name="Revenue",
        headers=["revenue"],
        row_count=1,
        col_count=1,
        rows=[{"revenue": 1}],
    )
    asyncio.run(
        agent._save_spreadsheet_table(
            doc_id="doc_fallback",
            source_document="report.xlsx",
            sheet=fallback_sheet,
            table_name="sheet_revenue",
        )
    )
    agent.sqlite_store.save_table.assert_called_once()
    assert agent._assign_cluster_safe.await_count == 2


def test_build_spreadsheet_chunks_preserves_metadata_and_windows_rows():
    """Spreadsheet chunks should cover row windows with stable metadata."""
    agent = make_agent()
    rows = [{"a": i, "b": i * 2} for i in range(250)]
    sheet = SimpleNamespace(
        sheet_name="Revenue",
        rows=rows,
        row_count=len(rows),
        headers=["a", "b", "c", "d", "e", "f"],
    )

    chunks = agent._build_spreadsheet_chunks("doc_123", sheet)

    assert len(chunks) == 3
    chunk = chunks[0]
    assert chunk.document_id == "doc_123"
    assert chunk.section_title == "Revenue"
    assert chunk.start_line == 1
    assert chunk.end_line == 100
    assert chunk.metadata["columns"] == ["a", "b", "c", "d", "e"]
    assert chunks[-1].start_line == 201
    assert chunks[-1].end_line == 250
