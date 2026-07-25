#!/usr/bin/env python3
"""Per-page resilience tests for VLMTableExtractor.extract_tables_from_pdf.

A single transient page failure (429, timeout, bad JSON) must not discard the
pages that succeeded. These tests monkeypatch ``_process_page`` so no network
call or API key is required.
"""

import asyncio
import logging
import sys
from pathlib import Path

import fitz
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.exceptions import ExtractionFailed
from src.ingestion.vlm_extractor import VLMTableExtractor
from src.models import ExtractedTable

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def _make_pdf(tmp_path: Path, pages: int) -> Path:
    """Build a multi-page PDF by repeating the sample fixture."""
    target = tmp_path / f"multi_{pages}.pdf"
    with fitz.open(FIXTURE_PDF) as base, fitz.open() as out:
        for _ in range(pages):
            out.insert_pdf(base)
        out.save(target)
    return target


def _table(page_number: int, document_id: str) -> ExtractedTable:
    return ExtractedTable(
        id=f"{document_id}_p{page_number}_t0",
        document_id=document_id,
        table_name=f"table_page_{page_number}",
        page_number=page_number,
        schema_description=f"Table on page {page_number}",
        columns=["year", "revenue"],
        rows=[{"year": "2024", "revenue": "100"}],
        raw_text="| year | revenue |\n| --- | --- |\n| 2024 | 100 |\n",
    )


def _extractor() -> VLMTableExtractor:
    # Dummy key so the early no-key guard does not fire.
    return VLMTableExtractor(api_key="test-key-no-network")


def test_partial_page_failure_keeps_successful_pages(tmp_path, caplog):
    """Middle page raises -> pages 1 and 3 still come back, no exception."""
    pdf_path = _make_pdf(tmp_path, 3)
    extractor = _extractor()

    async def fake_process_page(doc, page_num: int, document_id: str):
        if page_num == 1:
            raise ExtractionFailed("simulated 429 on page 2")
        return [_table(page_num + 1, document_id)]

    extractor._process_page = fake_process_page

    with caplog.at_level("WARNING"):
        tables = asyncio.run(extractor.extract_tables_from_pdf(pdf_path, "doc-partial"))

    assert [t.page_number for t in tables] == [1, 3]
    assert "page 2" in caplog.text
    assert "simulated 429 on page 2" in caplog.text


def test_all_pages_failing_raises_extraction_failed(tmp_path):
    """Every page fails -> ExtractionFailed so the caller falls back to Docling."""
    pdf_path = _make_pdf(tmp_path, 3)
    extractor = _extractor()

    async def fake_process_page(doc, page_num: int, document_id: str):
        raise ExtractionFailed(f"simulated failure on page {page_num + 1}")

    extractor._process_page = fake_process_page

    with pytest.raises(ExtractionFailed):
        asyncio.run(extractor.extract_tables_from_pdf(pdf_path, "doc-all-fail"))


def test_no_tables_found_returns_empty_without_raising(tmp_path):
    """All pages succeed but hold no tables -> [] and no exception."""
    pdf_path = _make_pdf(tmp_path, 3)
    extractor = _extractor()

    async def fake_process_page(doc, page_num: int, document_id: str):
        return []

    extractor._process_page = fake_process_page

    tables = asyncio.run(extractor.extract_tables_from_pdf(pdf_path, "doc-empty"))

    assert tables == []


def test_max_tables_cap_still_applies(tmp_path):
    """The max_tables soft cap survives the resilience changes."""
    pdf_path = _make_pdf(tmp_path, 3)
    extractor = _extractor()

    async def fake_process_page(doc, page_num: int, document_id: str):
        return [_table(page_num + 1, document_id), _table(page_num + 1, document_id)]

    extractor._process_page = fake_process_page

    tables = asyncio.run(
        extractor.extract_tables_from_pdf(pdf_path, "doc-cap", max_tables=2)
    )

    assert len(tables) == 2


# =============================================================================
# Round-2 regressions.
#
# NOTE ON THE TESTS ABOVE: only test_partial_page_failure_keeps_successful_pages
# discriminates against the pre-fix code. The all-failed, no-tables and
# max_tables tests all PASS against the original bare-gather implementation
# (it already raised on total failure, already returned [], already sliced).
# They are kept as GUARDS against over-correction, not as evidence of the fix.
# =============================================================================

def test_majority_page_failure_raises_so_docling_fallback_runs(tmp_path):
    """The defect this closes: returning 1 table salvaged from 20 pages.

    RAGAgent._extract_pdf_tables only falls back to Docling on ExtractionFailed
    or an empty list, and _get_or_create_document then marks the document
    ingested -- so a near-empty extraction became permanent and invisible.
    """
    pdf_path = _make_pdf(tmp_path, 20)
    extractor = _extractor()

    async def fake_process_page(doc, page_num: int, document_id: str):
        if page_num == 0:
            return [_table(1, document_id)]
        raise ExtractionFailed("simulated outage")

    extractor._process_page = fake_process_page

    with pytest.raises(ExtractionFailed) as exc_info:
        asyncio.run(extractor.extract_tables_from_pdf(pdf_path, "doc_majority"))

    message = str(exc_info.value)
    assert "19/20" in message, f"Error should quantify the loss; got {message!r}"


def test_failure_below_threshold_returns_partial_with_warning(tmp_path, caplog):
    """A few transient 429s must NOT suppress an otherwise good extraction."""
    pdf_path = _make_pdf(tmp_path, 10)
    extractor = _extractor()

    async def fake_process_page(doc, page_num: int, document_id: str):
        if page_num in (3, 7):  # 2/10 = 20%, under the threshold
            raise ExtractionFailed("simulated 429")
        return [_table(page_num + 1, document_id)]

    extractor._process_page = fake_process_page

    with caplog.at_level(logging.WARNING, logger="src.ingestion.vlm_extractor"):
        tables = asyncio.run(extractor.extract_tables_from_pdf(pdf_path, "doc_partial"))

    assert len(tables) == 8, "The 8 successful pages must survive"

    records = [
        r.getMessage()
        for r in caplog.records
        if r.name == "src.ingestion.vlm_extractor"
    ]
    assert any("2/10 failed pages" in m for m in records), (
        f"Partial extraction must be announced; got {records!r}"
    )


def test_failed_page_number_is_logged_by_the_extractor_itself(tmp_path, caplog):
    """The page number must come from OUR log record, not the fake's message.

    The round-1 assertion was satisfied by the traceback rendering of the
    fake's own exception text, so a 0-based off-by-one went undetected.
    """
    pdf_path = _make_pdf(tmp_path, 10)
    extractor = _extractor()

    async def fake_process_page(doc, page_num: int, document_id: str):
        if page_num == 1:  # 0-based -> must be reported as page 2
            raise ExtractionFailed("boom")  # deliberately carries no page number
        return [_table(page_num + 1, document_id)]

    extractor._process_page = fake_process_page

    with caplog.at_level(logging.WARNING, logger="src.ingestion.vlm_extractor"):
        asyncio.run(extractor.extract_tables_from_pdf(pdf_path, "doc_logging"))

    messages = [
        r.getMessage()
        for r in caplog.records
        if r.name == "src.ingestion.vlm_extractor"
    ]
    assert any("page 2 of" in m for m in messages), (
        f"Expected 1-based page number from the extractor's own log; got {messages!r}"
    )
    assert not any("page 1 of" in m for m in messages), "0-based off-by-one"


def test_cancellation_propagates_instead_of_counting_as_page_failure(tmp_path):
    """`isinstance(result, BaseException)` swallowed CancelledError.

    Cancellation is control flow; reclassifying it as a recoverable page error
    let a cancelled extraction report partial success.
    """
    pdf_path = _make_pdf(tmp_path, 3)
    extractor = _extractor()

    async def fake_process_page(doc, page_num: int, document_id: str):
        if page_num == 1:
            raise asyncio.CancelledError()
        return [_table(page_num + 1, document_id)]

    extractor._process_page = fake_process_page

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(extractor.extract_tables_from_pdf(pdf_path, "doc_cancel"))
