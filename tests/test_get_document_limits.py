"""Tests for GetDocumentTool content size limits.

get_document feeds straight into the synthesis prompt, so both the full-text
and section paths must be capped at MAX_PROMPT_CONTENT_LENGTH with a visible
marker plus explicit truncated/full_length metadata.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.get_document import GetDocumentTool  # noqa: E402
from src.validation import MAX_PROMPT_CONTENT_LENGTH  # noqa: E402

# Generous allowance for the appended truncation marker
MARKER_ALLOWANCE = 500


class FakeDocument:
    """Stand-in for models.Document with only the fields the tool reads."""

    def __init__(self, doc_id: str, filename: str = "10k.pdf",
                 title: str = "Annual Report", page_count: int = 120):
        self.id = doc_id
        self.filename = filename
        self.title = title
        self.page_count = page_count


class FakeDocumentStore:
    """Minimal document store stub: get_document / get_full_text / get_section."""

    def __init__(self, doc: FakeDocument | None, full_text: str | None = None,
                 sections: dict[str, str] | None = None):
        self._doc = doc
        self._full_text = full_text
        self._sections = sections or {}

    def get_document(self, doc_id: str) -> FakeDocument | None:
        if self._doc is not None and self._doc.id == doc_id:
            return self._doc
        return None

    def get_full_text(self, doc_id: str) -> str | None:
        return self._full_text

    def get_section(self, doc_id: str, section_title: str) -> str | None:
        return self._sections.get(section_title)


def _make_tool(full_text: str | None = None,
               sections: dict[str, str] | None = None,
               doc: FakeDocument | None = None) -> GetDocumentTool:
    document = doc if doc is not None else FakeDocument("doc123")
    return GetDocumentTool(
        document_store=FakeDocumentStore(document, full_text, sections)
    )


def test_oversized_full_text_is_truncated():
    """A 10-K-sized document must come back bounded, flagged, and sized."""
    original = "A" * (MAX_PROMPT_CONTENT_LENGTH * 4)
    tool = _make_tool(full_text=original)

    result = asyncio.run(tool.execute("doc123"))

    assert result["truncated"] is True
    assert result["full_length"] == len(original)
    assert len(result["content"]) <= MAX_PROMPT_CONTENT_LENGTH + MARKER_ALLOWANCE
    # The retained prefix is the real document text
    assert result["content"].startswith("A" * 100)


def test_truncation_marker_is_visible_in_content():
    """Truncation must be announced in the content itself, not silent."""
    original = "B" * (MAX_PROMPT_CONTENT_LENGTH + 25_000)
    tool = _make_tool(full_text=original)

    result = asyncio.run(tool.execute("doc123"))

    assert "TRUNCATED" in result["content"].upper()
    # The exact original size is reported via full_length rather than inside
    # the notice: a notice whose length varies with its own digits cannot be
    # fitted inside a fixed character budget in a single pass.
    assert result["full_length"] == len(original)
    assert result["truncated"] is True


def test_short_document_returned_unchanged():
    """Content under the limit is passed through byte-for-byte."""
    original = "Revenue grew 12% year over year.\nOperating margin: 21.4%."
    tool = _make_tool(full_text=original)

    result = asyncio.run(tool.execute("doc123"))

    assert result["content"] == original
    assert result["truncated"] is False
    assert result["full_length"] == len(original)


def test_content_exactly_at_limit_not_truncated():
    """Boundary: exactly MAX_PROMPT_CONTENT_LENGTH chars is still whole."""
    original = "C" * MAX_PROMPT_CONTENT_LENGTH
    tool = _make_tool(full_text=original)

    result = asyncio.run(tool.execute("doc123"))

    assert result["truncated"] is False
    assert result["content"] == original
    assert result["full_length"] == MAX_PROMPT_CONTENT_LENGTH


def test_section_path_is_also_capped():
    """The 'doc_id:section' path must be capped too."""
    original = "D" * (MAX_PROMPT_CONTENT_LENGTH * 2)
    tool = _make_tool(sections={"Risk Factors": original})

    result = asyncio.run(tool.execute("doc123:Risk Factors"))

    assert result["section"] == "Risk Factors"
    assert result["truncated"] is True
    assert result["full_length"] == len(original)
    assert len(result["content"]) <= MAX_PROMPT_CONTENT_LENGTH + MARKER_ALLOWANCE


def test_short_section_returned_unchanged():
    """A small section is untouched and flagged as complete."""
    original = "Item 1A. Risk Factors\nCompetition remains intense."
    tool = _make_tool(sections={"Risk Factors": original})

    result = asyncio.run(tool.execute("doc123:Risk Factors"))

    assert result["content"] == original
    assert result["truncated"] is False
    assert result["full_length"] == len(original)


def test_existing_keys_preserved():
    """The executor/synthesizer contract keys must all still be present."""
    tool = _make_tool(full_text="short body")

    result = asyncio.run(tool.execute("doc123"))

    for key in ("document_id", "filename", "title", "section",
                "content", "page_count"):
        assert key in result

    assert result["document_id"] == "doc123"
    assert result["filename"] == "10k.pdf"
    assert result["title"] == "Annual Report"
    assert result["section"] is None
    assert result["page_count"] == 120


def test_missing_section_fallback_preserved():
    """Unknown section still yields the original placeholder message."""
    tool = _make_tool(sections={})

    result = asyncio.run(tool.execute("doc123:Nonexistent"))

    assert result["content"] == "Section 'Nonexistent' not found"
    assert result["truncated"] is False


def test_missing_full_text_fallback_preserved():
    """Empty full text still yields the original placeholder message."""
    tool = _make_tool(full_text=None)

    result = asyncio.run(tool.execute("doc123"))

    assert result["content"] == "No content available"
    assert result["truncated"] is False


def test_empty_doc_id_error_shape_unchanged():
    """Empty ID returns the original single-key error dict."""
    tool = _make_tool(full_text="x")

    result = asyncio.run(tool.execute(""))

    assert result == {"error": "Document ID is required"}


def test_invalid_doc_id_error_shape_unchanged():
    """Malformed IDs return the original single-key error dict."""
    tool = _make_tool(full_text="x")

    result = asyncio.run(tool.execute("../../../etc/passwd"))
    assert result == {"error": "Invalid document ID format: ../../../etc/passwd"}


def test_document_not_found_error_shape_unchanged():
    """A valid-looking but absent ID returns the original error dict."""
    tool = _make_tool(full_text="x")

    result = asyncio.run(tool.execute("nosuchdoc"))

    assert result == {"error": "Document not found: nosuchdoc"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Round-2 regressions.
#
# NOTE: the four error-shape tests above pass against pre-fix code. They are
# GUARDS required by the spec, not evidence of the cap.
# =============================================================================

def test_returned_content_is_genuinely_bounded_at_the_limit():
    """The marker used to be appended OUTSIDE the budget.

    Output ran to LIMIT + ~201 chars, and for an input just past the threshold
    truncation produced a LARGER payload than the original.
    """
    from src.tools.get_document import _truncate_content
    from src.validation import MAX_PROMPT_CONTENT_LENGTH as limit

    for size in (limit - 1, limit, limit + 1, limit + 2, 500_000):
        content, _, full_length = _truncate_content("x" * size)
        assert len(content) <= limit, (
            f"input {size} produced {len(content)} chars, over the {limit} cap"
        )
        assert full_length == size
        assert len(content) <= size, (
            f"truncating {size} chars produced a LARGER payload ({len(content)})"
        )


def test_full_length_is_zero_when_nothing_was_retrieved():
    """full_length used to report len('No content available') == 20."""
    class _EmptyStore:
        def get_document(self, doc_id):
            return type(
                "Doc", (), {"id": doc_id, "filename": "f.pdf", "title": "T", "page_count": 3}
            )()

        def get_full_text(self, doc_id):
            return None

        def get_section(self, doc_id, section):
            return None

    tool = GetDocumentTool(document_store=_EmptyStore())

    result = asyncio.run(tool.execute("doc_1"))
    assert result["content"] == "No content available"
    assert result["full_length"] == 0
    assert result["truncated"] is False

    section_result = asyncio.run(tool.execute("doc_1:Risk Factors"))
    assert section_result["full_length"] == 0
