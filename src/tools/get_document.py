"""Tool for retrieving full document content."""

from typing import Any

from ..models import ToolName
from ..storage.document_store import DocumentStore
from ..validation import MAX_PROMPT_CONTENT_LENGTH, validate_document_id
from .base import Tool

# Deliberately independent of how much was kept: a notice whose length varies
# with its own digits cannot be fitted inside a fixed budget in one pass.
# The exact original size is reported separately as "full_length".
TRUNCATION_NOTICE = (
    "\n\n[CONTENT TRUNCATED to stay within the prompt content limit. "
    "Request a specific section to see the rest.]"
)


def _truncate_content(content: str) -> tuple[str, bool, int]:
    """
    Cap content at MAX_PROMPT_CONTENT_LENGTH with a visible marker.

    Args:
        content: Raw document or section text

    Returns:
        Tuple of (possibly truncated content, whether it was truncated,
        original character count)
    """
    full_length = len(content)

    if full_length <= MAX_PROMPT_CONTENT_LENGTH:
        return content, False, full_length

    # The notice must fit INSIDE the budget, not be appended past it. Appending
    # it made the return value exceed MAX_PROMPT_CONTENT_LENGTH -- and for an
    # input just over the threshold, truncation produced a LARGER payload than
    # the original. Reserve room for the notice, then fill the remainder.
    kept = max(0, MAX_PROMPT_CONTENT_LENGTH - len(TRUNCATION_NOTICE))
    return content[:kept] + TRUNCATION_NOTICE, True, full_length


class GetDocumentTool(Tool):
    """
    Retrieve full document or section content.
    
    Use when you need complete context, not just snippets.
    """
    
    name = ToolName.GET_DOCUMENT
    description = "Get full document or section content. Use when you need complete context."
    
    def __init__(self, document_store: DocumentStore | None = None):
        self.document_store = document_store or DocumentStore()
    
    async def execute(self, input_str: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Retrieve document content.
        
        Args:
            input_str: Document ID, optionally with section.
                      Format: "doc_id" or "doc_id:section_name"
            context: Optional context (not used for this tool)
            
        Returns:
            Document content and metadata. Content is capped at
            MAX_PROMPT_CONTENT_LENGTH characters; when that happens a marker is
            appended and "truncated"/"full_length" report the original size.
        """
        # Parse input
        parts = input_str.split(":", 1)
        doc_id = parts[0].strip()
        section = parts[1].strip() if len(parts) > 1 else None
        
        # Validate document ID format
        if not doc_id:
            return {"error": "Document ID is required"}
        if not validate_document_id(doc_id):
            return {"error": f"Invalid document ID format: {doc_id}"}
        
        # Get document
        doc = self.document_store.get_document(doc_id)
        
        if not doc:
            return {"error": f"Document not found: {doc_id}"}
        
        # Get content. `retrieved` is None when nothing was found, so the
        # placeholder text below never masquerades as real document content --
        # full_length must describe what was RETRIEVED, not the length of a
        # "not found" placeholder.
        if section:
            retrieved = self.document_store.get_section(doc_id, section)
            placeholder = f"Section '{section}' not found"
        else:
            retrieved = self.document_store.get_full_text(doc_id)
            placeholder = "No content available"

        if not retrieved:
            return {
                "document_id": doc.id,
                "filename": doc.filename,
                "title": doc.title,
                "section": section,
                "content": placeholder,
                "page_count": doc.page_count,
                "truncated": False,
                "full_length": 0,
            }

        # Cap what goes downstream into the synthesis prompt
        content, truncated, full_length = _truncate_content(retrieved)

        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "title": doc.title,
            "section": section,
            "content": content,
            "page_count": doc.page_count,
            "truncated": truncated,
            "full_length": full_length
        }
    
    def list_documents(self) -> list[dict[str, Any]]:
        """List all available documents."""
        docs = self.document_store.list_documents()
        
        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "title": doc.title,
                "page_count": doc.page_count
            }
            for doc in docs
        ]
