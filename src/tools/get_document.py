"""Tool for retrieving full document content."""

import re
from typing import Any

from .base import Tool
from ..models import ToolName
from ..storage.document_store import DocumentStore

# Regex for valid document IDs (hex hash strings or alphanumeric with underscores/hyphens)
VALID_DOC_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


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
            Document content and metadata
        """
        # Parse input
        parts = input_str.split(":", 1)
        doc_id = parts[0].strip()
        section = parts[1].strip() if len(parts) > 1 else None
        
        # Validate document ID format
        if not doc_id:
            return {"error": "Document ID is required"}
        if not VALID_DOC_ID_PATTERN.match(doc_id):
            return {"error": f"Invalid document ID format: {doc_id}"}
        
        # Get document
        doc = self.document_store.get_document(doc_id)
        
        if not doc:
            return {"error": f"Document not found: {doc_id}"}
        
        # Get content
        if section:
            content = self.document_store.get_section(doc_id, section)
            if not content:
                content = f"Section '{section}' not found"
        else:
            content = self.document_store.get_full_text(doc_id)
            if not content:
                content = "No content available"
        
        return {
            "document_id": doc.id,
            "filename": doc.filename,
            "title": doc.title,
            "section": section,
            "content": content,
            "page_count": doc.page_count
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

