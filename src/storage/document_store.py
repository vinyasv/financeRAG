"""Document store for full document content and metadata."""

import json
import logging
import re
from pathlib import Path
from typing import Any
import shutil

from ..models import Document, TextChunk
from ..config import config

logger = logging.getLogger(__name__)


# =============================================================================
# Security: Path Traversal Protection
# =============================================================================

class PathSecurityError(Exception):
    """Raised when a path security violation is detected."""
    pass


def secure_path_join(base_path: Path, untrusted_path: str) -> Path:
    """
    Safely join a base path with an untrusted filename.
    
    This prevents path traversal attacks by:
    1. Extracting only the filename (stripping directory components)
    2. Removing dangerous characters
    3. Validating the result stays within base_path
    
    Args:
        base_path: The trusted base directory
        untrusted_path: The untrusted filename from user input
        
    Returns:
        A safe Path object within base_path
        
    Raises:
        PathSecurityError: If the path is unsafe
    """
    if not untrusted_path:
        raise PathSecurityError("Empty path not allowed")
    
    # Extract just the filename, removing any directory components
    safe_name = Path(untrusted_path).name
    
    # Check for null bytes
    if '\x00' in safe_name:
        raise PathSecurityError("Null byte in filename")
    
    # Remove any remaining traversal patterns
    if '..' in safe_name:
        safe_name = safe_name.replace('..', '_')
    
    # Sanitize: only allow alphanumeric, hyphens, underscores, dots, spaces
    # This is strict but safe for document filenames
    if not re.match(r'^[\w\-. ]+$', safe_name):
        # Create a sanitized version
        safe_name = re.sub(r'[^\w\-. ]', '_', safe_name)
    
    if not safe_name or safe_name in ('.', '..'):
        raise PathSecurityError(f"Invalid filename: {untrusted_path}")
    
    # Construct full path
    full_path = (base_path / safe_name).resolve()
    base_resolved = base_path.resolve()
    
    # Verify the result is still within base_path
    if not str(full_path).startswith(str(base_resolved)):
        logger.warning(f"Path traversal attempt detected: {untrusted_path}")
        raise PathSecurityError(f"Path traversal detected: {untrusted_path}")
    
    return full_path


def validate_document_id(doc_id: str) -> bool:
    """
    Validate that a document ID is safe to use in filenames.
    
    Document IDs should be hex strings from hash functions.
    """
    if not doc_id:
        return False
    # Only allow alphanumeric and hyphens/underscores (typical for IDs)
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', doc_id))


class DocumentStore:
    """
    Simple file-based store for full document content.
    
    Stores:
    - Original document files
    - Extracted text content
    - Document metadata
    """
    
    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or config.documents_dir
        self.metadata_path = self.base_path / ".metadata"
        self.content_path = self.base_path / ".content"
        
        # Ensure directories exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.content_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Document Operations
    # =========================================================================
    
    def save_document(
        self,
        doc: Document,
        source_path: Path | None = None,
        full_text: str | None = None
    ) -> None:
        """
        Save a document with its content.
        
        Security: Uses secure path joining to prevent traversal attacks.
        
        Args:
            doc: Document metadata
            source_path: Path to original file (will be copied)
            full_text: Extracted text content
            
        Raises:
            PathSecurityError: If document ID or filename is unsafe
        """
        # Validate document ID
        if not validate_document_id(doc.id):
            raise PathSecurityError(f"Invalid document ID: {doc.id}")
        
        # Save metadata using validated ID
        metadata_file = self.metadata_path / f"{doc.id}.json"
        with open(metadata_file, "w") as f:
            json.dump(doc.model_dump(mode="json"), f, indent=2, default=str)
        
        # Copy original file if provided (use secure path joining)
        if source_path and source_path.exists():
            try:
                dest_path = secure_path_join(self.base_path, doc.filename)
                if not dest_path.exists():
                    shutil.copy2(source_path, dest_path)
            except PathSecurityError as e:
                logger.warning(f"Skipping file copy due to security error: {e}")
        
        # Save full text content
        if full_text:
            content_file = self.content_path / f"{doc.id}.txt"
            with open(content_file, "w") as f:
                f.write(full_text)
    
    def get_document(self, doc_id: str) -> Document | None:
        """Get document metadata by ID."""
        metadata_file = self.metadata_path / f"{doc_id}.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file) as f:
            data = json.load(f)
        
        return Document(**data)
    
    def get_full_text(self, doc_id: str) -> str | None:
        """Get the full extracted text of a document."""
        content_file = self.content_path / f"{doc_id}.txt"
        
        if not content_file.exists():
            return None
        
        with open(content_file) as f:
            return f.read()
    
    def get_section(self, doc_id: str, section_title: str) -> str | None:
        """
        Get a specific section from a document.
        
        This is a simple implementation that looks for section headers.
        """
        full_text = self.get_full_text(doc_id)
        if not full_text:
            return None
        
        lines = full_text.split("\n")
        in_section = False
        section_lines = []
        
        for line in lines:
            # Check if this is the section we're looking for
            if section_title.lower() in line.lower():
                in_section = True
                section_lines.append(line)
                continue
            
            # Check if we've hit a new section (simple heuristic)
            if in_section:
                # If line looks like a new section header, stop
                if line.strip() and line.strip().isupper() or line.startswith("#"):
                    break
                section_lines.append(line)
        
        return "\n".join(section_lines) if section_lines else None
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its content."""
        # Delete metadata
        metadata_file = self.metadata_path / f"{doc_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Delete content
        content_file = self.content_path / f"{doc_id}.txt"
        if content_file.exists():
            content_file.unlink()
    
    def list_documents(self) -> list[Document]:
        """List all documents."""
        documents = []
        
        for metadata_file in self.metadata_path.glob("*.json"):
            with open(metadata_file) as f:
                data = json.load(f)
            documents.append(Document(**data))
        
        return sorted(documents, key=lambda d: d.ingested_at, reverse=True)
    
    # =========================================================================
    # Search Operations
    # =========================================================================
    
    def search_content(self, query: str, doc_id: str | None = None) -> list[dict[str, Any]]:
        """
        Simple text search across document content.
        
        This is a basic implementation for fallback when vector search isn't suitable.
        """
        results = []
        
        if doc_id:
            docs = [self.get_document(doc_id)]
            docs = [d for d in docs if d is not None]
        else:
            docs = self.list_documents()
        
        query_lower = query.lower()
        
        for doc in docs:
            content = self.get_full_text(doc.id)
            if not content:
                continue
            
            # Find matching lines
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    # Get context (lines before and after)
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context = "\n".join(lines[start:end])
                    
                    results.append({
                        "document_id": doc.id,
                        "document_name": doc.filename,
                        "line_number": i + 1,
                        "content": context,
                        "match_line": line
                    })
        
        return results

