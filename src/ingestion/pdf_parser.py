"""PDF parsing with text and table extraction."""

from pathlib import Path
from dataclasses import dataclass
from typing import Any
import hashlib


@dataclass
class ParsedPage:
    """A parsed page from a PDF."""
    page_number: int
    text: str
    tables: list[list[list[str]]]  # List of tables, each table is list of rows, each row is list of cells
    width: float
    height: float
    line_count: int = 0  # Number of lines on this page
    start_line_offset: int = 0  # Cumulative line offset from start of document


@dataclass
class ParsedPDF:
    """Complete parsed PDF document."""
    filename: str
    page_count: int
    pages: list[ParsedPage]
    metadata: dict[str, Any]
    
    @property
    def full_text(self) -> str:
        """Get all text concatenated."""
        return "\n\n".join(page.text for page in self.pages)
    
    @property
    def all_tables(self) -> list[tuple[int, list[list[str]]]]:
        """Get all tables with their page numbers."""
        result = []
        for page in self.pages:
            for table in page.tables:
                result.append((page.page_number, table))
        return result


class PDFParser:
    """
    Parse PDFs to extract text and tables.
    
    Uses pdfplumber for reliable table extraction.
    """
    
    def __init__(self):
        self._pdfplumber = None
    
    def _ensure_pdfplumber(self):
        """Lazy import pdfplumber."""
        if self._pdfplumber is None:
            import pdfplumber
            self._pdfplumber = pdfplumber
    
    def parse(self, pdf_path: Path) -> ParsedPDF:
        """
        Parse a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedPDF with text and tables
        """
        self._ensure_pdfplumber()
        
        pages = []
        metadata = {}
        
        with self._pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata or {}
            cumulative_line_offset = 0
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text() or ""
                
                # Count lines on this page
                line_count = text.count('\n') + 1 if text.strip() else 0
                
                # Extract tables
                tables = []
                raw_tables = page.extract_tables() or []
                
                for raw_table in raw_tables:
                    if raw_table:
                        # Clean up table cells
                        cleaned_table = [
                            [self._clean_cell(cell) for cell in row]
                            for row in raw_table
                            if row  # Skip empty rows
                        ]
                        if cleaned_table:
                            tables.append(cleaned_table)
                
                pages.append(ParsedPage(
                    page_number=page_num,
                    text=text,
                    tables=tables,
                    width=float(page.width),
                    height=float(page.height),
                    line_count=line_count,
                    start_line_offset=cumulative_line_offset
                ))
                
                # Update cumulative offset for next page
                cumulative_line_offset += line_count
        
        return ParsedPDF(
            filename=pdf_path.name,
            page_count=len(pages),
            pages=pages,
            metadata=metadata
        )
    
    def _clean_cell(self, cell: Any) -> str:
        """Clean a table cell value."""
        if cell is None:
            return ""
        
        # Convert to string and clean whitespace
        text = str(cell).strip()
        
        # Replace multiple whitespace with single space
        text = " ".join(text.split())
        
        return text
    
    @staticmethod
    def generate_document_id(filename: str) -> str:
        """Generate a unique document ID from filename."""
        # Use hash of filename for consistent IDs
        hash_input = filename.encode()
        return hashlib.md5(hash_input).hexdigest()[:12]

