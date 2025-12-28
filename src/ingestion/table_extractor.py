"""Extract and structure tables from parsed PDFs."""

from typing import Any
import hashlib

from ..models import ExtractedTable, ColumnType
from .pdf_parser import ParsedPDF
from .utils import (
    parse_numeric,
    is_numeric,
    normalize_column_name,
    table_to_text,
    HEADER_ROW_MIN_FILL_RATIO,
    HEADER_ROW_TEXT_RATIO,
)


class TableExtractor:
    """
    Extract structured tables from parsed PDFs.
    
    Handles:
    - Table detection and parsing
    - Header row identification
    - Data type inference
    - Value normalization
    """
    
    def extract_tables(self, parsed_pdf: ParsedPDF, document_id: str) -> list[ExtractedTable]:
        """
        Extract all tables from a parsed PDF.
        
        Args:
            parsed_pdf: The parsed PDF document
            document_id: ID of the parent document
            
        Returns:
            List of extracted tables with structured data
        """
        tables = []
        
        for page_num, raw_table in parsed_pdf.all_tables:
            if len(raw_table) < 2:  # Need at least header + 1 data row
                continue
            
            table = self._process_table(raw_table, page_num, document_id)
            if table:
                tables.append(table)
        
        return tables
    
    def _process_table(
        self,
        raw_table: list[list[str]],
        page_number: int,
        document_id: str
    ) -> ExtractedTable | None:
        """Process a single raw table into an ExtractedTable."""
        
        if not raw_table:
            return None
        
        # Identify header row (usually first row with non-empty cells)
        header_idx = self._find_header_row(raw_table)
        if header_idx is None or header_idx >= len(raw_table) - 1:
            return None
        
        headers = raw_table[header_idx]
        
        # Clean and normalize headers
        columns = [self._normalize_header(h) for h in headers]
        
        # Filter out empty columns
        valid_cols = [i for i, col in enumerate(columns) if col]
        if not valid_cols:
            return None
        
        columns = [columns[i] for i in valid_cols]
        
        # Process data rows
        rows = []
        for row_idx in range(header_idx + 1, len(raw_table)):
            raw_row = raw_table[row_idx]
            
            # Skip empty rows
            if not any(raw_row):
                continue
            
            row_data = {}
            for i, col_idx in enumerate(valid_cols):
                if col_idx < len(raw_row):
                    value = raw_row[col_idx]
                    row_data[columns[i]] = self._parse_value(value)
            
            if row_data:
                rows.append(row_data)
        
        if not rows:
            return None
        
        # Generate table ID and name
        table_id = self._generate_table_id(document_id, page_number, columns)
        table_name = self._generate_table_name(columns, page_number)
        
        # Generate raw text representation
        raw_text = self._table_to_text(columns, rows)
        
        return ExtractedTable(
            id=table_id,
            document_id=document_id,
            table_name=table_name,
            page_number=page_number,
            schema_description=self._infer_schema_description(columns, rows),
            columns=columns,
            rows=rows,
            raw_text=raw_text
        )
    
    def _find_header_row(self, table: list[list[str]]) -> int | None:
        """Find the header row index (usually first row with mostly text)."""
        for i, row in enumerate(table):
            # Count non-empty cells
            non_empty = sum(1 for cell in row if cell and cell.strip())
            
            # Consider it a header if enough cells are non-empty
            if len(row) > 0 and non_empty >= len(row) * HEADER_ROW_MIN_FILL_RATIO:
                # Check if cells look like headers (more text than numbers)
                text_cells = sum(1 for cell in row if cell and not is_numeric(cell))
                if text_cells >= non_empty * HEADER_ROW_TEXT_RATIO:
                    return i
        
        return 0 if table else None
    
    def _normalize_header(self, header: str) -> str:
        """Normalize a header string to a valid column name."""
        return normalize_column_name(header)
    
    def _parse_value(self, value: str) -> Any:
        """Parse a cell value to appropriate type."""
        if not value or not value.strip():
            return None
        
        value = value.strip()
        
        # Try to parse as number
        numeric = parse_numeric(value)
        if numeric is not None:
            return numeric
        
        return value
    

    def _is_numeric(self, value: str) -> bool:
        """Check if a value appears to be numeric."""
        return is_numeric(value)
    
    def _generate_table_id(self, document_id: str, page_number: int, columns: list[str]) -> str:
        """Generate a unique table ID."""
        content = f"{document_id}:{page_number}:{':'.join(columns)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_table_name(self, columns: list[str], page_number: int) -> str:
        """Generate a descriptive table name."""
        # Use first few column names
        col_hint = "_".join(columns[:3])
        return f"table_p{page_number}_{col_hint}"[:50]
    
    def _infer_schema_description(self, columns: list[str], rows: list[dict]) -> str:
        """Infer a description of what the table contains."""
        # Simple heuristic based on column names
        col_names = ", ".join(columns[:5])
        row_count = len(rows)
        
        # Detect common table types
        col_set = set(c.lower() for c in columns)
        
        if any(term in col_set for term in ["revenue", "profit", "income", "expense"]):
            table_type = "financial data"
        elif any(term in col_set for term in ["date", "month", "quarter", "year"]):
            table_type = "time-series data"
        elif any(term in col_set for term in ["name", "id", "product", "item"]):
            table_type = "list/catalog"
        else:
            table_type = "data table"
        
        return f"{table_type.capitalize()} with {row_count} rows. Columns: {col_names}"
    
    def _table_to_text(self, columns: list[str], rows: list[dict]) -> str:
        """Convert table to readable text format."""
        return table_to_text(columns, rows)

