"""Table extraction using thepipe for fast, accurate PDF parsing."""

import re
import hashlib
from pathlib import Path
from typing import Any
import os

from ..models import ExtractedTable
from ..config import config


class VisionTableExtractor:
    """
    Extract tables from PDFs using thepipe.
    
    thepipe uses VLMs to extract clean markdown from PDFs,
    which we then parse into structured data for SQL queries.
    
    This is ~5-6x faster than our previous per-table vision approach.
    """
    
    def __init__(self, llm_client: Any = None, vision_model: str | None = None):
        """
        Initialize the table extractor.
        
        Args:
            llm_client: LLM client (used to get API key for OpenRouter)
            vision_model: Model to use for vision (default from config)
        """
        self.llm_client = llm_client
        self.vision_model = vision_model or config.vision_model
        self._thepipe = None
        self._openai = None
    
    def _ensure_thepipe(self):
        """Lazy import thepipe."""
        if self._thepipe is None:
            from thepipe.scraper import scrape_file
            self._thepipe = scrape_file
    
    def _get_openai_client(self):
        """Get OpenAI client configured for OpenRouter."""
        if self._openai is None:
            from openai import OpenAI
            
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return None
            
            self._openai = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        return self._openai
    
    async def extract_tables_from_pdf(
        self,
        pdf_path: Path,
        document_id: str,
        max_tables: int = 20,
        timeout_per_table: float = 60.0  # Not used, kept for compatibility
    ) -> list[ExtractedTable]:
        """
        Extract all tables from a PDF using thepipe.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: ID of the parent document
            max_tables: Maximum number of tables to extract
            timeout_per_table: Not used (kept for API compatibility)
            
        Returns:
            List of ExtractedTable objects
        """
        self._ensure_thepipe()
        
        client = self._get_openai_client()
        
        print(f"    Extracting with thepipe (model: {self.vision_model})...")
        
        # Use thepipe to scrape the PDF
        if client and config.use_vision_tables:
            chunks = self._thepipe(
                filepath=str(pdf_path),
                openai_client=client,
                model=self.vision_model
            )
        else:
            # Without VLM, just extract text
            chunks = self._thepipe(filepath=str(pdf_path))
        
        print(f"    Extracted {len(chunks)} chunks from PDF")
        
        # Parse markdown tables from chunks
        tables = []
        table_index = 0
        
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.text:
                continue
            
            # Find markdown tables in the chunk
            chunk_tables = self._extract_markdown_tables(chunk.text)
            
            for table_data in chunk_tables:
                if table_index >= max_tables:
                    break
                
                # Convert to ExtractedTable
                table = self._create_extracted_table(
                    table_data=table_data,
                    document_id=document_id,
                    page_number=chunk_idx + 1,  # Approximate page number
                    table_index=table_index
                )
                
                if table and len(table.rows) > 0:
                    tables.append(table)
                    print(f"    ✓ Table {table_index + 1}: {len(table.rows)} rows, {len(table.columns)} columns")
                    table_index += 1
        
        if not tables:
            print("    No tables found in document")
        
        return tables
    
    def _extract_markdown_tables(self, text: str) -> list[dict]:
        """
        Extract markdown tables from text.
        
        Returns list of dicts with 'headers' and 'rows' keys.
        """
        tables = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for table header (line with |)
            if '|' in line and not line.startswith('|---'):
                # Check if next line is separator (|---|---|)
                if i + 1 < len(lines) and re.match(r'^[\s|:-]+$', lines[i + 1]):
                    table = self._parse_markdown_table(lines, i)
                    if table and table.get('headers') and table.get('rows'):
                        tables.append(table)
                    # Skip past the table
                    i += len(table.get('rows', [])) + 2  # header + separator + rows
                    continue
            
            i += 1
        
        return tables
    
    def _parse_markdown_table(self, lines: list[str], start_idx: int) -> dict:
        """Parse a markdown table starting at the given index."""
        # Parse header
        header_line = lines[start_idx].strip()
        headers = self._parse_table_row(header_line)
        
        if not headers:
            return {}
        
        # Skip separator line
        rows = []
        i = start_idx + 2  # Skip header and separator
        
        while i < len(lines):
            line = lines[i].strip()
            
            # End of table
            if not line or '|' not in line:
                break
            
            # Skip separator lines
            if re.match(r'^[\s|:-]+$', line):
                i += 1
                continue
            
            row_values = self._parse_table_row(line)
            
            if row_values and len(row_values) > 0:
                # Create row dict
                row = {}
                for j, header in enumerate(headers):
                    if j < len(row_values):
                        row[header] = row_values[j]
                    else:
                        row[header] = None
                rows.append(row)
            
            i += 1
        
        return {
            'headers': headers,
            'rows': rows
        }
    
    def _parse_table_row(self, line: str) -> list[str]:
        """Parse a single table row, extracting cell values."""
        # Remove leading/trailing |
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        
        # Split by |
        cells = [cell.strip() for cell in line.split('|')]
        
        return cells
    
    def _create_extracted_table(
        self,
        table_data: dict,
        document_id: str,
        page_number: int,
        table_index: int
    ) -> ExtractedTable | None:
        """Convert parsed table data to ExtractedTable model."""
        headers = table_data.get('headers', [])
        rows_data = table_data.get('rows', [])
        
        if not headers or not rows_data:
            return None
        
        # Normalize column names
        columns = [self._normalize_column_name(h) for h in headers]
        
        # Filter out empty columns
        valid_indices = [i for i, col in enumerate(columns) if col]
        if not valid_indices:
            return None
        
        columns = [columns[i] for i in valid_indices]
        original_headers = [headers[i] for i in valid_indices]
        
        # Process rows
        rows = []
        for row_data in rows_data:
            processed_row = {}
            for orig_header, col_name in zip(original_headers, columns):
                value = row_data.get(orig_header)
                processed_row[col_name] = self._process_value(value)
            
            # Only add row if it has at least one non-null value
            if any(v is not None for v in processed_row.values()):
                rows.append(processed_row)
        
        if not rows:
            return None
        
        # Generate table ID and name
        table_id = hashlib.sha256(
            f"{document_id}:{page_number}:{table_index}:{':'.join(columns[:5])}".encode()
        ).hexdigest()[:16]
        
        table_name = f"table_p{page_number}_{self._generate_table_name(columns)}"
        
        # Build raw text representation
        raw_text = self._build_raw_text(columns, rows)
        
        # Generate description
        description = self._generate_description(columns, rows)
        
        return ExtractedTable(
            id=table_id,
            document_id=document_id,
            table_name=table_name[:50],
            page_number=page_number,
            schema_description=description,
            columns=columns,
            rows=rows,
            raw_text=raw_text
        )
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize a column name to a valid identifier."""
        if not name:
            return ""
        
        # Convert to lowercase, replace non-alphanumeric with underscore
        normalized = re.sub(r'[^a-z0-9]+', '_', name.lower().strip())
        normalized = normalized.strip('_')
        
        # Limit length
        if len(normalized) > 50:
            normalized = normalized[:50]
        
        return normalized or ""
    
    def _process_value(self, value: Any) -> Any:
        """Process a cell value, parsing numbers where appropriate."""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return value
        
        if not isinstance(value, str):
            value = str(value)
        
        value = value.strip()
        
        if not value or value.lower() == 'none':
            return None
        
        # Try to parse as number
        numeric = self._parse_numeric(value)
        if numeric is not None:
            return numeric
        
        return value
    
    def _parse_numeric(self, value: str) -> float | None:
        """Try to parse a string as a number."""
        if not value:
            return None
        
        # Remove common formatting
        cleaned = value.replace(",", "").replace("$", "").replace("%", "").strip()
        
        # Handle accounting notation (negative in parentheses)
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        
        # Handle pts suffix (percentage points)
        cleaned = cleaned.replace(" pts", "").replace("pts", "")
        
        # Handle K/M/B suffixes
        multiplier = 1
        if cleaned.endswith("K"):
            multiplier = 1000
            cleaned = cleaned[:-1]
        elif cleaned.endswith("M"):
            multiplier = 1000000
            cleaned = cleaned[:-1]
        elif cleaned.endswith("B"):
            multiplier = 1000000000
            cleaned = cleaned[:-1]
        
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None
    
    def _generate_table_name(self, columns: list[str]) -> str:
        """Generate a descriptive table name from columns."""
        name_parts = [c for c in columns[:3] if c]
        return "_".join(name_parts)[:30]
    
    def _generate_description(self, columns: list[str], rows: list[dict]) -> str:
        """Generate a description of what the table contains."""
        col_names = ", ".join(columns[:5])
        row_count = len(rows)
        
        # Detect common table types
        col_set = set(c.lower() for c in columns)
        
        if any(term in col_set for term in ["revenue", "profit", "income", "expense", "margin"]):
            table_type = "Financial data"
        elif any(term in col_set for term in ["q1", "q2", "q3", "q4", "fy"]):
            table_type = "Quarterly financial data"
        elif any(term in col_set for term in ["date", "month", "quarter", "year"]):
            table_type = "Time-series data"
        else:
            table_type = "Data table"
        
        return f"{table_type} with {row_count} rows. Columns: {col_names}"
    
    def _build_raw_text(self, columns: list[str], rows: list[dict]) -> str:
        """Build a text representation of the table."""
        lines = []
        
        # Header
        lines.append(" | ".join(columns))
        lines.append("-" * len(lines[0]))
        
        # Rows (limit to first 20 for raw text)
        for row in rows[:20]:
            row_values = [str(row.get(col, "") or "") for col in columns]
            lines.append(" | ".join(row_values))
        
        if len(rows) > 20:
            lines.append(f"... ({len(rows) - 20} more rows)")
        
        return "\n".join(lines)
