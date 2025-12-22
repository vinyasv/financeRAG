"""Spreadsheet parsing for Excel and CSV files."""

from pathlib import Path
from dataclasses import dataclass
from typing import Any
import hashlib
import re

import pandas as pd


@dataclass
class ParsedSheet:
    """A parsed sheet from a spreadsheet."""
    sheet_name: str
    headers: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    col_count: int
    raw_text: str  # Text representation for vector search
    dataframe: Any = None  # Raw DataFrame for native SQL insertion


@dataclass
class ParsedSpreadsheet:
    """Complete parsed spreadsheet document."""
    filename: str
    sheets: list[ParsedSheet]
    metadata: dict[str, Any]


class SpreadsheetParser:
    """
    Parse Excel/CSV files to extract structured data.
    
    Supports:
    - .xlsx, .xls (via pandas + openpyxl)
    - .csv (via pandas)
    
    Each sheet is treated as a separate document for querying.
    """
    
    SUPPORTED_EXTENSIONS = {'.xlsx', '.xls', '.csv'}
    
    def parse(self, file_path: Path) -> ParsedSpreadsheet:
        """
        Parse a spreadsheet file.
        
        Args:
            file_path: Path to .xlsx, .xls, or .csv file
            
        Returns:
            ParsedSpreadsheet with all sheets
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")
        
        if ext == '.csv':
            return self._parse_csv(file_path)
        else:
            return self._parse_excel(file_path)
    
    def _parse_excel(self, file_path: Path) -> ParsedSpreadsheet:
        """Parse Excel workbook (all sheets)."""
        try:
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
        except ImportError:
            raise ImportError(
                "openpyxl is required for Excel parsing. "
                "Install with: pip install openpyxl"
            )
        
        sheets = []
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Skip empty sheets
            if df.empty:
                continue
                
            parsed_sheet = self._dataframe_to_sheet(df, sheet_name)
            sheets.append(parsed_sheet)
        
        return ParsedSpreadsheet(
            filename=file_path.name,
            sheets=sheets,
            metadata={
                "source_type": "excel",
                "sheet_count": len(sheets),
                "original_sheet_names": excel_file.sheet_names
            }
        )
    
    def _parse_csv(self, file_path: Path) -> ParsedSpreadsheet:
        """Parse CSV file as single sheet."""
        # Try to detect encoding
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')
        
        # Use filename (without extension) as sheet name
        sheet_name = file_path.stem
        
        return ParsedSpreadsheet(
            filename=file_path.name,
            sheets=[self._dataframe_to_sheet(df, sheet_name)],
            metadata={"source_type": "csv"}
        )
    
    def _dataframe_to_sheet(self, df: pd.DataFrame, sheet_name: str) -> ParsedSheet:
        """Convert pandas DataFrame to ParsedSheet."""
        # Clean column names
        df.columns = [self._clean_column_name(str(col)) for col in df.columns]
        headers = list(df.columns)
        
        # Convert to list of dicts, handling various data types
        rows = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in headers:
                value = row[col]
                row_dict[col] = self._clean_value(value)
            rows.append(row_dict)
        
        # Create text representation for vector search
        raw_text = self._create_text_representation(headers, rows, sheet_name)
        
        return ParsedSheet(
            sheet_name=sheet_name,
            headers=headers,
            rows=rows,
            row_count=len(rows),
            col_count=len(headers),
            raw_text=raw_text,
            dataframe=df  # Preserve for native SQL insertion
        )
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name for use in SQL and display."""
        name = str(name).strip()
        
        # Handle unnamed columns (pandas default for empty headers)
        if name.startswith('Unnamed:'):
            return f"column_{name.split(':')[1].strip()}"
        
        # Replace problematic characters but keep readable
        name = re.sub(r'[\n\r\t]+', ' ', name)
        name = ' '.join(name.split())  # Normalize whitespace
        
        return name if name else "unnamed"
    
    def _clean_value(self, value: Any) -> Any:
        """Clean and normalize a cell value."""
        # Handle NaN/None
        if pd.isna(value):
            return None
        
        # Handle numpy types
        if hasattr(value, 'item'):
            value = value.item()
        
        # Handle dates
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        
        # Handle strings
        if isinstance(value, str):
            value = value.strip()
            return value if value else None
        
        return value
    
    def _create_text_representation(
        self, 
        headers: list[str], 
        rows: list[dict], 
        sheet_name: str
    ) -> str:
        """Create searchable text from spreadsheet data."""
        lines = [
            f"Spreadsheet Sheet: {sheet_name}",
            f"Total Rows: {len(rows)}",
            f"Columns: {', '.join(headers)}",
            ""
        ]
        
        # Add sample rows for semantic search context
        sample_rows = rows[:25]  # First 25 rows
        for i, row in enumerate(sample_rows):
            row_values = []
            for h in headers:
                val = row.get(h)
                if val is not None:
                    row_values.append(f"{h}: {val}")
            if row_values:
                lines.append(f"Row {i+1}: {' | '.join(row_values)}")
        
        if len(rows) > 25:
            lines.append(f"... and {len(rows) - 25} more rows")
        
        # Add summary of numeric columns if present
        numeric_summary = self._get_numeric_summary(headers, rows)
        if numeric_summary:
            lines.append("")
            lines.append("Numeric Summary:")
            lines.extend(numeric_summary)
        
        return "\n".join(lines)
    
    def _get_numeric_summary(self, headers: list[str], rows: list[dict]) -> list[str]:
        """Generate summary statistics for numeric columns."""
        summary = []
        
        for header in headers:
            values = [row.get(header) for row in rows if row.get(header) is not None]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            if len(numeric_values) >= 3:  # Need at least 3 values for meaningful stats
                total = sum(numeric_values)
                avg = total / len(numeric_values)
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                summary.append(
                    f"  {header}: min={min_val:,.2f}, max={max_val:,.2f}, "
                    f"avg={avg:,.2f}, sum={total:,.2f}"
                )
        
        return summary[:5]  # Limit to 5 columns
    
    @staticmethod
    def generate_document_id(filename: str, sheet_name: str | None = None) -> str:
        """Generate unique document ID from filename and optional sheet name."""
        content = filename
        if sheet_name:
            content = f"{filename}:{sheet_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @staticmethod
    def sanitize_table_name(name: str) -> str:
        """Sanitize sheet name for use as SQL table name."""
        # Replace non-alphanumeric with underscore, lowercase
        sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', name.lower())
        sanitized = sanitized.strip('_')
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"sheet_{sanitized}"
        
        return sanitized[:50] if sanitized else "unnamed_sheet"
