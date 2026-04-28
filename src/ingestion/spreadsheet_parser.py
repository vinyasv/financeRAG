"""Spreadsheet parsing for Excel and CSV files."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


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
        
        logger.info(f"Parsing spreadsheet: {file_path.name} (type: {ext})")
        
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
            preview = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=10)
            header_row = self._detect_header_row(preview)
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)
            
            # Skip empty sheets
            if df.empty:
                continue
                
            parsed_sheet = self._dataframe_to_sheet(df, sheet_name)
            sheets.append(parsed_sheet)
        
        logger.info(f"Parsed Excel file with {len(sheets)} sheets")
        
        return ParsedSpreadsheet(
            filename=file_path.name,
            sheets=sheets,
            metadata={
                "source_type": "excel",
                "sheet_count": len(sheets),
                "original_sheet_names": excel_file.sheet_names
            }
        )

    def _detect_header_row(self, preview: pd.DataFrame) -> int:
        """Infer a header row when workbooks have title rows above headers."""
        if preview.empty:
            return 0

        row_scores: list[tuple[int, int, int]] = []
        for idx, row in preview.iterrows():
            values = [v for v in row.tolist() if not pd.isna(v)]
            if not values:
                continue
            string_count = sum(1 for value in values if isinstance(value, str) and value.strip())
            unique_count = len({str(value).strip().lower() for value in values if str(value).strip()})
            row_scores.append((idx, len(values), string_count + unique_count))

        if not row_scores:
            return 0

        first_idx, first_non_null, _ = row_scores[0]
        for idx, non_null, score in row_scores[1:4]:
            if first_idx == 0 and first_non_null <= 2 and non_null >= 2 and score >= non_null:
                return idx

        return 0
    
    def _parse_csv(self, file_path: Path) -> ParsedSpreadsheet:
        """Parse CSV file as single sheet."""
        # Try to detect encoding
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')
        
        # Use filename (without extension) as sheet name
        sheet_name = file_path.stem
        
        logger.info(f"Parsed CSV file with {len(df)} rows")
        
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
    
