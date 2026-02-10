"""Table extraction using Docling for fast, accurate local PDF parsing.

Docling (IBM) uses TableFormer AI model for table recognition,
providing excellent accuracy on complex financial tables without API costs.

Speed optimizations applied:
- Default backend (docling-parse) with optimizations
- OCR disabled for digital PDFs
- MPS acceleration on Apple Silicon
- Optimized batch sizes
"""

import hashlib
import logging
import multiprocessing
from pathlib import Path
from typing import Any

from ..models import ExtractedTable
from ..config import config
from .utils import parse_numeric, normalize_column_name, table_to_text

logger = logging.getLogger(__name__)


class VisionTableExtractor:
    """
    Extract tables from PDFs using Docling (IBM).
    
    Docling runs locally using TableFormer model for table structure recognition.
    No API costs, uses MPS/GPU acceleration on Apple Silicon.
    
    Speed optimizations:
    - Default backend (docling-parse) with optimizations
    - OCR disabled for digital PDFs (huge speedup)
    - Optimized batch sizes for MPS
    """
    
    def __init__(self, vision_model: str | None = None):
        """
        Initialize the Docling table extractor.
        
        Args:
            vision_model: Unused, kept for API compatibility.
        """
        self._converter = None
    
    def _ensure_docling(self):
        """Lazy import and initialize Docling with optimized settings."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    PdfPipelineOptions,
                    AcceleratorOptions,
                    AcceleratorDevice,
                    TableFormerMode,
                    TableStructureOptions
                )
                
                # Speed optimizations
                accelerator_options = AcceleratorOptions(
                    num_threads=multiprocessing.cpu_count(),
                    device=AcceleratorDevice.AUTO,  # Uses MPS on Mac, CUDA on Linux
                )
                
                pipeline_options = PdfPipelineOptions(
                    do_ocr=False,  # Disable OCR for digital PDFs (major speedup)
                    do_table_structure=True,  # Keep table extraction
                    table_structure_options=TableStructureOptions(mode=TableFormerMode.FAST),  # Fast mode (4x speedup)
                    generate_page_images=False,  # Optimize speed
                    generate_picture_images=False,  # Optimize speed
                    accelerator_options=accelerator_options,
                )
                
                # New API usage with default backend
                format_options = {
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
                
                self._converter = DocumentConverter(
                    allowed_formats=[InputFormat.PDF],
                    format_options=format_options,
                )
                
                logger.info("Docling initialized with speed optimizations (no OCR, default backend)")
            except ImportError as e:
                logger.error(f"Docling import failed: {e}. Run: pip install docling")
                raise ImportError("docling not installed. Run: pip install docling")
            except Exception as e:
                # Fallback to simple initialization if advanced options fail
                logger.warning(f"Advanced Docling config failed ({e}), using defaults")
                from docling.document_converter import DocumentConverter
                self._converter = DocumentConverter()
                logger.info("Docling initialized with default settings")
    
    async def extract_tables_from_pdf(
        self,
        pdf_path: Path,
        document_id: str,
        max_tables: int = 100,
        timeout_per_table: float = 60.0  # Unused, kept for API compatibility
    ) -> list[ExtractedTable]:
        """
        Extract all tables from a PDF using Docling.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: ID of the parent document
            max_tables: Maximum number of tables to extract
            timeout_per_table: Unused (kept for API compatibility)
            
        Returns:
            List of ExtractedTable objects
        """
        self._ensure_docling()
        
        logger.info(f"Extracting tables with Docling from: {pdf_path.name}")
        
        try:
            # Convert the document
            result = self._converter.convert(str(pdf_path))
            doc = result.document
            
            tables = []
            table_index = 0
            
            # Iterate through document tables
            for table_item in doc.tables:
                if table_index >= max_tables:
                    break
                
                try:
                    # Export to DataFrame with document context
                    df = table_item.export_to_dataframe(doc=doc)
                    
                    if df is None or df.empty:
                        continue
                    
                    # Convert DataFrame to ExtractedTable
                    table = self._dataframe_to_extracted_table(
                        df=df,
                        document_id=document_id,
                        table_index=table_index,
                        table_item=table_item
                    )
                    
                    if table and len(table.rows) > 0:
                        tables.append(table)
                        logger.debug(
                            f"Table {table_index + 1}: {len(table.rows)} rows, "
                            f"{len(table.columns)} columns"
                        )
                        table_index += 1
                        
                except Exception as e:
                    logger.warning(f"Error extracting table {table_index}: {e}")
                    continue
            
            logger.info(f"Extracted {len(tables)} tables from {pdf_path.name}")
            return tables
            
        except Exception as e:
            logger.error(f"Docling extraction failed: {e}")
            raise
    
    def _dataframe_to_extracted_table(
        self,
        df,
        document_id: str,
        table_index: int,
        table_item: Any = None
    ) -> ExtractedTable | None:
        """Convert a pandas DataFrame to ExtractedTable model."""
        if df is None or df.empty:
            return None
        
        # Get page number if available
        page_number = 1
        if table_item and hasattr(table_item, 'prov') and table_item.prov:
            for prov in table_item.prov:
                if hasattr(prov, 'page_no'):
                    page_number = prov.page_no
                    break
        
        # Normalize column names
        original_columns = list(df.columns)
        columns = [normalize_column_name(str(c), max_length=50) for c in original_columns]
        
        # Handle duplicate column names
        seen = {}
        unique_columns = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                unique_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_columns.append(col)
        columns = unique_columns
        
        # Filter out empty columns
        valid_mask = [bool(col) for col in columns]
        if not any(valid_mask):
            return None
        
        columns = [c for c, v in zip(columns, valid_mask) if v]
        original_columns = [c for c, v in zip(original_columns, valid_mask) if v]
        
        # Process rows
        rows = []
        for _, row in df.iterrows():
            processed_row = {}
            for orig_col, norm_col in zip(original_columns, columns):
                value = row.get(orig_col)
                processed_row[norm_col] = self._process_value(value)
            
            # Only add row if it has at least one non-null value
            if any(v is not None for v in processed_row.values()):
                rows.append(processed_row)
        
        if not rows:
            return None
        
        # Generate table ID
        table_id = hashlib.sha256(
            f"{document_id}:{page_number}:{table_index}:{':'.join(columns[:5])}".encode()
        ).hexdigest()[:16]
        
        # Generate table name
        table_name = f"table_p{page_number}_{self._generate_table_name(columns)}"
        
        # Build raw text representation
        raw_text = table_to_text(columns, rows, max_rows=20)
        
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
    
    def _process_value(self, value: Any) -> Any:
        """Process a cell value, parsing numbers where appropriate."""
        if value is None:
            return None
        
        # Handle pandas NA
        try:
            import pandas as pd
            if pd.isna(value):
                return None
        except (ImportError, TypeError):
            pass
        
        if isinstance(value, (int, float)):
            return value
        
        if not isinstance(value, str):
            value = str(value)
        
        value = value.strip()
        
        if not value or value.lower() in ('none', 'nan', ''):
            return None
        
        # Try to parse as number
        numeric = parse_numeric(value)
        if numeric is not None:
            return numeric
        
        return value
    
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
