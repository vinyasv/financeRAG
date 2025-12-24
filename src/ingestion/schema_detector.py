"""LLM-based schema detection for extracted tables."""

import json
import logging
import re
from typing import Any

from ..models import ExtractedTable, TableColumn, ColumnType

logger = logging.getLogger(__name__)


# Default prompt for schema detection
SCHEMA_DETECTION_PROMPT = """Analyze this financial table and provide improved names and structure.

Original table name: {table_name}
Original columns: {columns}
Sample data (first 5 rows):
{sample_rows}

TASK: Generate semantic, queryable names for this table and its columns.

CRITICAL RULES:
1. If a column name looks like a number (e.g., "27_414", "16_628"), it's likely a HEADER VALUE that got parsed as a column name. Look at the data pattern to determine what it represents (e.g., fiscal quarter like "q1_fy26", or a date like "apr_2025").
2. Column names should be lowercase_snake_case and SEMANTIC (describe what data is in the column).
3. The table_name should describe the content (e.g., "income_statement", "balance_sheet", "cash_flow", "revenue_by_segment").
4. Look at the first column - it often contains row labels (like "Revenue", "Net Income", "Operating Expenses").

TEMPORAL CONTEXT EXTRACTION:
5. If column headers contain fiscal periods (Q1, Q2, FY24, 2024, "Three months ended"), encode in column names:
   - Use format: q1_fy26, q4_fy25, annual_2024
6. Add a "temporal_context" field ONLY if fiscal period is clearly identifiable:
   - fiscal_year: The primary fiscal year (e.g., 2024)
   - fiscal_quarter: Q1/Q2/Q3/Q4 or "annual"
   - periods_covered: List of periods in the table (e.g., ["Q1 FY26", "Q4 FY25"])

Provide a JSON response with:
{{
  "table_name": "semantic_table_name",
  "table_description": "Clear description of what this table contains",
  "column_mappings": {{
    "original_column_name": "new_semantic_name",
    "27_414": "q1_fy26",
    ...
  }},
  "temporal_context": {{
    "fiscal_year": 2025,
    "fiscal_quarter": "Q1",
    "periods_covered": ["Q1 FY26", "Q4 FY25", "Q1 FY25"]
  }}
}}

EXAMPLE for a table with columns ["gaap_metric", "27_414", "16_628", "15_345"]:
If the data shows these are quarterly values for FY26 Q1, FY25 Q4, FY25 Q1:
{{
  "table_name": "cash_flow_statement",
  "table_description": "GAAP cash flow metrics by quarter",
  "column_mappings": {{
    "gaap_metric": "metric_name",
    "27_414": "q1_fy26",
    "16_628": "q4_fy25",
    "15_345": "q1_fy25"
  }},
  "temporal_context": {{
    "fiscal_year": 2026,
    "fiscal_quarter": "Q1",
    "periods_covered": ["Q1 FY26", "Q4 FY25", "Q1 FY25"]
  }}
}}

Respond with only valid JSON, no markdown or explanation:"""


class SchemaDetector:
    """
    Detect and enhance table schemas using LLM.
    
    Improves upon basic extraction by:
    - Inferring column semantics
    - Detecting data types
    - Generating meaningful descriptions
    """
    
    def __init__(self, llm_client: Any = None):
        """
        Initialize schema detector.
        
        Args:
            llm_client: Optional LLM client. If not provided, uses basic heuristics.
        """
        self.llm_client = llm_client
    
    async def detect_schema(self, table: ExtractedTable) -> ExtractedTable:
        """
        Enhance table with detected schema.
        
        Args:
            table: The extracted table
            
        Returns:
            Table with enhanced schema description
        """
        if self.llm_client:
            return await self._detect_with_llm(table)
        else:
            return self._detect_with_heuristics(table)
    
    async def _detect_with_llm(self, table: ExtractedTable) -> ExtractedTable:
        """Use LLM to detect schema and rename columns/table."""
        # Prepare sample data
        sample_rows = table.rows[:5]
        sample_text = "\n".join(
            str(row) for row in sample_rows
        )
        
        prompt = SCHEMA_DETECTION_PROMPT.format(
            table_name=table.table_name,
            columns=", ".join(table.columns),
            sample_rows=sample_text
        )
        
        try:
            # Call LLM
            response = await self.llm_client.generate(prompt)
            
            # Clean up response (remove markdown if present)
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                response = "\n".join(lines)
            
            # Parse response
            schema_data = json.loads(response)
            
            # Update table name if provided
            if "table_name" in schema_data:
                new_table_name = schema_data["table_name"]
                # Ensure valid table name format
                new_table_name = re.sub(r'[^a-z0-9_]', '_', new_table_name.lower())
                table.table_name = new_table_name
            
            # Update table description
            table.schema_description = schema_data.get(
                "table_description",
                table.schema_description
            )
            
            # Apply column mappings if provided
            column_mappings = schema_data.get("column_mappings", {})
            if column_mappings:
                table = self._apply_column_mappings(table, column_mappings)
            
            return table
            
        except Exception as e:
            # Fall back to heuristics on error
            logger.warning(f"LLM schema detection failed: {e}")
            return self._detect_with_heuristics(table)
    
    def _apply_column_mappings(
        self, 
        table: ExtractedTable, 
        mappings: dict[str, str]
    ) -> ExtractedTable:
        """Apply column name mappings to table."""
        # Create new column list with mapped names
        new_columns = []
        for col in table.columns:
            new_name = mappings.get(col, col)
            # Ensure valid column name
            new_name = re.sub(r'[^a-z0-9_]', '_', new_name.lower())
            new_columns.append(new_name)
        
        # Update rows with new column names
        new_rows = []
        for row in table.rows:
            new_row = {}
            for old_col, value in row.items():
                new_col = mappings.get(old_col, old_col)
                new_col = re.sub(r'[^a-z0-9_]', '_', new_col.lower())
                new_row[new_col] = value
            new_rows.append(new_row)
        
        # Update table
        table.columns = new_columns
        table.rows = new_rows
        
        # Update raw_text with new column names
        table.raw_text = self._table_to_text(new_columns, new_rows)
        
        return table
    
    def _table_to_text(self, columns: list[str], rows: list[dict]) -> str:
        """Convert table to readable text format."""
        lines = []
        lines.append(" | ".join(columns))
        lines.append("-" * len(lines[0]))
        for row in rows:
            row_values = [str(row.get(col, "")) for col in columns]
            lines.append(" | ".join(row_values))
        return "\n".join(lines)
    
    def _detect_with_heuristics(self, table: ExtractedTable) -> ExtractedTable:
        """Use heuristics to detect schema."""
        column_types = {}
        
        for col in table.columns:
            # Sample values for this column
            values = [row.get(col) for row in table.rows if row.get(col) is not None]
            column_types[col] = self._infer_column_type(col, values)
        
        # Generate enhanced description
        table.schema_description = self._generate_description(
            table.columns,
            column_types,
            len(table.rows)
        )
        
        return table
    
    def _infer_column_type(self, column_name: str, values: list[Any]) -> ColumnType:
        """Infer the type of a column from its values."""
        if not values:
            return ColumnType.TEXT
        
        # Check column name for hints
        name_lower = column_name.lower()
        
        if any(term in name_lower for term in ["date", "time", "month", "year"]):
            return ColumnType.DATE
        
        if any(term in name_lower for term in ["price", "cost", "revenue", "profit", "amount", "total"]):
            return ColumnType.CURRENCY
        
        if any(term in name_lower for term in ["percent", "rate", "pct", "%"]):
            return ColumnType.PERCENTAGE
        
        # Check values
        numeric_count = sum(1 for v in values if isinstance(v, (int, float)))
        
        if numeric_count > len(values) * 0.8:
            # Mostly numeric
            # Check if values look like currency
            str_values = [str(v) for v in values[:5]]
            if any("$" in s or s.replace(",", "").replace(".", "").isdigit() for s in str_values):
                # Could be currency
                avg_value = sum(v for v in values if isinstance(v, (int, float))) / max(1, numeric_count)
                if avg_value > 100:  # Arbitrary threshold for "currency-like" values
                    return ColumnType.CURRENCY
            
            return ColumnType.NUMBER
        
        return ColumnType.TEXT
    
    def _generate_description(
        self,
        columns: list[str],
        column_types: dict[str, ColumnType],
        row_count: int
    ) -> str:
        """Generate a description based on detected types."""
        # Detect table type
        currency_cols = [c for c, t in column_types.items() if t == ColumnType.CURRENCY]
        date_cols = [c for c, t in column_types.items() if t == ColumnType.DATE]
        
        if currency_cols:
            table_type = "Financial data"
            key_cols = currency_cols[:3]
        elif date_cols:
            table_type = "Time-series data"
            key_cols = date_cols[:2] + [c for c in columns if c not in date_cols][:1]
        else:
            table_type = "Data table"
            key_cols = columns[:3]
        
        col_desc = ", ".join(key_cols)
        
        return f"{table_type} with {row_count} rows. Key columns: {col_desc}"


async def detect_schemas_batch(
    tables: list[ExtractedTable],
    llm_client: Any = None
) -> list[ExtractedTable]:
    """
    Detect schemas for multiple tables.
    
    Args:
        tables: List of extracted tables
        llm_client: Optional LLM client
        
    Returns:
        Tables with enhanced schemas
    """
    detector = SchemaDetector(llm_client)
    
    enhanced = []
    for table in tables:
        enhanced_table = await detector.detect_schema(table)
        enhanced.append(enhanced_table)
    
    return enhanced

