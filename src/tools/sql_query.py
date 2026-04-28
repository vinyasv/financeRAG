"""SQL query tool for structured data queries."""

import asyncio
import logging
import re
import sqlite3
from typing import Any

from ..common.json_utils import strip_markdown_fence
from ..models import ToolName
from ..storage.sqlite_store import SecurityError, SQLiteStore
from .base import Tool

logger = logging.getLogger(__name__)

# Timeout for LLM SQL generation (seconds)
LLM_TIMEOUT_SECONDS = 30


# Prompt for converting natural language to SQL
NL_TO_SQL_PROMPT = """Convert this natural language query to SQL.

Database Schema:
{schema}

DATA MODEL:
- All tables are standard SQL tables with proper columns
- Table names are prefixed with document source (e.g., berkshire_annual_performance, jpmorgan_balance_sheet)
- You can query them directly like regular SQL tables
- Column names are lowercase with underscores

EXAMPLES:

1. Get revenue from a financial table:
SELECT revenue FROM berkshire_annual_performance WHERE year = '2024'

2. Get stock data for a specific company:
SELECT date, close, volume FROM stock_details_5_years WHERE company = 'AAPL' ORDER BY date DESC LIMIT 10

3. Compare metrics across years:
SELECT year, revenue, net_income FROM jpmorgan_financial_summary ORDER BY year

4. Aggregate data:
SELECT company, AVG(close) as avg_price, MAX(close) as max_price 
FROM stock_details_5_years 
GROUP BY company

5. Filter with multiple conditions:
SELECT * FROM nvidia_quarterly_metrics WHERE quarter = 'Q1' AND year = '2025'

Natural Language Query: {query}

Return ONLY the SQL query, no explanation. Do not wrap in markdown code blocks:"""


class SQLQueryTool(Tool):
    """
    Query structured data using natural language.
    
    Converts natural language to SQL and executes against the SQLite database.
    """
    
    name = ToolName.SQL_QUERY
    description = "Query structured data extracted from documents. Use for numbers, metrics, comparisons."
    
    def __init__(
        self, 
        sqlite_store: SQLiteStore | None = None, 
        llm_client: Any = None,
        schema_cluster_manager: Any = None
    ):
        self.sqlite_store = sqlite_store or SQLiteStore()
        self.llm_client = llm_client
        self.schema_cluster_manager = schema_cluster_manager
    
    async def execute(self, input_str: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute a natural language query against structured data.
        
        Args:
            input_str: Natural language query (e.g., "Get Q3 revenue and costs")
            context: Optional context (not used for this tool)
            
        Returns:
            Query results as dict with columns and rows
        """
        # Generate SQL from natural language
        sql = await self._generate_sql(input_str)
        
        # Execute the SQL
        try:
            rows = self.sqlite_store.execute_query(sql)
            provenance = self._build_provenance(sql)
            
            # Format result
            if not rows:
                return {
                    "columns": [],
                    "rows": [],
                    "message": "No results found",
                    "__sql_provenance": provenance,
                }
            
            columns = list(rows[0].keys()) if rows else []
            
            # If single row with single column, return a dict so downstream
            # references like {step_id.value} resolve against a real field.
            # Returning a bare scalar here historically broke calculator
            # bindings because the executor/calculator could not navigate
            # into a non-dict context value.
            if len(rows) == 1 and len(columns) == 1:
                return {
                    "value": rows[0][columns[0]],
                    "column": columns[0],
                    "__sql_provenance": provenance,
                }

            # If single row, return as flat dict
            if len(rows) == 1:
                return {**rows[0], "__sql_provenance": provenance}
            
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "__sql_provenance": provenance,
            }
            
        except SecurityError as e:
            # Security violation - return sanitized error without SQL details
            return {"error": f"Query not allowed: {str(e)}"}
        except sqlite3.Error as e:
            # Database errors - log but don't expose SQL details
            logger.warning(f"SQL execution error for query '{input_str[:50]}...': {e}")
            return {"error": "Query execution failed. Please try rephrasing your question."}
        except Exception:
            # Unexpected errors - log and re-raise for debugging
            logger.exception(f"Unexpected error in SQL query tool for: {input_str[:50]}...")
            raise
    
    async def _generate_sql(self, query: str) -> str:
        """Convert natural language to SQL."""
        if self.llm_client:
            return await self._generate_sql_with_llm(query)
        raise RuntimeError("SQL query generation requires an LLM client")
    
    async def _generate_sql_with_llm(self, query: str) -> str:
        """Use LLM to generate SQL with clustered schema context."""
        # Use cluster manager for focused schema if available
        if self.schema_cluster_manager:
            schema = self.schema_cluster_manager.get_schemas_for_query(
                query=query,
                sqlite_store=self.sqlite_store
            )
        else:
            # Fallback: include all schemas (original behavior)
            schema = self.sqlite_store.get_schema_for_llm()
        
        prompt = NL_TO_SQL_PROMPT.format(
            schema=schema,
            query=query
        )
        
        try:
            response = await asyncio.wait_for(
                self.llm_client.generate(prompt),
                timeout=LLM_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.warning(f"LLM SQL generation timed out after {LLM_TIMEOUT_SECONDS}s")
            raise ValueError("SQL generation timed out. Please try a simpler query.")
        
        return strip_markdown_fence(response)

    def _build_provenance(self, sql: str) -> dict[str, Any]:
        """Attach source-table metadata for citation extraction."""
        table_names = self._extract_table_names(sql)
        if not table_names:
            return {"sql": sql, "tables": []}

        table_map = {
            table["table_name"]: table
            for table in self.sqlite_store.list_spreadsheet_tables()
        }
        return {
            "sql": sql,
            "tables": [
                {
                    "table_name": name,
                    "document_id": table_map.get(name, {}).get("document_id", "unknown"),
                    "columns": table_map.get(name, {}).get("columns", []),
                }
                for name in table_names
            ],
        }

    def _extract_table_names(self, sql: str) -> list[str]:
        """Extract FROM/JOIN table names from a generated SELECT."""
        names: list[str] = []
        for match in re.finditer(
            r'\b(?:FROM|JOIN)\s+(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*))',
            sql,
            re.IGNORECASE,
        ):
            name = match.group(1) or match.group(2)
            if name and name not in names:
                names.append(name)
        return names


class SQLExecutor:
    """
    Direct SQL execution for when you have a SQL query.
    
    Use this for executing pre-generated or validated SQL.
    """
    
    def __init__(self, sqlite_store: SQLiteStore | None = None):
        self.sqlite_store = sqlite_store or SQLiteStore()
    
    def execute(self, sql: str) -> dict[str, Any]:
        """Execute a SQL query and return structured result."""
        try:
            rows = self.sqlite_store.execute_query(sql)
            columns = list(rows[0].keys()) if rows else []
            
            return {"query": sql, "columns": columns, "rows": rows, "row_count": len(rows)}
        except SecurityError as e:
            logger.warning(f"Security error in SQLExecutor: {e}")
            return {"query": sql, "columns": [], "rows": [], "row_count": 0, "error": f"Query not allowed: {e}"}
        except sqlite3.Error as e:
            logger.warning(f"SQLExecutor database error: {e}")
            return {"query": sql, "columns": [], "rows": [], "row_count": 0, "error": f"Database error: {e}"}
        except Exception as e:
            logger.exception("Unexpected error in SQLExecutor")
            return {"query": sql, "columns": [], "rows": [], "row_count": 0, "error": f"Unexpected error: {e}"}
