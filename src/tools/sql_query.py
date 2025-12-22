"""SQL query tool for structured data queries."""

from typing import Any
import json

from .base import Tool
from ..models import ToolName, SQLQueryResult
from ..storage.sqlite_store import SQLiteStore, SecurityError


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
    
    def __init__(self, sqlite_store: SQLiteStore | None = None, llm_client: Any = None):
        self.sqlite_store = sqlite_store or SQLiteStore()
        self.llm_client = llm_client
    
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
            
            # Format result
            if not rows:
                return {"columns": [], "rows": [], "message": "No results found"}
            
            columns = list(rows[0].keys()) if rows else []
            
            # If single row with single column, return just the value
            if len(rows) == 1 and len(columns) == 1:
                return rows[0][columns[0]]
            
            # If single row, return as flat dict
            if len(rows) == 1:
                return rows[0]
            
            return {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows)
            }
            
        except SecurityError as e:
            # Security violation - return sanitized error without SQL details
            return {"error": f"Query not allowed: {str(e)}"}
        except Exception as e:
            # Other errors - log internally but don't expose SQL to response
            return {"error": "Query execution failed. Please try rephrasing your question."}
    
    async def _generate_sql(self, query: str) -> str:
        """Convert natural language to SQL."""
        if self.llm_client:
            return await self._generate_sql_with_llm(query)
        else:
            return self._generate_sql_heuristic(query)
    
    async def _generate_sql_with_llm(self, query: str) -> str:
        """Use LLM to generate SQL."""
        schema = self.sqlite_store.get_schema_for_llm()
        
        prompt = NL_TO_SQL_PROMPT.format(
            schema=schema,
            query=query
        )
        
        response = await self.llm_client.generate(prompt)
        
        # Clean up response
        sql = response.strip()
        
        # Remove markdown code blocks if present
        if sql.startswith("```"):
            lines = sql.split("\n")
            # Remove first line (```sql or ```)
            lines = lines[1:]
            # Remove last line if it's just closing backticks
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            sql = "\n".join(lines)
        
        return sql.strip()
    
    def _generate_sql_heuristic(self, query: str) -> str:
        """
        Generate SQL using simple heuristics.
        
        This is a fallback when no LLM is available.
        """
        query_lower = query.lower()
        
        # Get available tables
        tables = self.sqlite_store.list_spreadsheet_tables()
        if not tables:
            return "SELECT 'No tables available' as message"
        
        # Find relevant table based on query keywords
        matching_table = None
        for table in tables:
            table_name = table['table_name'].lower()
            # Check if table name words appear in query
            for word in table_name.split('_'):
                if len(word) > 3 and word in query_lower:
                    matching_table = table
                    break
            if matching_table:
                break
        
        if not matching_table:
            # Default to first table
            matching_table = tables[0]
        
        table_name = matching_table['table_name']
        columns = matching_table.get('columns', ['*'])
        
        # Return simple SELECT
        return f"SELECT * FROM \"{table_name}\" LIMIT 20"


class SQLExecutor:
    """
    Direct SQL execution for when you have a SQL query.
    
    Use this for executing pre-generated or validated SQL.
    """
    
    def __init__(self, sqlite_store: SQLiteStore | None = None):
        self.sqlite_store = sqlite_store or SQLiteStore()
    
    def execute(self, sql: str) -> SQLQueryResult:
        """Execute a SQL query and return structured result."""
        try:
            rows = self.sqlite_store.execute_query(sql)
            columns = list(rows[0].keys()) if rows else []
            
            return SQLQueryResult(
                query=sql,
                columns=columns,
                rows=rows,
                row_count=len(rows)
            )
        except Exception as e:
            return SQLQueryResult(
                query=sql,
                columns=[],
                rows=[],
                row_count=0
            )

