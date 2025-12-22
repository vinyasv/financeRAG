"""SQLite storage for structured data (extracted tables)."""

import sqlite3
import json
import re
import logging
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import pandas as pd

from ..models import Document, ExtractedTable, TableSchema
from ..config import config

logger = logging.getLogger(__name__)


# =============================================================================
# Security: SQL Injection Protection
# =============================================================================

class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


# Forbidden SQL keywords that could modify data or schema
FORBIDDEN_SQL_KEYWORDS = {
    'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE',
    'EXEC', 'EXECUTE', 'GRANT', 'REVOKE', 'ATTACH', 'DETACH', 'PRAGMA',
    'VACUUM', 'REINDEX', 'REPLACE', 'MERGE'
}

# Maximum result rows to prevent DoS
MAX_SQL_RESULT_ROWS = 10000


def validate_sql_query(sql: str) -> tuple[bool, str]:
    """
    Validate that a SQL query is safe to execute.
    
    Args:
        sql: The SQL query string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not sql or not sql.strip():
        return False, "Empty query"
    
    sql_stripped = sql.strip()
    sql_upper = sql_stripped.upper()
    
    # Must start with SELECT (case insensitive)
    if not sql_upper.startswith('SELECT'):
        return False, "Only SELECT queries are allowed"
    
    # Check for forbidden keywords
    # Use word boundaries to avoid false positives (e.g., "UPDATED_AT" column)
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        # Pattern: word boundary + keyword + word boundary (not part of identifier)
        pattern = rf'\b{keyword}\b'
        if re.search(pattern, sql_upper):
            return False, f"Forbidden keyword detected: {keyword}"
    
    # Check for multiple statements (semicolon followed by non-whitespace)
    # Allow trailing semicolon but not multiple statements
    semicolon_count = sql_stripped.rstrip(';').count(';')
    if semicolon_count > 0:
        return False, "Multiple SQL statements are not allowed"
    
    # Check for comment injection (could hide malicious code)
    if '--' in sql or '/*' in sql:
        return False, "SQL comments are not allowed"
    
    return True, ""


def add_limit_clause(sql: str, max_rows: int = MAX_SQL_RESULT_ROWS) -> str:
    """
    Add a LIMIT clause to SQL if not already present.
    
    This prevents unbounded result sets that could cause DoS.
    """
    sql_upper = sql.upper()
    
    # If already has LIMIT, don't add another
    if 'LIMIT' in sql_upper:
        return sql
    
    # Remove trailing semicolon if present
    sql_clean = sql.rstrip().rstrip(';')
    
    return f"{sql_clean} LIMIT {max_rows}"


class SQLiteStore:
    """
    SQLite-based storage for structured data.
    
    Stores:
    - Document metadata
    - Extracted tables with flexible schema
    - Table data as key-value pairs for query flexibility
    """
    
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or config.sqlite_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    title TEXT,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    page_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Extracted tables metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_tables (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    page_number INTEGER,
                    schema_description TEXT,
                    columns TEXT NOT NULL,
                    raw_text TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            
            # Table data - flexible key-value storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS table_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_id TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    column_name TEXT NOT NULL,
                    value TEXT,
                    numeric_value REAL,
                    FOREIGN KEY (table_id) REFERENCES extracted_tables(id)
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_table_data_table_id 
                ON table_data(table_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_table_data_column 
                ON table_data(column_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_extracted_tables_doc 
                ON extracted_tables(document_id)
            """)
            
            # Spreadsheet native tables metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spreadsheet_tables (
                    table_name TEXT PRIMARY KEY,
                    document_id TEXT,
                    columns TEXT NOT NULL,
                    row_count INTEGER DEFAULT 0,
                    source_type TEXT DEFAULT 'spreadsheet',
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # =========================================================================
    # Document Operations
    # =========================================================================
    
    def save_document(self, doc: Document) -> None:
        """Save a document record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents (id, filename, title, ingested_at, page_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                doc.id,
                doc.filename,
                doc.title,
                doc.ingested_at.isoformat(),
                doc.page_count,
                json.dumps(doc.metadata)
            ))
            conn.commit()
    
    def get_document(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return Document(
                id=row["id"],
                filename=row["filename"],
                title=row["title"],
                page_count=row["page_count"],
                metadata=json.loads(row["metadata"])
            )
    
    def list_documents(self) -> list[Document]:
        """List all documents."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY ingested_at DESC")
            
            return [
                Document(
                    id=row["id"],
                    filename=row["filename"],
                    title=row["title"],
                    page_count=row["page_count"],
                    metadata=json.loads(row["metadata"])
                )
                for row in cursor.fetchall()
            ]
    
    # =========================================================================
    # Table Operations
    # =========================================================================
    
    def save_table(self, table: ExtractedTable) -> None:
        """
        Save an extracted table as a native SQL table.
        
        Creates a real SQL table with proper columns instead of EAV format.
        This allows direct SQL queries like: SELECT * FROM table_name WHERE col = value
        """
        import pandas as pd
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Save table metadata in extracted_tables
            cursor.execute("""
                INSERT OR REPLACE INTO extracted_tables 
                (id, document_id, table_name, page_number, schema_description, columns, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                table.id,
                table.document_id,
                table.table_name,
                table.page_number,
                table.schema_description,
                json.dumps(table.columns),
                table.raw_text
            ))
            
            # Create native SQL table from rows
            if table.rows:
                # Convert rows to DataFrame
                df = pd.DataFrame(table.rows)
                
                # Sanitize column names
                df.columns = [self._sanitize_column_name(c) for c in df.columns]
                
                # Create unique table name (include doc context to avoid collisions)
                native_table_name = self._make_unique_table_name(table.table_name, table.document_id)
                
                # Drop existing table
                conn.execute(f'DROP TABLE IF EXISTS "{native_table_name}"')
                
                # Create table with proper types
                df.to_sql(
                    native_table_name,
                    conn,
                    if_exists='replace',
                    index=False,
                    dtype=self._infer_sql_types(df)
                )
                
                # Register in spreadsheet_tables for unified tracking
                conn.execute("""
                    INSERT OR REPLACE INTO spreadsheet_tables 
                    (table_name, document_id, columns, row_count, source_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    native_table_name,
                    table.document_id,
                    json.dumps(df.columns.tolist()),
                    len(df),
                    'pdf'  # Track source type
                ))
            
            conn.commit()
    
    def _make_unique_table_name(self, table_name: str, doc_id: str) -> str:
        """Create a unique table name with document context."""
        # Get document info for context
        doc = self.get_document(doc_id)
        if doc:
            # Use first 10 chars of sanitized filename
            doc_prefix = re.sub(r'[^a-z0-9]', '', doc.filename.lower().replace('.pdf', '').replace('.xlsx', '').replace('.csv', ''))[:10]
            return f"{doc_prefix}_{table_name}"[:63]  # SQLite max identifier length
        return f"{doc_id[:8]}_{table_name}"[:63]
    
    def _parse_numeric(self, value: Any) -> float | None:
        """Try to parse a value as a number."""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove common formatting
            cleaned = value.replace(",", "").replace("$", "").replace("%", "").strip()
            
            # Handle parentheses as negative (accounting notation)
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = "-" + cleaned[1:-1]
            
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    def save_spreadsheet_native(self, table_name: str, df: pd.DataFrame, doc_id: str) -> None:
        """
        Save a spreadsheet as a native SQL table for fast queries.
        
        This creates a real SQL table with proper column types instead of
        using the normalized EAV format. Much faster for large datasets.
        
        Args:
            table_name: Sanitized table name (SQL-safe)
            df: Pandas DataFrame with the data
            doc_id: Document ID for tracking
        """
        with self._get_connection() as conn:
            # Drop existing table if it exists
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            
            # Sanitize column names for SQL
            df_clean = df.copy()
            df_clean.columns = [self._sanitize_column_name(c) for c in df.columns]
            
            # Create table and insert data using pandas
            df_clean.to_sql(
                table_name, 
                conn, 
                if_exists='replace', 
                index=False,
                dtype=self._infer_sql_types(df_clean)
            )
            
            # Create indexes on common query columns
            self._create_indexes_for_table(conn, table_name, df_clean.columns.tolist())
            
            # Register in metadata table
            conn.execute("""
                INSERT OR REPLACE INTO spreadsheet_tables 
                (table_name, document_id, columns, row_count, source_type)
                VALUES (?, ?, ?, ?, ?)
            """, (
                table_name,
                doc_id,
                json.dumps(df_clean.columns.tolist()),
                len(df),
                'spreadsheet'
            ))
            
            conn.commit()
    
    def _sanitize_column_name(self, name: str) -> str:
        """Sanitize column name for SQL compatibility."""
        # Replace spaces and special chars with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
        # Ensure starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        return sanitized.lower()
    
    def _infer_sql_types(self, df: pd.DataFrame) -> dict:
        """Infer SQL types from DataFrame dtypes."""
        type_map = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                type_map[col] = 'INTEGER'
            elif pd.api.types.is_float_dtype(dtype):
                type_map[col] = 'REAL'
            else:
                type_map[col] = 'TEXT'
        return type_map
    
    def _create_indexes_for_table(self, conn, table_name: str, columns: list[str]) -> None:
        """
        Create indexes on commonly-queried columns.
        
        Indexes are created for:
        - Date/time columns
        - Company/symbol columns  
        - Category/type columns
        """
        index_keywords = ['date', 'time', 'company', 'symbol', 'ticker', 'type', 'category', 'name', 'id']
        
        for col in columns:
            col_lower = col.lower()
            for keyword in index_keywords:
                if keyword in col_lower:
                    try:
                        index_name = f"idx_{table_name}_{col}"
                        conn.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table_name}"("{col}")')
                    except Exception:
                        pass  # Skip if index creation fails
                    break
    
    def list_spreadsheet_tables(self) -> list[dict]:
        """List all native spreadsheet tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT table_name, document_id, columns, row_count FROM spreadsheet_tables")
            return [
                {
                    'table_name': row['table_name'],
                    'document_id': row['document_id'],
                    'columns': json.loads(row['columns']),
                    'row_count': row['row_count']
                }
                for row in cursor.fetchall()
            ]
    
    def get_table(self, table_id: str) -> ExtractedTable | None:
        """Get a table by ID with all its data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get table metadata
            cursor.execute("SELECT * FROM extracted_tables WHERE id = ?", (table_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Get table data
            cursor.execute("""
                SELECT row_index, column_name, value, numeric_value
                FROM table_data
                WHERE table_id = ?
                ORDER BY row_index, column_name
            """, (table_id,))
            
            # Reconstruct rows
            rows: dict[int, dict[str, Any]] = {}
            for data_row in cursor.fetchall():
                row_idx = data_row["row_index"]
                if row_idx not in rows:
                    rows[row_idx] = {}
                
                col_name = data_row["column_name"]
                # Use numeric value if available, otherwise string value
                value = data_row["numeric_value"] if data_row["numeric_value"] is not None else data_row["value"]
                rows[row_idx][col_name] = value
            
            return ExtractedTable(
                id=row["id"],
                document_id=row["document_id"],
                table_name=row["table_name"],
                page_number=row["page_number"],
                schema_description=row["schema_description"],
                columns=json.loads(row["columns"]),
                rows=[rows[i] for i in sorted(rows.keys())],
                raw_text=row["raw_text"]
            )
    
    def list_tables(self, document_id: str | None = None) -> list[TableSchema]:
        """List all tables, optionally filtered by document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if document_id:
                cursor.execute("""
                    SELECT id, table_name, schema_description, columns, document_id, page_number
                    FROM extracted_tables
                    WHERE document_id = ?
                """, (document_id,))
            else:
                cursor.execute("""
                    SELECT id, table_name, schema_description, columns, document_id, page_number
                    FROM extracted_tables
                """)
            
            return [
                TableSchema(
                    table_name=row["table_name"],
                    description=row["schema_description"] or "",
                    source_document_id=row["document_id"],
                    page_number=row["page_number"]
                )
                for row in cursor.fetchall()
            ]
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        """
        Execute a validated SQL query and return results.
        
        Security measures:
        - Only SELECT queries allowed
        - Forbidden keywords blocked (DROP, DELETE, etc.)
        - Multiple statements blocked
        - Automatic LIMIT clause added if missing
        
        Args:
            sql: SQL query (must be a SELECT statement)
            
        Returns:
            List of result rows as dictionaries
            
        Raises:
            SecurityError: If the query fails validation
        """
        # Validate the SQL before execution
        is_valid, error_msg = validate_sql_query(sql)
        if not is_valid:
            logger.warning(f"SQL validation failed: {error_msg}. Query: {sql[:100]}...")
            raise SecurityError(f"Query validation failed: {error_msg}")
        
        # Add LIMIT clause to prevent unbounded results
        safe_sql = add_limit_clause(sql)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(safe_sql)
            except sqlite3.Error as e:
                # Log the error but don't expose SQL details to caller
                logger.error(f"SQL execution error: {e}")
                raise SecurityError("Query execution failed")
            
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            return [
                dict(zip(columns, row))
                for row in cursor.fetchall()
            ]
    
    def get_schema_for_llm(self) -> str:
        """
        Get a text representation of the database schema for the LLM.
        
        All tables are now native SQL tables that can be queried directly.
        """
        schema_parts = []
        
        # Add schema explanation for native tables
        schema_parts.append("""
-- ═══════════════════════════════════════════════════════════════════════════
-- DATABASE SCHEMA - NATIVE SQL TABLES
-- ═══════════════════════════════════════════════════════════════════════════
--
-- All tables are standard SQL tables with proper columns.
-- Query them directly: SELECT column FROM table_name WHERE condition
--
-- Table names are prefixed with document source for clarity.
-- Example: berkshire_annual_performance, jpmorgan_balance_sheet
--
-- ═══════════════════════════════════════════════════════════════════════════
""")
        
        # Get all native tables from spreadsheet_tables tracking
        native_tables = self.list_spreadsheet_tables()
        
        if not native_tables:
            schema_parts.append("-- No tables available.")
            return "\n".join(schema_parts)
        
        # Group tables by document
        docs = self.list_documents()
        doc_map = {d.id: d for d in docs}
        
        tables_by_doc: dict[str, list] = {}
        for table in native_tables:
            doc_id = table.get('document_id', 'unknown')
            if doc_id not in tables_by_doc:
                tables_by_doc[doc_id] = []
            tables_by_doc[doc_id].append(table)
        
        for doc_id, doc_tables in tables_by_doc.items():
            # Get document name
            doc = doc_map.get(doc_id)
            if doc:
                doc_name = doc.filename.replace('.pdf', '').replace('_', ' ')
            else:
                doc_name = doc_id
            
            # Add document header
            schema_parts.append(f"\n-- ───────────────────────────────────────────────────────────────────────────")
            schema_parts.append(f"-- 📄 DOCUMENT: {doc_name}")
            schema_parts.append(f"--    Tables: {len(doc_tables)}")
            schema_parts.append(f"-- ───────────────────────────────────────────────────────────────────────────")
            
            for table in doc_tables:
                table_name = table['table_name']
                columns = table.get('columns', [])
                row_count = table.get('row_count', 0)
                
                schema_parts.append(f"--")
                schema_parts.append(f"-- TABLE: {table_name} ({row_count} rows)")
                if columns:
                    schema_parts.append(f"--   Columns: {', '.join(columns[:10])}")
                    if len(columns) > 10:
                        schema_parts.append(f"--   ... and {len(columns) - 10} more columns")
        
        return "\n".join(schema_parts)


