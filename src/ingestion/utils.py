"""Shared utilities for ingestion module."""

import hashlib
import re
from typing import Any


# Constants for header detection thresholds
HEADER_ROW_MIN_FILL_RATIO = 0.5  # At least 50% of cells must be non-empty
HEADER_ROW_TEXT_RATIO = 0.5  # At least 50% of non-empty cells should be text (not numbers)


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a unique chunk ID from document ID and index."""
    content = f"{document_id}:{chunk_index}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def generate_table_id(document_id: str, page_number: int, columns: list[str]) -> str:
    """Generate a unique table ID."""
    content = f"{document_id}:{page_number}:{':'.join(columns)}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def normalize_column_name(name: str, max_length: int = 50) -> str:
    """
    Normalize a column name to a valid SQL-safe identifier.
    
    Args:
        name: Original column name
        max_length: Maximum length for the normalized name
        
    Returns:
        Normalized column name (lowercase, alphanumeric with underscores)
    """
    if not name:
        return ""
    
    # Convert to lowercase, replace non-alphanumeric with underscore
    normalized = re.sub(r'[^a-z0-9]+', '_', name.lower().strip())
    normalized = normalized.strip('_')
    
    # Limit length
    if len(normalized) > max_length:
        normalized = normalized[:max_length]
    
    return normalized or ""


def parse_numeric(value: str) -> float | None:
    """
    Try to parse a string as a number, handling common financial formats.
    
    Handles:
    - Comma separators (1,234.56)
    - Currency symbols ($, €)
    - Percentage signs (%)
    - Accounting notation ((1234) for negative)
    - K/M/B suffixes
    - Percentage points suffix (pts)
    
    Args:
        value: String value to parse
        
    Returns:
        Float value or None if not parseable
    """
    if not value:
        return None
    
    # Remove common formatting
    cleaned = value.replace(",", "").replace("$", "").replace("%", "").replace("€", "").strip()
    
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


def is_numeric(value: str) -> bool:
    """Check if a value appears to be numeric."""
    return parse_numeric(value) is not None


def table_to_text(columns: list[str], rows: list[dict], max_rows: int | None = None) -> str:
    """
    Convert a table (columns + rows) to readable text format.
    
    Args:
        columns: List of column names
        rows: List of row dictionaries
        max_rows: Maximum rows to include (None for all)
        
    Returns:
        Text representation of the table
    """
    lines = []
    
    # Header
    lines.append(" | ".join(columns))
    lines.append("-" * len(lines[0]))
    
    # Rows
    display_rows = rows if max_rows is None else rows[:max_rows]
    for row in display_rows:
        row_values = [str(row.get(col, "") or "") for col in columns]
        lines.append(" | ".join(row_values))
    
    if max_rows and len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    
    return "\n".join(lines)


def sanitize_table_name(name: str, max_length: int = 50) -> str:
    """
    Sanitize a name for use as SQL table name.
    
    Args:
        name: Original name
        max_length: Maximum length for the sanitized name
        
    Returns:
        Sanitized table name
    """
    # Replace non-alphanumeric with underscore, lowercase
    sanitized = re.sub(r'[^a-zA-Z0-9]+', '_', name.lower())
    sanitized = sanitized.strip('_')
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = f"table_{sanitized}"
    
    return sanitized[:max_length] if sanitized else "unnamed_table"
