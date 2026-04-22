"""Shared utilities for ingestion module."""

from ..common.ids import chunk_id as generate_chunk_id
from ..common.ids import table_id as generate_table_id
from ..common.naming import normalize_column_name, sanitize_table_name
from ..common.text_rendering import table_to_text

__all__ = [
    "HEADER_ROW_MIN_FILL_RATIO",
    "HEADER_ROW_TEXT_RATIO",
    "generate_chunk_id",
    "generate_table_id",
    "is_numeric",
    "normalize_column_name",
    "parse_numeric",
    "sanitize_table_name",
    "table_to_text",
]

# Constants for header detection thresholds
HEADER_ROW_MIN_FILL_RATIO = 0.5  # At least 50% of cells must be non-empty
HEADER_ROW_TEXT_RATIO = 0.5  # At least 50% of non-empty cells should be text (not numbers)
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
