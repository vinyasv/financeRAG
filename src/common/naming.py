"""Canonical naming helpers."""

from __future__ import annotations

import re


def normalize_column_name(name: str, max_length: int = 50) -> str:
    """Normalize a column name to a SQL-safe identifier."""
    if not name:
        return ""

    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower().strip())
    normalized = normalized.strip("_")

    if len(normalized) > max_length:
        normalized = normalized[:max_length]

    return normalized or ""


def sanitize_table_name(name: str, max_length: int = 50, numeric_prefix: str = "table_") -> str:
    """Sanitize a name for use as a SQL table name."""
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
    sanitized = sanitized.strip("_")

    if sanitized and sanitized[0].isdigit():
        sanitized = f"{numeric_prefix}{sanitized}"

    return sanitized[:max_length] if sanitized else "unnamed_table"
