"""Shared text rendering helpers."""

from __future__ import annotations


def table_to_text(columns: list[str], rows: list[dict], max_rows: int | None = None) -> str:
    """Convert a table into a readable text representation."""
    lines = [" | ".join(columns)]
    lines.append("-" * len(lines[0]))

    display_rows = rows if max_rows is None else rows[:max_rows]
    for row in display_rows:
        row_values = [str(row.get(col, "") or "") for col in columns]
        lines.append(" | ".join(row_values))

    if max_rows and len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")

    return "\n".join(lines)
