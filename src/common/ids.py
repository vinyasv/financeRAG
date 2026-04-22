"""Canonical ID generation helpers."""

from __future__ import annotations

import hashlib


def _hash_short(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def document_id_from_filename(filename: str) -> str:
    """Generate a stable document ID from a filename."""
    return _hash_short(filename)


def sheet_document_id(filename: str, sheet_name: str) -> str:
    """Generate a stable document ID for a spreadsheet sheet."""
    return _hash_short(f"{filename}:{sheet_name}")


def chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a stable chunk ID."""
    return _hash_short(f"{document_id}:{chunk_index}")


def table_id(document_id: str, page_number: int, columns: list[str]) -> str:
    """Generate a stable table ID."""
    return _hash_short(f"{document_id}:{page_number}:{':'.join(columns)}")
