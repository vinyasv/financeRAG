"""Shared validation and policy helpers."""

from __future__ import annotations

import re
from pathlib import Path

MAX_QUERY_LENGTH = 5000
MAX_PROMPT_CONTENT_LENGTH = 50000
MAX_FILE_SIZE_MB = 500
SUPPORTED_INGEST_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv"}
VALID_DOCUMENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"forget\s+(everything|all|what)",
    r"you\s+are\s+now\s+(?:a|an)",
    r"your\s+new\s+(?:role|purpose|instructions?)\s+(?:is|are)",
    r"respond\s+as\s+(?:if|though)",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"act\s+as\s+(?:if|though|a|an)",
    r"override\s+(?:your|the)",
    r"bypass\s+(?:your|the|all)",
    r"reveal\s+(?:your|the)",
    r"show\s+(?:me\s+)?(?:your|the)\s+(?:system|prompt|instructions?)",
    r"what\s+(?:is|are)\s+your\s+(?:system|instructions?|prompt)",
    r"output\s+(?:your|the)\s+(?:system|prompt|instructions?)",
    r"print\s+(?:your|the)\s+(?:system|prompt|instructions?)",
]
_COMPILED_INJECTION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]


def detect_injection_attempt(text: str) -> tuple[bool, list[str]]:
    """Detect likely prompt-injection attempts in user input."""
    text_lower = text.lower()
    matches = [
        INJECTION_PATTERNS[index]
        for index, pattern in enumerate(_COMPILED_INJECTION_PATTERNS)
        if pattern.search(text_lower)
    ]
    return bool(matches), matches


def validate_query(query: str) -> tuple[bool, str, list[str]]:
    """Validate a user query and return matched suspicious patterns."""
    if not query or not query.strip():
        return False, "Query cannot be empty", []

    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query too long ({len(query)} chars). Maximum: {MAX_QUERY_LENGTH}", []

    is_suspicious, patterns = detect_injection_attempt(query)
    return True, "", patterns if is_suspicious else []


def validate_file_size(size_bytes: int, max_mb: int = MAX_FILE_SIZE_MB) -> tuple[bool, str]:
    """Validate a file size against the configured maximum."""
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        return False, f"File size ({size_bytes / 1024 / 1024:.1f}MB) exceeds limit ({max_mb}MB)"
    return True, ""


def sanitize_filename(name: str) -> str:
    """Sanitize an untrusted filename into a safe local filename."""
    safe_name = Path(name).name
    if ".." in safe_name:
        safe_name = safe_name.replace("..", "_")
    return re.sub(r"[^\w\-. ]", "_", safe_name)


def validate_safe_filename(path_str: str, allowed_extensions: set[str] | None = None) -> tuple[bool, str]:
    """Validate a path or filename for traversal and extension safety."""
    if not path_str:
        return False, "Empty path not allowed"
    if ".." in path_str:
        return False, "Path traversal detected (..)"
    if "\x00" in path_str:
        return False, "Null byte in path"
    if allowed_extensions:
        ext = Path(path_str).suffix.lower()
        if ext not in allowed_extensions:
            return False, f"Extension {ext} not allowed"
    return True, ""


def validate_document_id(doc_id: str) -> bool:
    """Validate that a document ID is safe to use in filenames and queries."""
    return bool(doc_id and VALID_DOCUMENT_ID_PATTERN.match(doc_id))


def validate_ingestion_file(
    file_path: Path,
    allowed_extensions: set[str] | None = None,
    max_file_size_mb: int = MAX_FILE_SIZE_MB,
) -> tuple[bool, str]:
    """Validate a candidate input file for ingestion."""
    allowed = allowed_extensions or SUPPORTED_INGEST_EXTENSIONS

    if file_path.suffix.lower() not in allowed:
        return False, f"Unsupported file type: {file_path.suffix}"

    is_safe, error = validate_safe_filename(str(file_path.name), allowed)
    if not is_safe:
        return False, f"Path security issue: {error}"

    try:
        size_bytes = file_path.stat().st_size
    except OSError as exc:
        return False, f"Cannot read file: {exc}"

    return validate_file_size(size_bytes, max_file_size_mb)
