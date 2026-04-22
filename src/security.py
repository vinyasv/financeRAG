"""Security-facing compatibility wrappers."""

from .common.prompts import (
    STANDARD_PROMPT_GUARD,
    prepare_prompt_user_content,
    sanitize_user_input,
    wrap_user_content,
)
from .validation import (
    MAX_PROMPT_CONTENT_LENGTH,
    MAX_QUERY_LENGTH,
    detect_injection_attempt,
    validate_file_size,
    validate_safe_filename,
)


def create_prompt_guard() -> str:
    """Return the standard prompt guard notice."""
    return STANDARD_PROMPT_GUARD


def validate_path_safety(path_str: str, allowed_extensions: set[str] | None = None) -> tuple[bool, str]:
    """Backward-compatible wrapper for filename/path validation."""
    return validate_safe_filename(path_str, allowed_extensions)


__all__ = [
    "MAX_PROMPT_CONTENT_LENGTH",
    "MAX_QUERY_LENGTH",
    "STANDARD_PROMPT_GUARD",
    "create_prompt_guard",
    "detect_injection_attempt",
    "prepare_prompt_user_content",
    "sanitize_user_input",
    "validate_file_size",
    "validate_path_safety",
    "wrap_user_content",
]
