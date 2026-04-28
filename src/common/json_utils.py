"""Helpers for parsing LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def strip_markdown_fence(text: str) -> str:
    """Return text without a surrounding markdown code fence."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_response(text: str) -> Any:
    """Parse JSON from a plain or fenced LLM response."""
    cleaned = strip_markdown_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
