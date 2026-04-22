"""Local script bootstrap helpers."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_root() -> Path:
    """Ensure the project root is importable when running scripts directly."""
    project_root = Path(__file__).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root


PROJECT_ROOT = bootstrap_project_root()
