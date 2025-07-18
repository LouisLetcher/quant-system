"""Common file and directory utilities to eliminate code duplication."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_file(path: str | Path) -> dict[str, Any]:
    """Load JSON file with error handling."""
    path = Path(path)
    with path.open() as f:
        return json.load(f)


def save_json_file(path: str | Path, data: dict[str, Any], indent: int = 2) -> None:
    """Save JSON file with consistent formatting."""
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w") as f:
        json.dump(data, f, indent=indent)


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Get UTC timestamp in specified format."""
    return datetime.now(timezone.utc).strftime(format_str)


def safe_file_exists(path: str | Path) -> bool:
    """Check if file exists safely."""
    return Path(path).exists()
