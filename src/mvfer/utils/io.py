"""Small path and JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, value) -> None:
    ensure_parent(path).write_text(json.dumps(value, indent=2) + "\n")
