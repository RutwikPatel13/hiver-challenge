"""Dataset loading and email rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


_cache: list[dict] | None = None


def load_dataset() -> list[dict]:
    global _cache
    if _cache is None:
        if not config.DATASET_PATH.exists():
            raise FileNotFoundError(
                f"{config.DATASET_PATH} not found. Run: python src/build_dataset.py"
            )
        _cache = [
            json.loads(line)
            for line in config.DATASET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return _cache


def split(name: str) -> list[dict]:
    return [r for r in load_dataset() if r["split"] == name]


def render_email(record: dict) -> str:
    """Render an incoming email as the generator/judge see it."""
    return f"Subject: {record['subject']}\n\n{record['customer_message']}"
