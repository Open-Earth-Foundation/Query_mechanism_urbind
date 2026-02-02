from __future__ import annotations

from pathlib import Path


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


__all__ = ["load_prompt"]
