"""Human block-game counting for remote 95/100 evaluation."""

from __future__ import annotations

import json
from pathlib import Path


def counts_as_human_block_game(record: dict) -> bool:
    """Return True when a remote log row counts toward the human block."""
    if not record.get("counts_toward_block"):
        return False
    # Require explicit human opponent; null/True bot labels are excluded.
    return record.get("opponent_is_bot") is False


def count_human_block_games(
    log_dir: Path,
    *,
    since_mtime: float | None = None,
) -> int:
    """Count block-eligible human games under log_dir (newest mtime filter optional)."""
    if not log_dir.is_dir():
        return 0
    total = 0
    for path in log_dir.glob("*.json"):
        if path.name.startswith("session_error_"):
            continue
        try:
            if since_mtime is not None and path.stat().st_mtime < since_mtime:
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if counts_as_human_block_game(record):
            total += 1
    return total
