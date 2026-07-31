#!/usr/bin/env python3
"""Thin CLI: refit arena ratings from data/games/ and print the leaderboard."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.records.ratings.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
