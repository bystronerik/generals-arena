#!/usr/bin/env python3
"""Thin CLI: rebuild Elo leaderboard from data/games/."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.ratings import main

if __name__ == "__main__":
    # Default to printing the table for a quick glance.
    argv = sys.argv[1:]
    if "--print" not in argv and "-h" not in argv and "--help" not in argv:
        argv = [*argv, "--print"]
    raise SystemExit(main(argv))
