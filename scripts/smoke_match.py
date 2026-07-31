#!/usr/bin/env python3
"""Thin CLI: one stored smoke vs expander match (competition mode)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.matches.run_match import run_and_store

DEFAULT_A = REPO_ROOT / "bots" / "smoke" / "run.sh"
DEFAULT_B = (
    REPO_ROOT
    / "competition-module"
    / "competition"
    / "agents"
    / "expander_python"
    / "run.sh"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bot-a", type=Path, default=DEFAULT_A)
    parser.add_argument("--bot-b", type=Path, default=DEFAULT_B)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument(
        "--update-ratings",
        action="store_true",
        help="refit ratings after the game is stored",
    )
    args = parser.parse_args(argv)
    run_and_store(
        args.bot_a,
        args.bot_b,
        seed=args.seed,
        mode="competition",
        timeout=args.timeout,
        update_ratings=args.update_ratings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
