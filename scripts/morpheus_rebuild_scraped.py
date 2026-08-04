#!/usr/bin/env python3
"""Rebuild scraped leaderboard replays into arena trajectories.

Usage:
    python scripts/morpheus_rebuild_scraped.py \\
      --player ResBot \\
      --replays competition-replays/ResBot \\
      --output data/trajectories/resbot-reconstructions \\
      --report docs/research/measurements/morpheus-resbot-rebuild.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", required=True, help="leaderboard player folder name")
    parser.add_argument(
        "--replays",
        type=Path,
        default=None,
        help="player replay directory (default: competition-replays/<player>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="trajectory round directory, e.g. data/trajectories/resbot-reconstructions",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path",
    )
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing trajectory files",
    )
    args = parser.parse_args(argv)

    from training.morpheus.scraped_rebuild.rebuild import rebuild_player

    report = rebuild_player(
        player=args.player,
        replays=args.replays,
        output=args.output,
        report_path=args.report,
        max_games=args.max_games,
        force=args.force,
        repo_root=REPO,
    )
    print(json.dumps(report.to_dict(), indent=2))
    # Non-zero only when every scanned game failed and at least one was scanned.
    if report.scanned > 0 and report.kept == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
