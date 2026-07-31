#!/usr/bin/env python3
"""
Aggregate remote game JSON and emit a human-block markdown report.

Counts only games where opponent_is_bot is False and counts_toward_block is true.
See docs/engine/remote-play-setup.md and docs/research/strategies/human-95-plan.md §3.5.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arena.remote_env import REMOTE_GAMES_DIR, REPO_ROOT, ensure_repo_on_path, load_dotenv_files
from arena.remote_report import aggregate_remote_games, format_markdown_report, load_remote_records

ensure_repo_on_path()


def main(argv: list[str] | None = None) -> int:
    load_dotenv_files()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REMOTE_GAMES_DIR,
        help=f"Directory of remote game JSON (default: {REMOTE_GAMES_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write markdown report to this path (default: stdout only)",
    )
    args = parser.parse_args(argv)

    log_dir = args.log_dir.resolve()
    records = load_remote_records(log_dir)
    stats = aggregate_remote_games(records)
    report = format_markdown_report(stats, log_dir=log_dir)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
