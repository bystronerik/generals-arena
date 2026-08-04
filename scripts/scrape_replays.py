#!/usr/bin/env python3
"""Thin CLI: scrape generals.bot leaderboard replays into competition-replays/.

Wraps competition-scraper/scrape.py so the output directory and the default
player are fixed here instead of retyped. Runs are incremental: replays already
on disk are skipped, so re-running periodically is how history accumulates (the
list endpoint returns only a recent window, with no pagination).

Scraped replays are competition-rules games played by real leaderboard
entrants. They are observational data, not arena matches: they never enter
data/games/ or the rating fit. See docs/engine/leaderboard-replays.md.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRAPER = REPO_ROOT / "competition-scraper" / "scrape.py"
REPLAYS_DIR = REPO_ROOT / "competition-replays"
DEFAULT_PLAYERS = ["erik.bystron"]
OUTCOMES = ("win", "lose", "draw")


def load_scraper() -> ModuleType:
    """Import the submodule script by path — it is not a package on sys.path."""
    if not SCRAPER.exists():
        raise SystemExit(
            f"{SCRAPER} is missing — run: git submodule update --init competition-scraper"
        )
    spec = importlib.util.spec_from_file_location("competition_scraper", SCRAPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves the module's postponed annotations through sys.modules,
    # so register before executing.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"{exc} — install the scraper dependency: pip install httpx") from exc
    return module


def replay_files(directory: Path) -> list[Path]:
    return [p for p in directory.glob("*.json") if not p.name.endswith(".meta.json")]


def print_inventory(out: Path, players: list[str]) -> None:
    """Report what is on disk after the run, so the next run's baseline is visible."""
    for player in players:
        player_dir = out / player.replace("/", "_")
        counts = {o: replay_files(player_dir / o) for o in OUTCOMES if (player_dir / o).is_dir()}
        total = sum(len(files) for files in counts.values())
        if not total:
            print(f"{player}: no replays on disk under {player_dir}")
            continue
        megabytes = sum(p.stat().st_size for files in counts.values() for p in files) / 1e6
        breakdown = " / ".join(f"{len(counts[o])} {o}" for o in OUTCOMES if o in counts)
        print(f"{player}: {total} replays on disk ({breakdown}) — {megabytes:.0f} MB in {player_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "players",
        nargs="*",
        help=f"leaderboard player name(s) (default: {' '.join(DEFAULT_PLAYERS)})",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="parallel downloads (default: 8)")
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="max requests per second (default: the scraper's own cap; 0 disables pacing)",
    )
    parser.add_argument("--out", type=Path, default=REPLAYS_DIR, help="output directory")
    args = parser.parse_args(argv)

    players = args.players or DEFAULT_PLAYERS
    scraper = load_scraper()
    forwarded = ["--concurrency", str(args.concurrency), "--out", str(args.out)]
    if args.rate is not None:
        forwarded += ["--rate", str(args.rate)]
    scraper_args = scraper.parse_args([*players, *forwarded])
    try:
        exit_code = asyncio.run(scraper.main_async(scraper_args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    print_inventory(args.out, players)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
