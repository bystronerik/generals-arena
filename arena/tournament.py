"""N seeds × bot pairs; store each game, then update ratings."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.ratings import rate_stored_game
from arena.run_match import run_and_store
from arena.store import GAMES_DIR, GameRecord, bot_id_from_run_sh


def parse_seeds(spec: str) -> list[int]:
    """Parse `0,1,2` or `0-3` or `0-3,10` into a sorted unique seed list."""
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"invalid seed range: {part!r}")
            seeds.update(range(lo, hi + 1))
        else:
            seeds.add(int(part))
    if not seeds:
        raise ValueError("no seeds parsed")
    return sorted(seeds)


def bot_pairs(run_scripts: list[Path], *, include_self: bool = False) -> list[tuple[Path, Path]]:
    """Unordered unique pairs (A,B) with A before B in the input list order."""
    scripts = [p.resolve() for p in run_scripts]
    if len(scripts) < 2 and not include_self:
        raise ValueError("need at least two bot run.sh paths")
    pairs: list[tuple[Path, Path]] = []
    if include_self:
        for a, b in itertools.product(scripts, repeat=2):
            pairs.append((a, b))
        return pairs
    for i, a in enumerate(scripts):
        for b in scripts[i + 1 :]:
            pairs.append((a, b))
    return pairs


def run_tournament(
    run_scripts: list[Path],
    seeds: list[int],
    *,
    games_dir: Path | None = None,
    update_ratings: bool = True,
    timeout: float | None = None,
    include_self: bool = False,
    swap_sides: bool = False,
) -> list[GameRecord]:
    """
    Run every seed × pair under competition mode.

    Store each game under data/games/, then rate when update_ratings is True.
    """
    pairs = bot_pairs(run_scripts, include_self=include_self)
    if swap_sides:
        mirrored = [(b, a) for a, b in pairs if a != b]
        pairs = pairs + mirrored

    records: list[GameRecord] = []
    total = len(seeds) * len(pairs)
    n = 0
    for seed in seeds:
        for a, b in pairs:
            n += 1
            a_id = bot_id_from_run_sh(a)
            b_id = bot_id_from_run_sh(b)
            print(f"[tournament] ({n}/{total}) {a_id} vs {b_id} seed={seed}")
            # Always store first; rate after store when requested.
            record = run_and_store(
                a,
                b,
                seed=seed,
                mode="competition",
                games_dir=games_dir or GAMES_DIR,
                timeout=timeout,
                update_ratings=False,
            )
            records.append(record)
            if update_ratings:
                rate_stored_game(record)
                print(f"[tournament] rated game_id={record.game_id}")
    print(f"[tournament] finished {len(records)} game(s)")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run N seeds × bot pairs under --mode competition; store then rate."
    )
    parser.add_argument(
        "bots",
        nargs="+",
        type=Path,
        help="two or more bot run.sh paths",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma list and/or ranges, e.g. 0-3,10 (default: 0)",
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=GAMES_DIR,
        help=f"game JSON directory (default: {GAMES_DIR})",
    )
    parser.add_argument(
        "--no-ratings",
        action="store_true",
        help="store games only; do not update Elo",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="optional per-match subprocess timeout in seconds",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="also play each bot against itself (and all ordered pairs)",
    )
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help="also play each pair with sides swapped",
    )
    args = parser.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    run_tournament(
        args.bots,
        seeds,
        games_dir=args.games_dir,
        update_ratings=not args.no_ratings,
        timeout=args.timeout,
        include_self=args.include_self,
        swap_sides=args.swap_sides,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
