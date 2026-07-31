"""N games × bot pairs under competition mode; parallel worker pool."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any

from arena.records.ratings import rebuild_from_games
from arena.records.registry import Registry
from arena.records.store import (
    GAMES_DIR,
    GameRecord,
    bot_id_from_run_sh,
    engine_version,
    round_games_dir,
    utc_now_iso,
)
from arena.tournaments.parallel import cap_jobs, default_jobs, run_pool
from arena.tournaments.worker import run_one_worker

DEFAULT_GAMES_PER_PAIR = 50


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


def bot_pairs(
    run_scripts: list[Path], *, include_self: bool = False
) -> list[tuple[Path, Path]]:
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


def _pair_rng(round_seed: int, bot_a: str, bot_b: str) -> random.Random:
    """Deterministic RNG stream for one pair, derived from the round seed."""
    material = f"{round_seed}:{bot_a}\0{bot_b}".encode()
    # Prefer hashlib for stable cross-platform mix; avoid Python hash salt.
    import hashlib

    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def expand_pair_seeds(
    pairs: list[tuple[Path, Path]],
    *,
    games_per_pair: int,
    round_seed: int,
    fixed_seeds: list[int] | None = None,
) -> list[tuple[Path, Path, int]]:
    """
    Expand pairs into (bot_a, bot_b, map_seed) match specs.

    When `fixed_seeds` is set, every pair plays that seed list.
    Otherwise each pair draws `games_per_pair` distinct random map seeds.
    """
    if games_per_pair < 1:
        raise ValueError(f"games_per_pair must be >= 1 (got {games_per_pair})")

    specs: list[tuple[Path, Path, int]] = []
    if fixed_seeds is not None:
        for a, b in pairs:
            for seed in fixed_seeds:
                specs.append((a, b, seed))
        return specs

    for a, b in pairs:
        a_id = bot_id_from_run_sh(a)
        b_id = bot_id_from_run_sh(b)
        rng = _pair_rng(round_seed, a_id, b_id)
        chosen: set[int] = set()
        while len(chosen) < games_per_pair:
            chosen.add(rng.randrange(0, 2**31 - 1))
        for seed in sorted(chosen):
            specs.append((a, b, seed))
    return specs


def write_round_manifest(
    games_dir: Path,
    *,
    round_name: str,
    round_seed: int,
    games_per_pair: int,
    jobs: int,
    bots: list[str],
    specs: list[tuple[Path, Path, int]],
    swap_sides: bool,
    fixed_seeds: list[int] | None,
    content_hashes: dict[str, str] | None = None,
) -> Path:
    """Write `manifest.json` describing the round grid."""
    assignments: list[dict[str, Any]] = []
    for a, b, seed in specs:
        assignments.append(
            {
                "bot_a": bot_id_from_run_sh(a),
                "bot_b": bot_id_from_run_sh(b),
                "seed": seed,
            }
        )
    payload = {
        "round": round_name,
        "round_seed": round_seed,
        "games_per_pair": games_per_pair,
        "jobs": jobs,
        "bots": bots,
        "bot_content_hashes": content_hashes or {},
        "swap_sides": swap_sides,
        "fixed_seeds": fixed_seeds,
        "match_count": len(assignments),
        "assignments": assignments,
        "written_at": utc_now_iso(),
    }
    games_dir.mkdir(parents=True, exist_ok=True)
    path = games_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_tournament(
    run_scripts: list[Path],
    *,
    round_name: str,
    games_per_pair: int = DEFAULT_GAMES_PER_PAIR,
    round_seed: int = 0,
    fixed_seeds: list[int] | None = None,
    games_dir: Path | None = None,
    update_ratings: bool = True,
    timeout: float | None = None,
    include_self: bool = False,
    swap_sides: bool = False,
    strict_versions: bool = False,
    jobs: int | None = None,
) -> list[GameRecord]:
    """
    Run games_per_pair (or fixed_seeds) × pairs under competition mode.

    Stores each game under data/games/<round>/, then rebuilds Elo once when
    update_ratings is True.
    """
    pairs = bot_pairs(run_scripts, include_self=include_self)
    if swap_sides:
        mirrored = [(b, a) for a, b in pairs if a != b]
        pairs = pairs + mirrored

    specs = expand_pair_seeds(
        pairs,
        games_per_pair=games_per_pair,
        round_seed=round_seed,
        fixed_seeds=fixed_seeds,
    )
    directory = games_dir or round_games_dir(round_name)
    worker_jobs = cap_jobs(default_jobs() if jobs is None else jobs)
    engine = engine_version()
    bot_ids = [bot_id_from_run_sh(p) for p in run_scripts]
    # Hash and register the roster once here, in the parent, rather than per
    # match in every worker: one hash for the whole round cannot go stale
    # mid-round the way a long-lived worker's own hash could, and one writer
    # means no concurrent writes to the registry.
    content_hashes = Registry().register_run_scripts(
        run_scripts, strict=strict_versions
    )

    manifest_path = write_round_manifest(
        directory,
        round_name=round_name,
        round_seed=round_seed,
        games_per_pair=games_per_pair if fixed_seeds is None else len(fixed_seeds),
        jobs=worker_jobs,
        bots=bot_ids,
        specs=specs,
        swap_sides=swap_sides,
        fixed_seeds=fixed_seeds,
        content_hashes=content_hashes,
    )
    print(f"[tournament] wrote {manifest_path}")
    print(
        f"[tournament] round={round_name} matches={len(specs)} "
        f"jobs={worker_jobs} games_dir={directory}"
    )

    payloads = [
        {
            "bot_a_run": str(a.resolve()),
            "bot_b_run": str(b.resolve()),
            "seed": seed,
            "games_dir": str(directory),
            "mode": "competition",
            "round": round_name,
            "timeout": timeout,
            "engine_version": engine,
            "bot_a_content_hash": content_hashes[bot_id_from_run_sh(a)],
            "bot_b_content_hash": content_hashes[bot_id_from_run_sh(b)],
        }
        for a, b, seed in specs
    ]

    def _on_result(done: int, total: int, record: GameRecord) -> None:
        print(
            f"[tournament] ({done}/{total}) {record.bot_a} vs {record.bot_b} "
            f"seed={record.seed} -> {record.winner} turns={record.turns} "
            f"game_id={record.game_id}"
        )

    records = run_pool(
        payloads,
        run_one_worker,
        jobs=worker_jobs,
        on_result=_on_result,
    )
    records.sort(key=lambda r: (r.seed, r.bot_a, r.bot_b, r.game_id))
    print(f"[tournament] finished {len(records)} game(s)")

    if update_ratings:
        book = rebuild_from_games(games_dir=GAMES_DIR)
        print(
            f"[tournament] rebuilt ratings from {GAMES_DIR} "
            f"({len(book.rated_game_ids)} rated game(s))"
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run games-per-pair × bot pairs under --mode competition "
            "(parallel workers); store under data/games/<round>/, then rate."
        )
    )
    parser.add_argument(
        "bots",
        nargs="+",
        type=Path,
        help="two or more bot run.sh paths",
    )
    parser.add_argument(
        "--round",
        required=True,
        help="round name; games store under data/games/<round>/",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=DEFAULT_GAMES_PER_PAIR,
        help=f"random map seeds per unordered pair (default: {DEFAULT_GAMES_PER_PAIR})",
    )
    parser.add_argument(
        "--round-seed",
        type=int,
        default=0,
        help="RNG seed for map-seed generation (default: 0)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="optional fixed seed list/ranges (overrides --games-per-pair RNG)",
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=None,
        help="override output directory (default: data/games/<round>/)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="parallel workers (default: physical CPU cores; capped to that)",
    )
    parser.add_argument(
        "--no-ratings",
        action="store_true",
        help="store games only; do not rebuild Elo",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="optional per-match wall-clock timeout in seconds",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="also play each bot against itself (and all ordered pairs)",
    )
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help="also play each pair with sides swapped (off by default)",
    )
    parser.add_argument(
        "--strict-versions",
        action="store_true",
        help="refuse to run if any bot's closure differs from HEAD (published rounds)",
    )
    args = parser.parse_args(argv)

    if args.games_per_pair < 1:
        parser.error("--games-per-pair must be >= 1")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be >= 1")

    fixed = parse_seeds(args.seeds) if args.seeds else None
    games_dir = args.games_dir or round_games_dir(args.round)
    run_tournament(
        args.bots,
        round_name=args.round,
        games_per_pair=args.games_per_pair,
        round_seed=args.round_seed,
        fixed_seeds=fixed,
        games_dir=games_dir,
        update_ratings=not args.no_ratings,
        timeout=args.timeout,
        include_self=args.include_self,
        swap_sides=args.swap_sides,
        strict_versions=args.strict_versions,
        jobs=args.jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
