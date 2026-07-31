"""N games × bot pairs under competition mode; parallel worker pool."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from arena.records.registry import Registry
from arena.records.store import (
    GAMES_DIR,
    GameRecord,
    bot_id_from_run_sh,
    engine_version,
    round_games_dir,
    utc_now_iso,
)
from arena.records.trajectories import round_trajectory_dir
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
    """
    Unordered unique pairs (A,B) with A before B in the input list order.

    Order within a pair carries no meaning any more: seat is drawn per game
    (`expand_pair_seeds`), so emitting both orderings would only duplicate
    work. `include_self` adds each bot against itself — those games say nothing
    about strength but are a clean, strength-free estimator of the seat and
    draw parameters.
    """
    scripts = [p.resolve() for p in run_scripts]
    if len(scripts) < 2 and not include_self:
        raise ValueError("need at least two bot run.sh paths")
    pairs: list[tuple[Path, Path]] = []
    for i, a in enumerate(scripts):
        for b in scripts[i + 1 :]:
            pairs.append((a, b))
    if include_self:
        pairs.extend((a, a) for a in scripts)
    return pairs


# How a pair's games are split between the two seats.
RANDOM_SEATS = "random"  # one game per map seed, orientation drawn from the stream
ALTERNATING_SEATS = "alternate"  # each map seed played both ways: exactly 50/50
SEAT_POLICIES = (RANDOM_SEATS, ALTERNATING_SEATS)


def _pair_rng(round_seed: int, bot_a: str, bot_b: str) -> random.Random:
    """
    Deterministic RNG stream for one pair, derived from the round seed.

    The key is **orientation-independent** (the two ids are sorted). It has to
    be: the stream is what chooses the seat, so keying it on the orientation
    would make the pair's RNG depend on the answer it is being asked for. The
    old oriented key is also why `--swap-sides` never actually mirrored —
    `(a, b)` and `(b, a)` drew entirely different map seeds, so it bought 2×
    the games and an unmatched sample.
    """
    first, second = sorted((bot_a, bot_b))
    material = f"{round_seed}:{first}\0{second}".encode()
    # Prefer hashlib for stable cross-platform mix; avoid Python hash salt.
    import hashlib

    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _draw_seeds(rng: random.Random, count: int) -> list[int]:
    chosen: set[int] = set()
    while len(chosen) < count:
        chosen.add(rng.randrange(0, 2**31 - 1))
    return sorted(chosen)


def expand_pair_seeds(
    pairs: list[tuple[Path, Path]],
    *,
    games_per_pair: int,
    round_seed: int,
    fixed_seeds: list[int] | None = None,
    seat_policy: str = RANDOM_SEATS,
) -> list[tuple[Path, Path, int]]:
    """
    Expand pairs into (seat_a, seat_b, map_seed) match specs.

    Seat assignment is part of the draw, not a property of roster position.
    Under the old scheme "sits in seat A" and "appears earlier in the roster"
    were the same variable, so the fitted seat advantage was aliased with the
    strength parameters and its variance inflated ~2.9×.

    `seat_policy`:

    - `"random"` — one game per map seed, orientation drawn from the pair's
      stream. Balances in expectation and costs **zero extra games**, which is
      what makes it right for large exploratory rounds.
    - `"alternate"` — every map seed is played in **both** orientations, so the
      split is exactly 50/50 by construction and map difficulty cancels within
      each matched pair. Half as many distinct seeds for the same game count;
      an odd `games_per_pair` rounds up to keep the balance exact. This is what
      decision arms use, where balancing only in expectation is not good enough
      (over 1150 games the seat-A count is 575 ± 17).

    Self-play pairs play one game per seed under either policy — both seats
    hold the same program, so mirroring would just buy the same game twice.
    """
    if games_per_pair < 1:
        raise ValueError(f"games_per_pair must be >= 1 (got {games_per_pair})")
    if seat_policy not in SEAT_POLICIES:
        raise ValueError(f"unknown seat policy {seat_policy!r}; use one of {SEAT_POLICIES}")

    specs: list[tuple[Path, Path, int]] = []
    for a, b in pairs:
        a_id = bot_id_from_run_sh(a)
        b_id = bot_id_from_run_sh(b)
        rng = _pair_rng(round_seed, a_id, b_id)
        # Resolve the pair to a canonical orientation so the seat decision does
        # not depend on how the caller happened to order it.
        low, high = (a, b) if a_id <= b_id else (b, a)
        self_play = a_id == b_id

        paired = seat_policy == ALTERNATING_SEATS and not self_play
        wanted = -(-games_per_pair // 2) if paired else games_per_pair
        seeds = list(fixed_seeds) if fixed_seeds is not None else _draw_seeds(rng, wanted)

        for seed in seeds:
            if paired:
                specs.append((low, high, seed))
                specs.append((high, low, seed))
            elif self_play or rng.random() < 0.5:
                specs.append((low, high, seed))
            else:
                specs.append((high, low, seed))
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
    seat_policy: str,
    fixed_seeds: list[int] | None,
    engine: str | None = None,
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
        "engine_version": engine,
        "seat_policy": seat_policy,
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
    seat_policy: str = RANDOM_SEATS,
    strict_versions: bool = False,
    jobs: int | None = None,
    record_trajectories: bool = False,
) -> list[GameRecord]:
    """
    Run games_per_pair (or fixed_seeds) × pairs under competition mode.

    Stores each game under data/games/<round>/, then refits ratings once when
    update_ratings is True.
    """
    pairs = bot_pairs(run_scripts, include_self=include_self)
    specs = expand_pair_seeds(
        pairs,
        games_per_pair=games_per_pair,
        round_seed=round_seed,
        fixed_seeds=fixed_seeds,
        seat_policy=seat_policy,
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
        seat_policy=seat_policy,
        fixed_seeds=fixed_seeds,
        engine=engine,
        content_hashes=content_hashes,
    )
    print(f"[tournament] wrote {manifest_path}")
    print(
        f"[tournament] round={round_name} matches={len(specs)} "
        f"jobs={worker_jobs} games_dir={directory}"
    )

    # The parent creates the round's trajectory directory once; each worker
    # then writes only its own game's files into it.
    trajectories_dir = None
    if record_trajectories:
        trajectories_dir = round_trajectory_dir(round_name)
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        print(f"[tournament] recording trajectories under {trajectories_dir}")

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
            "trajectories_dir": str(trajectories_dir) if trajectories_dir else None,
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
        from arena.records.ratings.cli import MissingAnchor, refit

        try:
            fit = refit(games_dir=GAMES_DIR)
        except MissingAnchor as exc:
            # The games are stored; a refit is a pure function of them and can
            # be re-run at any time. Losing a round's results over a missing
            # anchor would be the worse outcome.
            print(f"[tournament] ratings not refitted: {exc}")
        else:
            print(
                f"[tournament] refitted ratings from {GAMES_DIR} "
                f"({fit.counts.games} rated game(s), {len(fit.entities)} entit(ies))"
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
        help="store games only; do not refit ratings",
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
        "--seat-policy",
        choices=SEAT_POLICIES,
        default=RANDOM_SEATS,
        help=(
            "how a pair's games split between seats: 'random' draws the seat "
            "per game (balanced in expectation, zero extra games); 'alternate' "
            "plays every map seed both ways (exactly 50/50, matched pairs) — "
            f"use it for decision arms (default: {RANDOM_SEATS})"
        ),
    )
    parser.add_argument(
        "--strict-versions",
        action="store_true",
        help="refuse to run if any bot's closure differs from HEAD (published rounds)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "also write per-turn trajectories under data/trajectories/<round>/ "
            "(off by default; see docs/arena/trajectories.md)"
        ),
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
        seat_policy=args.seat_policy,
        strict_versions=args.strict_versions,
        jobs=args.jobs,
        record_trajectories=args.record,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
