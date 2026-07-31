#!/usr/bin/env python3
"""Heuristic bot measurement suite via the parallel competition tournament API.

Default grid: unordered pairs among the heuristic roster, games-per-pair random
map seeds (Rule C), stored under data/games/<round>/, Elo rebuilt once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.tournaments.parallel import default_jobs
from arena.records.ratings.policy import Policy
from arena.records.reporting import (
    aggregate_stats,
    bot_run_sh,
    leaderboard_table_lines,
    winner_bot_id,
    winrate_table_lines,
)
from arena.records.store import GameRecord, round_games_dir, utc_now_iso
from arena.tournaments.competition import (
    DEFAULT_GAMES_PER_PAIR,
    RANDOM_SEATS,
    SEAT_POLICIES,
    parse_seeds,
)

MEASUREMENTS_DIR = REPO_ROOT / "docs" / "research" / "measurements"

NEW_BOTS = [
    "fog_scout",
    "army_convey",
    "garrison",
    "late_rush",
    "splitter",
    "choke_control",
    "phase_switch",
    "castle_rush",
    "blitz",
    "boom",
    "metro",
    "aegis",
    "proteus",
]

BASELINE_BOTS = ["smoke", "expand_plus", "castle_builder", "general_hunter"]

BENCHMARK_BOTS = ["cm_random", "cm_expander", "cm_hunter", "cm_harvester"]

BENCHMARK_ANCHOR = "army_convey"

# Default measurement roster: new heuristics + baselines + cm_* benchmarks.
DEFAULT_ROSTER = NEW_BOTS + BASELINE_BOTS + BENCHMARK_BOTS


@dataclass
class MatchSpec:
    bot_a: str
    bot_b: str
    seed: int
    tag: str


@dataclass
class GameEntry:
    bot_a: str
    bot_b: str
    seed: int
    winner: str
    winner_bot: str
    turns: int
    terminated: bool
    truncated: bool
    castles_a: int | None = None
    castles_b: int | None = None
    land_margin_a: int | None = None
    land_margin_b: int | None = None
    game_id: str = ""
    tag: str = ""


def wait_for_bots(
    names: list[str],
    *,
    poll_seconds: float = 20.0,
    timeout_seconds: float = 120.0,
) -> None:
    """Poll until every bot has a run.sh (up to timeout)."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [n for n in names if not bot_run_sh(n).exists()]
        if not missing:
            print(f"[measure] all {len(names)} bot(s) ready")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out after {timeout_seconds:.0f}s waiting for: {', '.join(missing)}"
            )
        print(f"[measure] waiting for: {', '.join(missing)} (poll {poll_seconds:.0f}s)")
        time.sleep(poll_seconds)


def both_seat_orders(specs: list[MatchSpec]) -> list[MatchSpec]:
    """Emit A vs B and B vs A for each spec (legacy seat-swap grid)."""
    out: list[MatchSpec] = []
    seen: set[tuple[str, str, int, str]] = set()
    for spec in specs:
        for bot_a, bot_b in ((spec.bot_a, spec.bot_b), (spec.bot_b, spec.bot_a)):
            key = (bot_a, bot_b, spec.seed, spec.tag)
            if key in seen:
                continue
            seen.add(key)
            out.append(MatchSpec(bot_a, bot_b, spec.seed, spec.tag))
    return out


def build_grid() -> list[MatchSpec]:
    """Legacy tagged grid (optional --legacy-grid)."""
    base: list[MatchSpec] = []

    for bot in NEW_BOTS:
        for seed in (0, 1):
            base.append(MatchSpec(bot, "smoke", seed, "new_vs_smoke"))

    for bot in NEW_BOTS:
        base.append(MatchSpec(bot, "expand_plus", 0, "new_vs_expand_plus"))

    for a, b in itertools.combinations(NEW_BOTS, 2):
        base.append(MatchSpec(a, b, 0, "new_round_robin"))

    economy = ["castle_builder", "castle_rush", "phase_switch"]
    for a, b in itertools.combinations(economy, 2):
        for seed in (0, 1):
            base.append(MatchSpec(a, b, seed, "economy_cluster"))

    for bot in BENCHMARK_BOTS:
        for seed in (0, 1):
            base.append(MatchSpec(bot, "smoke", seed, "benchmark_vs_smoke"))

    for bot in BENCHMARK_BOTS:
        base.append(MatchSpec(bot, BENCHMARK_ANCHOR, 0, "benchmark_vs_army_convey"))

    return both_seat_orders(base)


def game_entry_from_record(record: GameRecord, *, tag: str = "") -> GameEntry:
    metrics = record.metrics or {}
    return GameEntry(
        bot_a=record.bot_a,
        bot_b=record.bot_b,
        seed=record.seed,
        winner=record.winner,
        winner_bot=winner_bot_id(record),
        turns=record.turns,
        terminated=record.terminated,
        truncated=record.truncated,
        castles_a=record.castles_built_a,
        castles_b=record.castles_built_b,
        land_margin_a=metrics.get("land_margin_a"),
        land_margin_b=metrics.get("land_margin_b"),
        game_id=record.game_id,
        tag=tag,
    )


def run_one(
    spec: MatchSpec,
    *,
    update_ratings: bool,
    round_name: str,
    games_dir: Path | None = None,
) -> GameEntry:
    from arena.matches.run_match import run_and_store

    a_path = bot_run_sh(spec.bot_a)
    b_path = bot_run_sh(spec.bot_b)
    record = run_and_store(
        a_path,
        b_path,
        seed=spec.seed,
        mode="competition",
        round_name=round_name,
        games_dir=games_dir,
        update_ratings=update_ratings,
    )
    return game_entry_from_record(record, tag=spec.tag)


def notable_matchups(games: list[GameEntry]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for g in games:
        if g.terminated and g.turns < 400:
            notes.append(
                {
                    "kind": "fast_win",
                    "matchup": f"{g.bot_a} vs {g.bot_b}",
                    "seed": g.seed,
                    "winner": g.winner_bot,
                    "turns": g.turns,
                }
            )
        if g.castles_a is not None and (g.castles_a + (g.castles_b or 0)) >= 6:
            notes.append(
                {
                    "kind": "high_castles",
                    "matchup": f"{g.bot_a} vs {g.bot_b}",
                    "seed": g.seed,
                    "castles": f"{g.castles_a} vs {g.castles_b}",
                    "winner": g.winner_bot,
                }
            )
    return notes[:20]


ROUND_LOCAL_BANNER = (
    "**Round-local ratings.** Fitted over this round's games only and anchored "
    "on the round's most-played bot, keyed on `bot_id` rather than on the "
    "content hash. They are **not comparable** to `data/ratings/leaderboard.md` "
    "or to any other round. Use the global fit for decisions."
)


def round_leaderboard_snippet(games: list[GameEntry]) -> str:
    """
    Ratings from this round's games alone, explicitly labelled as such.

    The old version built a fresh sequential Elo book in list order, so every
    published round report showed numbers that disagreed with the global
    leaderboard *and* with any other run of the same round. This one is a real
    batch fit, but it is still round-local: it sees a fraction of the games and
    a different anchor, so it carries a banner saying so.
    """
    from arena.records.ratings.counts import build
    from arena.records.ratings.fit import fit_ratings
    from arena.records.ratings.io import leaderboard_rows

    tallies: dict[tuple[str, str], list[int]] = {}
    appearances: dict[str, int] = {}
    for g in games:
        cell = tallies.setdefault((g.bot_a, g.bot_b), [0, 0, 0])
        cell[0 if g.winner == "a" else 1 if g.winner == "b" else 2] += 1
        for bot in (g.bot_a, g.bot_b):
            appearances[bot] = appearances.get(bot, 0) + 1
    if not tallies:
        return "_no games in this round_"

    table = build({k: (v[0], v[1], v[2]) for k, v in tallies.items()})
    anchor = max(sorted(appearances), key=lambda b: appearances[b])
    fit = fit_ratings(table, anchor=anchor, policy=Policy(min_games_display=0))
    return "\n".join(
        [
            ROUND_LOCAL_BANNER,
            "",
            f"Anchor: `{anchor}` pinned at 1500.0.",
            "",
            *leaderboard_table_lines(leaderboard_rows(fit)),
        ]
    )


def write_reports(
    games: list[GameEntry],
    *,
    round_name: str,
    grid_desc: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = aggregate_stats(games)
    notable = notable_matchups(games)
    now = utc_now_iso()

    payload = {
        "round": round_name,
        "generated_at": now,
        "grid": grid_desc
        or {
            "rule": "C",
            "games_per_pair": "random map seeds per unordered pair; no seat swap",
        },
        "summary": stats,
        "notable_matchups": notable,
        "games": [
            {
                "game_id": g.game_id,
                "bot_a": g.bot_a,
                "bot_b": g.bot_b,
                "seed": g.seed,
                "winner": g.winner,
                "winner_bot": g.winner_bot,
                "turns": g.turns,
                "terminated": g.terminated,
                "truncated": g.truncated,
                "castles_a": g.castles_a,
                "castles_b": g.castles_b,
                "land_margin_a": g.land_margin_a,
                "land_margin_b": g.land_margin_b,
                "tag": g.tag,
            }
            for g in games
        ],
    }

    json_path = MEASUREMENTS_DIR / f"{round_name}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        f"# Heuristic measurement — {round_name}",
        "",
        f"Generated: {now}",
        f"Games: {stats['total_games']} | Draw rate: {stats['draw_rate']:.1%} | Mean turns: {stats['mean_turns']}",
        "",
        "## Winrate by bot",
        "",
        *winrate_table_lines(stats["by_bot"]),
    ]

    md_lines.extend(
        [
            "",
            "## Round Elo (this grid only)",
            "",
            round_leaderboard_snippet(games),
            "",
            "## Notable matchups",
            "",
        ]
    )
    if notable:
        for n in notable:
            if n["kind"] == "fast_win":
                md_lines.append(
                    f"- **fast_win**: `{n['winner']}` beat opponent in {n['turns']} turns "
                    f"({n['matchup']}, seed {n['seed']})"
                )
            elif n["kind"] == "high_castles":
                md_lines.append(
                    f"- **high_castles**: {n['matchup']} — {n['castles']} castles, "
                    f"{n['winner']} (seed {n['seed']})"
                )
            else:
                md_lines.append(f"- **{n['kind']}**: {n}")
    else:
        md_lines.append("- No decisive fast wins or high-castle games in this grid.")

    md_lines.extend(
        [
            "",
            "## Open questions",
            "",
            "- Which bots separate on Elo with games-per-pair sampling?",
            "- Are draw-heavy matchups truncating before strategic differences show?",
            "- Should the next round add expander_python or cm_* anchors?",
            "",
            f"Machine-readable: [`{round_name}.json`]({round_name}.json)",
            "",
        ]
    )

    md_path = MEASUREMENTS_DIR / f"{round_name}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def _run_legacy_grid(
    *,
    round_name: str,
    update_ratings: bool,
) -> list[GameEntry]:
    specs = build_grid()
    games_dir = round_games_dir(round_name)
    print(f"[measure] legacy grid: {len(specs)} match(es) -> {games_dir}")
    games: list[GameEntry] = []
    for i, spec in enumerate(specs, start=1):
        print(
            f"[measure] ({i}/{len(specs)}) {spec.bot_a} vs {spec.bot_b} "
            f"seed={spec.seed} [{spec.tag}]"
        )
        entry = run_one(
            spec, update_ratings=False, round_name=round_name, games_dir=games_dir
        )
        games.append(entry)
        print(
            f"[measure]   -> {entry.winner_bot} turns={entry.turns} "
            f"terminated={entry.terminated} truncated={entry.truncated}"
        )
    if update_ratings:
        from arena.records.ratings.cli import refit
        from arena.records.store import GAMES_DIR

        fit = refit(games_dir=GAMES_DIR)
        print(f"[measure] refitted ratings ({fit.counts.games} game(s))")
    return games


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run heuristic measurement via parallel tournament "
            "(games-per-pair × roster pairs)."
        )
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="poll until every bot run.sh exists (default: fail fast if missing)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=20.0,
        help="poll interval when --wait is set (default: 20)",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=120.0,
        help="max wait for bots in seconds when --wait is set (default: 120)",
    )
    parser.add_argument(
        "--no-ratings",
        action="store_true",
        help="store games only; skip global Elo rebuild",
    )
    parser.add_argument(
        "--round",
        default=None,
        help="round name under data/games/<round>/ and docs/research/measurements/ "
        "(default: round-YYYYMMDDTHHMMSSZ)",
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
        "--seat-policy",
        choices=SEAT_POLICIES,
        default=RANDOM_SEATS,
        help=(
            "'random' draws the seat per game (zero extra games); 'alternate' "
            f"plays every seed both ways for exactly 50/50 (default: {RANDOM_SEATS})"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=f"parallel workers (default: physical cores = {default_jobs()})",
    )
    parser.add_argument(
        "--bots",
        nargs="+",
        default=None,
        help="bot ids to include (default: NEW_BOTS + BASELINE_BOTS)",
    )
    parser.add_argument(
        "--legacy-grid",
        action="store_true",
        help="run the old tagged seat-swap grid instead of Rule C",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="optional per-match wall-clock timeout in seconds",
    )
    args = parser.parse_args(argv)

    round_name = args.round or datetime.now(timezone.utc).strftime("round-%Y%m%dT%H%M%SZ")
    roster = list(args.bots) if args.bots else list(DEFAULT_ROSTER)

    if args.legacy_grid:
        required = NEW_BOTS + BASELINE_BOTS + BENCHMARK_BOTS
    else:
        required = roster

    if args.wait:
        wait_for_bots(required, poll_seconds=args.poll_seconds, timeout_seconds=args.wait_timeout)
    else:
        missing = [n for n in required if not bot_run_sh(n).exists()]
        if missing:
            print(f"[measure] missing bots: {', '.join(missing)}", file=sys.stderr)
            return 1

    if args.legacy_grid:
        games = _run_legacy_grid(
            round_name=round_name,
            update_ratings=not args.no_ratings,
        )
        grid_desc = {
            "legacy": True,
            "new_vs_smoke": "each new bot vs smoke seeds 0,1",
            "new_vs_expand_plus": "each new bot vs expand_plus seed 0",
            "new_round_robin": "round-robin among new bots seed 0",
            "economy_cluster": "castle_builder vs castle_rush vs phase_switch seeds 0,1",
            "benchmark_vs_smoke": "each cm_* bot vs smoke seeds 0,1",
            "benchmark_vs_army_convey": "each cm_* bot vs army_convey seed 0",
        }
    else:
        if args.games_per_pair < 1:
            parser.error("--games-per-pair must be >= 1")
        if args.jobs is not None and args.jobs < 1:
            parser.error("--jobs must be >= 1")

        from arena.tournaments.competition import run_tournament

        fixed = parse_seeds(args.seeds) if args.seeds else None
        run_scripts = [bot_run_sh(n) for n in roster]
        records = run_tournament(
            run_scripts,
            round_name=round_name,
            games_per_pair=args.games_per_pair,
            round_seed=args.round_seed,
            fixed_seeds=fixed,
            games_dir=round_games_dir(round_name),
            update_ratings=not args.no_ratings,
            timeout=args.timeout,
            seat_policy=args.seat_policy,
            jobs=args.jobs,
        )
        games = [game_entry_from_record(r, tag="pair") for r in records]
        grid_desc = {
            "rule": "C",
            "roster": roster,
            "games_per_pair": args.games_per_pair if fixed is None else len(fixed),
            "round_seed": args.round_seed,
            "fixed_seeds": fixed,
            "seat_policy": args.seat_policy,
            "jobs": args.jobs if args.jobs is not None else default_jobs(),
            "games_dir": str(round_games_dir(round_name)),
        }

    json_path, md_path = write_reports(games, round_name=round_name, grid_desc=grid_desc)
    print(f"[measure] wrote {json_path}")
    print(f"[measure] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
