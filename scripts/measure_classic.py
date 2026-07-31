#!/usr/bin/env python3
"""Classic-approximate bot measurement grid.

Runs N seeds × bot pairs via arena.tournaments.classic, stores games under
data/classic_games/, and writes docs/research/measurements/<round>.{json,md}.
Classic results never enter data/games/ or the rating fit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.tournaments.classic import (
    CLASSIC_GAMES_DIR,
    ClassicGameRecord,
    build_match_specs,
    run_classic_tournament,
)
from arena.records.reporting import (
    aggregate_stats,
    bot_run_sh,
    matchup_table,
    matchup_table_lines,
    winner_bot_id,
    winrate_table_lines,
)
from arena.records.store import bot_id_from_run_sh, utc_now_iso
from arena.tournaments.competition import parse_seeds

MEASUREMENTS_DIR = REPO_ROOT / "docs" / "research" / "measurements"


def write_reports(
    records: list[ClassicGameRecord],
    *,
    round_name: str,
    seeds: list[int],
    swap_sides: bool,
) -> tuple[Path, Path]:
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = aggregate_stats(records)
    matchups = matchup_table(records)
    now = utc_now_iso()

    payload = {
        "round": round_name,
        "mode": "classic",
        "generated_at": now,
        "seeds": seeds,
        "swap_sides": swap_sides,
        "summary": stats,
        "matchups": matchups,
        "games": [
            {
                "game_id": g.game_id,
                "bot_a": g.bot_a,
                "bot_b": g.bot_b,
                "seed": g.seed,
                "winner": g.winner,
                "winner_bot": winner_bot_id(g),
                "turns": g.turns,
                "terminated": g.terminated,
                "truncated": g.truncated,
                "grid_size": g.grid_size,
                "truncation_limit": g.truncation_limit,
            }
            for g in records
        ],
    }

    json_path = MEASUREMENTS_DIR / f"{round_name}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        f"# Classic measurement — {round_name}",
        "",
        f"Generated: {now}",
        f"Mode: classic (not Elo) | Seeds: {','.join(str(s) for s in seeds)} | "
        f"Swap sides: {swap_sides}",
        f"Games: {stats['total_games']} | Draw rate: {stats['draw_rate']:.1%} | "
        f"Mean turns: {stats['mean_turns']}",
        "",
        "Stored under `data/classic_games/` only — never `data/games/`.",
        "",
        "## Winrate by bot",
        "",
        *winrate_table_lines(stats["by_bot"]),
        "",
        "## Matchups",
        "",
        *matchup_table_lines(matchups),
    ]

    md_lines.extend(
        [
            "",
            f"Machine-readable: [`{round_name}.json`]({round_name}.json)",
            "",
        ]
    )

    md_path = MEASUREMENTS_DIR / f"{round_name}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a classic measurement grid; store under data/classic_games/."
    )
    parser.add_argument(
        "bots",
        nargs="+",
        help="bot names (under bots/) or paths to run.sh",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma list and/or ranges, e.g. 0-4 (default: 0)",
    )
    parser.add_argument(
        "--round",
        default="classic-round",
        help="output basename under docs/research/measurements/ (default: classic-round)",
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=CLASSIC_GAMES_DIR,
        help=f"classic game JSON directory (default: {CLASSIC_GAMES_DIR})",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="square board side (classic default: 24)",
    )
    parser.add_argument(
        "--truncation",
        type=int,
        default=None,
        help="max turns before stop (classic default: 5000)",
    )
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help="also play each pair with sides swapped",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="also play each bot against itself",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel match workers (default: 1)",
    )
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    run_scripts = [bot_run_sh(name) for name in args.bots]
    missing = [str(p) for p in run_scripts if not p.exists()]
    if missing:
        print(f"[measure_classic] missing run.sh: {', '.join(missing)}", file=sys.stderr)
        return 1

    seeds = parse_seeds(args.seeds)
    specs = build_match_specs(
        run_scripts,
        seeds,
        include_self=args.include_self,
        swap_sides=args.swap_sides,
    )
    print(f"[measure_classic] running {len(specs)} classic match(es)")
    for i, spec in enumerate(specs, start=1):
        a_id = bot_id_from_run_sh(spec.bot_a_run)
        b_id = bot_id_from_run_sh(spec.bot_b_run)
        print(f"[measure_classic] plan ({i}/{len(specs)}) {a_id} vs {b_id} seed={spec.seed}")

    records = run_classic_tournament(
        run_scripts,
        seeds,
        games_dir=args.games_dir,
        include_self=args.include_self,
        swap_sides=args.swap_sides,
        grid_size=args.grid_size,
        truncation=args.truncation,
        jobs=args.jobs,
    )

    json_path, md_path = write_reports(
        records,
        round_name=args.round,
        seeds=seeds,
        swap_sides=args.swap_sides,
    )
    print(f"[measure_classic] wrote {json_path}")
    print(f"[measure_classic] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
