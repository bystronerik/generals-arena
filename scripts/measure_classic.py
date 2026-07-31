#!/usr/bin/env python3
"""Classic-approximate bot measurement grid.

Runs N seeds × bot pairs via arena.classic_tournament, stores games under
data/classic_games/, and writes docs/research/measurements/<round>.{json,md}.
Classic results never enter data/games/ or Elo.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.classic_tournament import (
    CLASSIC_GAMES_DIR,
    ClassicGameRecord,
    build_match_specs,
    run_classic_tournament,
)
from arena.store import bot_id_from_run_sh, utc_now_iso
from arena.tournament import parse_seeds

MEASUREMENTS_DIR = REPO_ROOT / "docs" / "research" / "measurements"


def bot_run_sh(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".sh" or path.is_dir():
        if path.is_dir():
            return path / "run.sh"
        return path
    return REPO_ROOT / "bots" / name_or_path / "run.sh"


def winner_bot_id(record: ClassicGameRecord) -> str:
    if record.winner == "a":
        return record.bot_a
    if record.winner == "b":
        return record.bot_b
    return "draw"


def aggregate_stats(records: list[ClassicGameRecord]) -> dict[str, Any]:
    bot_games: dict[str, list[ClassicGameRecord]] = defaultdict(list)
    for record in records:
        bot_games[record.bot_a].append(record)
        bot_games[record.bot_b].append(record)

    rows: list[dict[str, Any]] = []
    for bot_id in sorted(bot_games):
        played = bot_games[bot_id]
        wins = sum(1 for g in played if winner_bot_id(g) == bot_id)
        draws = sum(1 for g in played if g.winner == "draw")
        losses = len(played) - wins - draws
        turns = [g.turns for g in played]
        rows.append(
            {
                "bot_id": bot_id,
                "games": len(played),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "winrate": round(wins / len(played), 3) if played else 0.0,
                "draw_rate": round(draws / len(played), 3) if played else 0.0,
                "mean_turns": round(sum(turns) / len(turns), 1) if turns else 0.0,
            }
        )
    rows.sort(key=lambda r: (-r["winrate"], r["bot_id"]))

    total_draws = sum(1 for g in records if g.winner == "draw")
    return {
        "total_games": len(records),
        "draw_rate": round(total_draws / len(records), 3) if records else 0.0,
        "mean_turns": round(sum(g.turns for g in records) / len(records), 1) if records else 0.0,
        "by_bot": rows,
    }


def matchup_table(records: list[ClassicGameRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ClassicGameRecord]] = defaultdict(list)
    for record in records:
        key = tuple(sorted((record.bot_a, record.bot_b)))
        grouped[key].append(record)

    rows: list[dict[str, Any]] = []
    for (bot_a, bot_b), games in sorted(grouped.items()):
        wins_a = sum(1 for g in games if winner_bot_id(g) == bot_a)
        wins_b = sum(1 for g in games if winner_bot_id(g) == bot_b)
        draws = sum(1 for g in games if g.winner == "draw")
        rows.append(
            {
                "bot_a": bot_a,
                "bot_b": bot_b,
                "games": len(games),
                "wins_a": wins_a,
                "wins_b": wins_b,
                "draws": draws,
            }
        )
    return rows


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
        "| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in stats["by_bot"]:
        md_lines.append(
            f"| `{row['bot_id']}` | {row['games']} | {row['wins']} | {row['losses']} "
            f"| {row['draws']} | {row['winrate']:.1%} | {row['draw_rate']:.1%} "
            f"| {row['mean_turns']} |"
        )

    md_lines.extend(
        [
            "",
            "## Matchups",
            "",
            "| Bot A | Bot B | Games | A wins | B wins | Draws |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in matchups:
        md_lines.append(
            f"| `{row['bot_a']}` | `{row['bot_b']}` | {row['games']} "
            f"| {row['wins_a']} | {row['wins_b']} | {row['draws']} |"
        )

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
        print("[measure_classic] --jobs must be >= 1", file=sys.stderr)
        return 1

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
