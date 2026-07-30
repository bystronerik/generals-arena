#!/usr/bin/env python3
"""Smart-grid heuristic bot measurement suite (round 1).

Runs a fixed matchup grid via arena.run_match, stores games under data/games/,
updates ratings, and writes docs/research/measurements/round1.{json,md}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.ratings import RatingBook, rate_stored_game, write_leaderboard
from arena.run_match import run_matchup
from arena.store import GAMES_DIR, GameRecord, bot_id_from_run_sh, git_commit_or_tag, make_game_id, save_game, utc_now_iso

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
]

BASELINE_BOTS = ["smoke", "expand_plus", "castle_builder", "general_hunter"]

EXPANDER_PYTHON = (
    REPO_ROOT / "competition-module" / "competition" / "agents" / "expander_python" / "run.sh"
)

_CASTLE_RE = re.compile(
    r"\[matchup\] castles built: (?P<a>\d+) \((?P<a_label>[^)]+)\) "
    r"vs (?P<b>\d+) \((?P<b_label>[^)]+)\)"
)


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
    game_id: str = ""
    tag: str = ""


def bot_run_sh(name: str) -> Path:
    if name == "expander_python":
        return EXPANDER_PYTHON
    return REPO_ROOT / "bots" / name / "run.sh"


def wait_for_bots(
    names: list[str],
    *,
    poll_seconds: float = 20.0,
    timeout_seconds: float = 1200.0,
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


def parse_castles(combined: str) -> tuple[int, int] | None:
    match = _CASTLE_RE.search(combined)
    if not match:
        return None
    return int(match.group("a")), int(match.group("b"))


def build_grid() -> list[MatchSpec]:
    specs: list[MatchSpec] = []

    for bot in NEW_BOTS:
        for seed in (0, 1):
            specs.append(MatchSpec(bot, "smoke", seed, "new_vs_smoke"))

    for bot in NEW_BOTS:
        specs.append(MatchSpec(bot, "expand_plus", 0, "new_vs_expand_plus"))

    for a, b in itertools.combinations(NEW_BOTS, 2):
        specs.append(MatchSpec(a, b, 0, "new_round_robin"))

    economy = ["castle_builder", "castle_rush", "phase_switch"]
    for a, b in itertools.combinations(economy, 2):
        for seed in (0, 1):
            specs.append(MatchSpec(a, b, seed, "economy_cluster"))

    return specs


def winner_bot_id(record: GameRecord) -> str:
    if record.winner == "a":
        return record.bot_a
    if record.winner == "b":
        return record.bot_b
    return "draw"


def run_one(spec: MatchSpec, *, update_ratings: bool) -> GameEntry:
    a_path = bot_run_sh(spec.bot_a)
    b_path = bot_run_sh(spec.bot_b)
    bot_a = bot_id_from_run_sh(a_path)
    bot_b = bot_id_from_run_sh(b_path)
    commit = git_commit_or_tag()

    started_at = utc_now_iso()
    result = run_matchup(a_path, b_path, seed=spec.seed, mode="competition")
    finished_at = utc_now_iso()
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    castles = parse_castles(combined)

    record = GameRecord(
        game_id=make_game_id(bot_a, bot_b, spec.seed),
        seed=spec.seed,
        mode="competition",
        bot_a=bot_a,
        bot_b=bot_b,
        bot_a_commit_or_tag=commit,
        bot_b_commit_or_tag=commit,
        winner=result.winner,  # type: ignore[arg-type]
        turns=result.turns,
        terminated=result.terminated,
        truncated=result.truncated,
        started_at=started_at,
        finished_at=finished_at,
    )
    save_game(record, GAMES_DIR)
    if update_ratings:
        rate_stored_game(record)

    entry = GameEntry(
        bot_a=bot_a,
        bot_b=bot_b,
        seed=spec.seed,
        winner=record.winner,
        winner_bot=winner_bot_id(record),
        turns=record.turns,
        terminated=record.terminated,
        truncated=record.truncated,
        game_id=record.game_id,
        tag=spec.tag,
    )
    if castles is not None:
        entry.castles_a, entry.castles_b = castles
    return entry


def aggregate_stats(games: list[GameEntry]) -> dict[str, Any]:
    bot_games: dict[str, list[GameEntry]] = defaultdict(list)
    for g in games:
        bot_games[g.bot_a].append(g)
        bot_games[g.bot_b].append(g)

    rows: list[dict[str, Any]] = []
    for bot_id in sorted(bot_games):
        played = bot_games[bot_id]
        wins = sum(1 for g in played if g.winner_bot == bot_id)
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

    total_draws = sum(1 for g in games if g.winner == "draw")
    return {
        "total_games": len(games),
        "draw_rate": round(total_draws / len(games), 3) if games else 0.0,
        "mean_turns": round(sum(g.turns for g in games) / len(games), 1) if games else 0.0,
        "by_bot": rows,
    }


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


def round_leaderboard_snippet(games: list[GameEntry]) -> str:
    book = RatingBook()
    for g in games:
        record = GameRecord(
            game_id=g.game_id,
            seed=g.seed,
            mode="competition",
            bot_a=g.bot_a,
            bot_b=g.bot_b,
            bot_a_commit_or_tag="",
            bot_b_commit_or_tag="",
            winner=g.winner,  # type: ignore[arg-type]
            turns=g.turns,
            terminated=g.terminated,
            truncated=g.truncated,
            started_at="",
            finished_at="",
        )
        book.apply_game(record, skip_if_rated=False)

    rows = book.leaderboard()
    lines = [
        "| Rank | Bot | Elo | Games | W | L | D |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.rank} | `{r.bot_id}` | {r.rating:.1f} | {r.games} "
            f"| {r.wins} | {r.losses} | {r.draws} |"
        )
    return "\n".join(lines)


def write_reports(games: list[GameEntry], *, round_name: str = "round1") -> tuple[Path, Path]:
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = aggregate_stats(games)
    notable = notable_matchups(games)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    payload = {
        "round": round_name,
        "generated_at": now,
        "grid": {
            "new_vs_smoke": "each new bot vs smoke seeds 0,1",
            "new_vs_expand_plus": "each new bot vs expand_plus seed 0",
            "new_round_robin": "round-robin among 8 new bots seed 0",
            "economy_cluster": "castle_builder vs castle_rush vs phase_switch seeds 0,1",
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
            "- Do economy-cluster bots (castle_builder, castle_rush, phase_switch) separate on Elo?",
            "- Which new bots beat smoke on both seeds 0 and 1?",
            "- Are draw-heavy matchups truncating before strategic differences show?",
            "- Should the next round add expander_python or general_hunter as anchors?",
            "",
            f"Machine-readable: [`{round_name}.json`]({round_name}.json)",
            "",
        ]
    )

    md_path = MEASUREMENTS_DIR / f"{round_name}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run round-1 heuristic measurement grid.")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="fail immediately if a bot run.sh is missing",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=20.0,
        help="poll interval when waiting for bots (default: 20)",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=1200.0,
        help="max wait for bots in seconds (default: 1200 = 20 min)",
    )
    parser.add_argument(
        "--no-ratings",
        action="store_true",
        help="store games only; skip global Elo update",
    )
    parser.add_argument(
        "--round",
        default="round1",
        help="output basename under docs/research/measurements/ (default: round1)",
    )
    args = parser.parse_args(argv)

    required = NEW_BOTS + BASELINE_BOTS
    if args.no_wait:
        missing = [n for n in required if not bot_run_sh(n).exists()]
        if missing:
            print(f"[measure] missing bots: {', '.join(missing)}", file=sys.stderr)
            return 1
    else:
        wait_for_bots(required, poll_seconds=args.poll_seconds, timeout_seconds=args.wait_timeout)

    specs = build_grid()
    print(f"[measure] running {len(specs)} match(es)")
    games: list[GameEntry] = []
    for i, spec in enumerate(specs, start=1):
        print(
            f"[measure] ({i}/{len(specs)}) {spec.bot_a} vs {spec.bot_b} "
            f"seed={spec.seed} [{spec.tag}]"
        )
        entry = run_one(spec, update_ratings=not args.no_ratings)
        games.append(entry)
        print(
            f"[measure]   -> {entry.winner_bot} turns={entry.turns} "
            f"terminated={entry.terminated} truncated={entry.truncated}"
        )

    json_path, md_path = write_reports(games, round_name=args.round)
    print(f"[measure] wrote {json_path}")
    print(f"[measure] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
