"""
Shared measurement aggregation and markdown table rendering.

Used by `scripts/measure_heuristics.py` (competition rounds),
`scripts/measure_classic.py` (classic rounds), and
`arena.records.ratings.io`. The aggregation helpers are duck-typed on
anything carrying `bot_a`, `bot_b`, `winner`, and `turns` — GameRecord,
ClassicGameRecord, and the measure scripts' GameEntry all qualify — so the two
round types share one definition of winrate and one table layout.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol

from arena.paths import REPO_ROOT

EXPANDER_PYTHON = (
    REPO_ROOT / "competition-module" / "competition" / "agents" / "expander_python" / "run.sh"
)

WINRATE_TABLE_HEADER = (
    "| Bot | Games | W | L | D | Winrate | Draw rate | Mean turns |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
)

LEADERBOARD_TABLE_HEADER = (
    "| Rank | Entity | Rating | 95% CI | Games | W | L | D | Prov. |",
    "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
)


class GameLike(Protocol):
    """Minimum surface the aggregation helpers read off a finished game."""

    bot_a: str
    bot_b: str
    winner: str
    turns: int


def bot_run_sh(name_or_path: str, *, repo_root: Path | None = None) -> Path:
    """
    Resolve a bot reference to its run.sh.

    Accepts a bot name under `bots/`, a directory holding a run.sh, or a direct
    path to a .sh file. `expander_python` resolves to the competition-module
    agent of that name, which lives outside `bots/`.
    """
    root = repo_root or REPO_ROOT
    if name_or_path == "expander_python":
        return EXPANDER_PYTHON
    path = Path(name_or_path)
    if path.is_dir():
        return path / "run.sh"
    if path.suffix == ".sh":
        return path
    return root / "bots" / name_or_path / "run.sh"


def winner_bot_id(game: GameLike) -> str:
    """Bot id that won `game`, or "draw"."""
    if game.winner == "a":
        return game.bot_a
    if game.winner == "b":
        return game.bot_b
    return "draw"


def aggregate_stats(games: Iterable[GameLike]) -> dict[str, Any]:
    """Per-bot W/L/D, winrate, and mean turns, plus round-level totals."""
    games = list(games)
    bot_games: dict[str, list[GameLike]] = defaultdict(list)
    for game in games:
        bot_games[game.bot_a].append(game)
        bot_games[game.bot_b].append(game)

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

    total_draws = sum(1 for g in games if g.winner == "draw")
    return {
        "total_games": len(games),
        "draw_rate": round(total_draws / len(games), 3) if games else 0.0,
        "mean_turns": round(sum(g.turns for g in games) / len(games), 1) if games else 0.0,
        "by_bot": rows,
    }


def matchup_table(games: Iterable[GameLike]) -> list[dict[str, Any]]:
    """Head-to-head totals per unordered bot pair."""
    grouped: dict[tuple[str, str], list[GameLike]] = defaultdict(list)
    for game in games:
        key = tuple(sorted((game.bot_a, game.bot_b)))
        grouped[key].append(game)

    rows: list[dict[str, Any]] = []
    for (bot_a, bot_b), pair_games in sorted(grouped.items()):
        rows.append(
            {
                "bot_a": bot_a,
                "bot_b": bot_b,
                "games": len(pair_games),
                "wins_a": sum(1 for g in pair_games if winner_bot_id(g) == bot_a),
                "wins_b": sum(1 for g in pair_games if winner_bot_id(g) == bot_b),
                "draws": sum(1 for g in pair_games if g.winner == "draw"),
            }
        )
    return rows


def winrate_table_lines(by_bot: list[dict[str, Any]]) -> list[str]:
    """Render `aggregate_stats()["by_bot"]` as a markdown table."""
    lines = list(WINRATE_TABLE_HEADER)
    for row in by_bot:
        lines.append(
            f"| `{row['bot_id']}` | {row['games']} | {row['wins']} | {row['losses']} "
            f"| {row['draws']} | {row['winrate']:.1%} | {row['draw_rate']:.1%} "
            f"| {row['mean_turns']} |"
        )
    return lines


def matchup_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Render `matchup_table()` as a markdown table."""
    lines = [
        "| Bot A | Bot B | Games | A wins | B wins | Draws |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['bot_a']}` | `{row['bot_b']}` | {row['games']} "
            f"| {row['wins_a']} | {row['wins_b']} | {row['draws']} |"
        )
    return lines


def leaderboard_table_lines(rows) -> list[str]:
    """
    Render `ratings.io.leaderboard_rows()` as a markdown table.

    The interval is the point of the table: a 20-Elo gap over 30 games is
    indistinguishable from noise, and the old point-estimate-only layout gave
    a reader no way to see that. Provisional rows carry a `—` rank because
    they are in the fit but deliberately not ranked.
    """
    lines = list(LEADERBOARD_TABLE_HEADER)
    for r in rows:
        rank = "—" if r.provisional else str(r.rank)
        lines.append(
            f"| {rank} | `{r.entity}` | {r.rating:.1f} "
            f"| [{r.ci_low:.0f}, {r.ci_high:.0f}] | {r.games} "
            f"| {r.wins} | {r.losses} | {r.draws} "
            f"| {'yes' if r.provisional else ''} |"
        )
    return lines
