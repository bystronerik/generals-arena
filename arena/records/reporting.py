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

# Used instead when the pool is not connected. A single ranked list asserts
# that every row is comparable to every other; when the games do not link them
# that assertion is false, so the group has to be on the row.
LEADERBOARD_TABLE_HEADER_SPLIT = (
    "| Rank | Group | Entity | Rating | 95% CI | Games | W | L | D | Prov. |",
    "| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
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


def _round_connectivity_lines(result) -> list[str]:
    """
    The split warning, scoped to one round.

    Scoped is the point: a split in one round must not annotate another, and the
    group holding the anchor has to be named — when the anchor lands in a
    minority group, most of the round's rows are prior-located rather than
    measured.
    """
    if result.connected:
        return []
    sizes = ", ".join(str(len(group)) for group in result.components)
    lines = [
        f"> **This round is not connected: {len(result.components)} groups "
        f"({sizes} entities).**",
        ">",
        "> Groups share no games, so nothing links their scales — the offset",
        "> between them comes from the prior, not from evidence. Ratings and",
        "> ranks are meaningful **only within one group**. A contrast across",
        "> groups reports an infinite interval and `P(better) = 0.50`.",
    ]
    if result.anchor_component is not None:
        lines += [
            ">",
            f"> The anchor sits in group {result.anchor_component}, so that is the",
            "> group whose ratings are measured against it. Every other group's",
            "> location comes from the prior.",
        ]
    lines += [
        ">",
        "> Fix it by playing games between the groups, not by comparing anyway.",
        "",
    ]
    return lines


def round_section_lines(result, *, ranked=(), provisional=()) -> list[str]:
    """
    One round's section of `leaderboard.md`: header, warnings, tables.

    Duck-typed on `ratings.rounds.RoundResult` so this module keeps knowing
    nothing about ratings. Rank numbers restart at 1 in every section, which is
    correct and is why the document's banner says rank is per-round.
    """
    if not result.rated:
        return _unrated_section_lines(result)

    era = _era_phrase(result)
    groups = (
        "one connectivity group"
        if result.connected
        else f"{len(result.components)} connectivity groups"
    )
    seat, draw = result.seat_advantage, result.draw_log_nu
    lines = [
        f"## {result.round}",
        "",
        f"Scale `{result.scale_token}` · Anchor `{result.anchor}` "
        f"({result.anchor_kind}) pinned at {result.anchor_rating:.1f}",
        f"Rated {result.rated_games:,} of {result.stored_games:,} stored · {era} "
        f"· {groups}",
        f"Seat-A advantage {seat.value:+.1f} ± {seat.se:.1f} Elo "
        f"· Draw log-nu {draw.value:.3f} ± {draw.se:.3f}",
        "",
    ]
    if result.era_split:
        excluded = result.excluded.get("engine_mismatch", 0)
        lines += [
            f"**This round spans {len(result.engine_versions)} engine eras.** It is "
            f"rated on the era above only; {excluded:,} game(s) are excluded as "
            "`engine_mismatch`. A round is one design, so it is not split into "
            "two per-era tables — re-run it instead.",
            "",
        ]
    if result.solver is not None and not result.solver.converged:
        # The residual is on the line because it is what decides whether this
        # matters. A round with no draws leaves `draw_log_nu` running toward −∞
        # against its prior alone, which stalls just above the 1e-9 gradient
        # tolerance while every strength is settled far tighter than 2 dp.
        lines += [
            f"**The solver did not converge for this round** after "
            f"{result.solver.iterations} iterations, at max|grad| "
            f"{result.solver.max_abs_grad:.1e}. Read the residual before reading "
            f"the table: near the tolerance the strengths are settled, but a large "
            f"one means these numbers describe no optimum.",
            "",
        ]
    lines += _round_connectivity_lines(result)

    if ranked:
        lines += leaderboard_table_lines(ranked, split=not result.connected)
    else:
        lines += [
            f"No entity in this round reached {result.min_games_display} games, so "
            "this round ranks nobody and",
            "supplies no decision baseline. Every row below is provisional.",
        ]
    if provisional:
        lines += [
            "",
            f"### Provisional in this round (< {result.min_games_display} games)",
            "",
            *leaderboard_table_lines(provisional, split=not result.connected),
        ]
    return lines


def _unrated_section_lines(result) -> list[str]:
    """A stub section: no table, but every actionable count kept."""
    excluded = ", ".join(
        f"`{reason}` {count:,}" for reason, count in sorted(result.excluded.items())
    )
    lines = [
        f"## {result.round} — unrated",
        "",
        f"Stored {result.stored_games:,} games, rated {result.rated_games:,}. "
        f"**Not rated: {result.reason_text}.**",
    ]
    if excluded:
        lines.append(f"Excluded: {excluded}.")
    if result.remediation:
        lines += ["", result.remediation]
    return lines


def _era_phrase(result) -> str:
    versions = result.engine_versions
    if not versions:
        return "no stored engine era"
    if len(versions) == 1:
        return f"engine era `{versions[0][:12]}`"
    return "engine eras " + ", ".join(f"`{v[:12]}`" for v in versions)


def leaderboard_table_lines(rows, *, split: bool | None = None) -> list[str]:
    """
    Render `ratings.io.leaderboard_rows()` as a markdown table.

    The interval is the point of the table: a 20-Elo gap over 30 games is
    indistinguishable from noise, and the old point-estimate-only layout gave
    a reader no way to see that. Provisional rows carry a `—` rank because
    they are in the fit but deliberately not ranked.

    `split` adds the connectivity group to every row. Pass it whenever the
    pool has more than one component, so a rank is never read as a comparison
    between entities that never met.
    """
    rows = list(rows)
    # Decided by the caller, not inferred from `rows`: the ranked and
    # provisional tables are rendered separately, and either one on its own can
    # look connected while the pool is not.
    if split is None:
        split = len({r.component for r in rows}) > 1
    lines = list(LEADERBOARD_TABLE_HEADER_SPLIT if split else LEADERBOARD_TABLE_HEADER)
    for r in rows:
        rank = "—" if r.provisional else str(r.rank)
        group = f" {r.component} |" if split else ""
        lines.append(
            f"| {rank} |{group} `{r.entity}` | {r.rating:.1f} "
            f"| [{r.ci_low:.0f}, {r.ci_high:.0f}] | {r.games} "
            f"| {r.wins} | {r.losses} | {r.draws} "
            f"| {'yes' if r.provisional else ''} |"
        )
    return lines
