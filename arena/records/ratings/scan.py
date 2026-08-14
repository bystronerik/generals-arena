"""
Walk `data/games/` and turn each round into its count table.

One round in, one `RoundCounts` out. The round is the unit ratings are fitted
over (`rounds.py`), so this module hands back a list in round order and does
nothing else with it.

This module used to cache the tables to `data/ratings/cache/`. The cache was
removed: it saved about one second on a full refit of a 9,000-game store, and
cost a signature scheme, a rules digest, a prune step and a staleness API that
could disagree with the games on disk. Reading every game is 1.3 s and cannot
go stale. Reinstate a cache when a cold refit costs seconds, not before.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from arena.records.ratings.counts import CountTable, count_table
from arena.records.ratings.policy import Policy
from arena.records.registry import Registry
from arena.records.store import GAMES_DIR, list_game_paths, load_game

# Games sitting directly in data/games/ rather than in a round folder.
ROOT_ROUND = "_root"


@dataclass(frozen=True)
class RoundCounts:
    """One round's sufficient statistics, plus the provenance a report needs."""

    round: str
    table: CountTable
    # Distinct `engine_version` values over the round's stored games, before the
    # era filter. More than one means the round spans an engine bump, which the
    # era filter then silently halves — so it is reported, not inferred.
    engine_versions: tuple[str, ...] = ()

    @property
    def stored_games(self) -> int:
        """Records in the round: the rated ones plus every rejection."""
        return self.table.games + sum(self.table.excluded.values())

    @property
    def era_split(self) -> bool:
        return len(self.engine_versions) > 1


def round_directories(
    games_dir: Path, *, include_root: bool = True
) -> list[tuple[str, Path]]:
    """
    `(round name, directory)` for every round.

    `include_root=False` drops the loose `data/games/*.json` bucket. Those files
    carry a `round` field that no directory backs, so they are not a round and
    cannot be fitted as one — see `rounds.fit_rounds`, the one caller that
    passes it.
    """
    rounds: list[tuple[str, Path]] = []
    if not games_dir.exists():
        return rounds
    if include_root and any(
        p.is_file() and p.name != "manifest.json" for p in games_dir.glob("*.json")
    ):
        rounds.append((ROOT_ROUND, games_dir))
    for child in sorted(games_dir.iterdir()):
        if child.is_dir():
            rounds.append((child.name, child))
    return rounds


def round_count_tables(
    *,
    games_dir: Path | None = None,
    policy: Policy,
    registry: Registry | None = None,
    include_root: bool = True,
) -> list[RoundCounts]:
    """
    One count table per round, in the order `round_directories` walks them.

    This is the whole read of `data/games/`. Building a table is integer
    addition over the round's games, so the result is bit-identical for any
    file order — see `counts.py`.
    """
    directory = games_dir or GAMES_DIR
    rounds: list[RoundCounts] = []
    for round_name, round_dir in round_directories(directory, include_root=include_root):
        games = [load_game(p) for p in list_game_paths(round_dir)]
        rounds.append(
            RoundCounts(
                round=round_name,
                table=count_table(games, policy=policy, registry=registry),
                engine_versions=tuple(sorted({g.engine_version for g in games})),
            )
        )
    return rounds


__all__ = [
    "ROOT_ROUND",
    "RoundCounts",
    "round_count_tables",
    "round_directories",
]
