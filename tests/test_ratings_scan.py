"""
Walking `data/games/` into one count table per round.

What matters here is the round split itself: which directories become rounds,
that `_root` can be excluded, that a round remembers the engine eras of its own
games, and that splitting by round loses no games. The per-round count cache
these tests used to cover was removed — it saved ~1 s on a full refit and cost a
staleness surface that could disagree with the games on disk.
"""
from __future__ import annotations

import pytest

from arena.records.ratings.counts import count_table, merge
from arena.records.ratings.policy import Policy
from arena.records.ratings.scan import (
    ROOT_ROUND,
    round_count_tables,
    round_directories,
)
from arena.records.store import save_game
from test_ratings import ENGINE, FakeRegistry, record, synthetic_games


@pytest.fixture
def games_tree(tmp_path):
    """Three rounds plus a couple of loose games at the root."""
    root = tmp_path / "games"
    games = synthetic_games(120)
    for i, game in enumerate(games):
        directory = root if i < 4 else root / f"round{i % 3}"
        save_game(game, directory)
    (root / "round0" / "manifest.json").write_text("{}", encoding="utf-8")
    return root


def _tables(games_tree, **kwargs):
    return round_count_tables(
        games_dir=games_tree,
        policy=Policy(engine_version=ENGINE),
        registry=FakeRegistry(),
        **kwargs,
    )


def test_round_directories_include_loose_root_games(games_tree):
    names = [name for name, _ in round_directories(games_tree)]
    assert names[0] == ROOT_ROUND
    assert names[1:] == ["round0", "round1", "round2"]


def test_round_directories_can_exclude_root(games_tree):
    """The `_root` exclusion the per-round fitter relies on, in one flag."""
    names = [name for name, _ in round_directories(games_tree, include_root=False)]
    assert names == ["round0", "round1", "round2"]


def test_one_table_per_round_in_directory_order(games_tree):
    rounds = _tables(games_tree, include_root=False)
    assert [r.round for r in rounds] == ["round0", "round1", "round2"]
    assert all(r.table.games > 0 for r in rounds)


def test_a_round_records_the_engine_eras_of_its_stored_games(games_tree):
    """An era-spanning round must be visible in the report, not inferred."""
    save_game(
        record("smoke", "blitz", "a", index=8888, engine="0" * 40),
        games_tree / "round1",
    )
    rounds = {r.round: r for r in _tables(games_tree, include_root=False)}

    assert rounds["round1"].engine_versions == ("0" * 40, ENGINE)
    assert rounds["round1"].era_split
    assert not rounds["round0"].era_split


def test_splitting_by_round_loses_no_games(games_tree):
    """
    Merging the per-round tables reproduces one table over every game.

    Count tables are exactly additive, so the round split is a regrouping of
    the same integers rather than a different reading of the store.
    """
    from arena.records.store import list_game_paths, load_game

    every_game = [
        load_game(p)
        for _, directory in round_directories(games_tree)
        for p in list_game_paths(directory)
    ]
    direct = count_table(
        every_game, policy=Policy(engine_version=ENGINE), registry=FakeRegistry()
    )
    merged = merge([r.table for r in _tables(games_tree)])

    assert merged.digest == direct.digest
    assert merged.cells == direct.cells


def test_a_fit_file_for_a_removed_round_is_pruned(tmp_path):
    """
    A stale *published fit* is a table for a round that no longer exists, with
    nothing on the table saying so.
    """
    from arena.records.ratings import io
    from test_ratings import fit_tree, games_tree as write_tree, pair_games

    root = write_tree(
        tmp_path / "games",
        {
            "kept": pair_games("smoke", "blitz", wins_a=2, wins_b=1, start=0),
            "removed": pair_games("aegis", "metro", wins_a=2, wins_b=1, start=50),
        },
    )
    ratings = tmp_path / "ratings"
    io.write_round_fits(fit_tree(root), ratings)
    assert sorted(io.available_rounds(ratings)) == ["kept", "removed"]

    for path in (root / "removed").glob("*.json"):
        path.unlink()
    (root / "removed").rmdir()
    written, pruned = io.write_round_fits(fit_tree(root), ratings)

    assert [p.stem for p in written] == ["kept"]
    assert [p.stem for p in pruned] == ["removed"]
    assert io.available_rounds(ratings) == ["kept"]
