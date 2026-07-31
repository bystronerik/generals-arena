"""
Per-round count caches: faster, never different.

The one property that matters is that the cache changes only how much work a
refit does, never its answer.
"""
from __future__ import annotations

import json

import pytest

from arena.records.ratings.cache import (
    ROOT_ROUND,
    cached_count_table,
    round_directories,
    round_signature,
    rules_key,
)
from arena.records.ratings.counts import count_table
from arena.records.ratings.policy import Policy
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


def _table(games_tree, cache_dir, **kwargs):
    return cached_count_table(
        games_dir=games_tree,
        cache_dir=cache_dir,
        policy=Policy(engine_version=ENGINE),
        registry=FakeRegistry(),
        **kwargs,
    )


def test_round_directories_include_loose_root_games(games_tree):
    names = [name for name, _ in round_directories(games_tree)]
    assert names[0] == ROOT_ROUND
    assert names[1:] == ["round0", "round1", "round2"]


def test_cached_table_equals_the_uncached_one(games_tree, tmp_path):
    from arena.records.store import list_game_paths, load_game

    every_game = [
        load_game(p)
        for _, directory in round_directories(games_tree)
        for p in list_game_paths(directory)
    ]
    direct = count_table(
        every_game, policy=Policy(engine_version=ENGINE), registry=FakeRegistry()
    )
    cached, stats = _table(games_tree, tmp_path / "cache")

    assert cached.digest == direct.digest
    assert cached.cells == direct.cells
    assert stats.misses == 4 and stats.hits == 0


def test_a_second_pass_hits_every_round(games_tree, tmp_path):
    cache_dir = tmp_path / "cache"
    first, _ = _table(games_tree, cache_dir)
    second, stats = _table(games_tree, cache_dir)

    assert stats.hits == 4 and stats.misses == 0
    assert second.digest == first.digest


def test_adding_a_game_invalidates_only_that_round(games_tree, tmp_path):
    cache_dir = tmp_path / "cache"
    before, _ = _table(games_tree, cache_dir)

    save_game(record("smoke", "blitz", "a", index=9999), games_tree / "round1")
    after, stats = _table(games_tree, cache_dir)

    assert stats.misses == 1 and stats.hits == 3
    assert after.digest != before.digest
    assert after.games == before.games + 1


def test_changing_the_policy_invalidates_every_round(games_tree, tmp_path):
    cache_dir = tmp_path / "cache"
    _table(games_tree, cache_dir)
    _, stats = cached_count_table(
        games_dir=games_tree,
        cache_dir=cache_dir,
        policy=Policy(engine_version=ENGINE, include_self_play=False),
        registry=FakeRegistry(),
    )
    assert stats.misses == 4 and stats.hits == 0


def test_registering_a_hash_invalidates_the_cache():
    """Eligibility can change with no game file touched."""
    policy = Policy()
    lean = FakeRegistry({"smoke": "aaaaaaaaaaaa"})
    full = FakeRegistry()
    assert rules_key(policy, lean) != rules_key(policy, full)


def test_rules_key_ignores_the_registry_when_registration_is_not_required():
    policy = Policy(require_registered=False)
    assert rules_key(policy, FakeRegistry({})) == rules_key(policy, FakeRegistry())


def test_signature_changes_when_a_game_is_rewritten(games_tree):
    round_dir = games_tree / "round2"
    before = round_signature(round_dir)
    target = sorted(round_dir.glob("*.json"))[0]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["winner"] = "draw"
    target.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert round_signature(round_dir) != before


def test_use_cache_false_writes_nothing(games_tree, tmp_path):
    cache_dir = tmp_path / "cache"
    table, stats = _table(games_tree, cache_dir, use_cache=False)
    assert stats.misses == 4
    assert not cache_dir.exists()
    assert table.games > 0


def test_extra_entities_survive_the_cache(games_tree, tmp_path):
    table, _ = _table(games_tree, tmp_path / "cache", extra_entities=["anchor@ffff"])
    assert "anchor@ffff" in table.entities
