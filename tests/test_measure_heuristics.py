"""Tests for scripts/measure_heuristics.py grid and record mapping."""
from __future__ import annotations

from scripts.measure_heuristics import (
    MatchSpec,
    both_seat_orders,
    build_grid,
    game_entry_from_record,
)
from arena.records.store import GameRecord


def test_both_seat_orders_emits_swap():
    specs = [MatchSpec("a", "b", 0, "pair")]
    got = both_seat_orders(specs)
    assert len(got) == 2
    assert got[0] == MatchSpec("a", "b", 0, "pair")
    assert got[1] == MatchSpec("b", "a", 0, "pair")


def test_both_seat_orders_dedupes():
    specs = [
        MatchSpec("a", "b", 0, "pair"),
        MatchSpec("b", "a", 0, "pair"),
    ]
    assert len(both_seat_orders(specs)) == 2


def test_build_grid_includes_swapped_seat_orders():
    grid = build_grid()
    keys = {(g.bot_a, g.bot_b, g.seed, g.tag) for g in grid}
    for bot_a, bot_b, seed, tag in list(keys):
        assert (bot_b, bot_a, seed, tag) in keys


def test_game_entry_from_record_derives_castles_and_land_margin():
    record = GameRecord(
        game_id="g1",
        seed=0,
        mode="competition",
        round="roundT",
        bot_a="smoke",
        bot_b="rush",
        bot_a_content_hash="0123456789ab",
        bot_b_content_hash="ba9876543210",
        engine_version="9e3b9d1",
        winner="draw",
        turns=1200,
        truncated=True,
        metrics={
            "castles_built_a": 2,
            "castles_built_b": 1,
            "land_margin_a": 15,
            "land_margin_b": -15,
        },
    )
    entry = game_entry_from_record(record, tag="economy_cluster")
    assert entry.castles_a == 2
    assert entry.castles_b == 1
    assert entry.land_margin_a == 15
    assert entry.land_margin_b == -15
    assert entry.tag == "economy_cluster"
