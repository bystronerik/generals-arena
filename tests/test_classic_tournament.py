"""Light tests for classic tournament helpers and record store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.classic_match import classic_winner_seat
from arena.classic_tournament import (
    ClassicGameRecord,
    ClassicMatchSpec,
    build_match_specs,
    save_classic_game,
)


@pytest.mark.parametrize(
    ("player_id", "truncated", "expected"),
    [
        (0, False, "a"),
        (1, True, "b"),
        (-1, True, "draw"),
        (-1, False, "draw"),
    ],
)
def test_classic_winner_seat(player_id, truncated, expected):
    assert classic_winner_seat(player_id, truncated=truncated) == expected


def test_build_match_specs_seeds_and_pairs(tmp_path: Path):
    a = tmp_path / "alpha" / "run.sh"
    b = tmp_path / "beta" / "run.sh"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/bash\n")

    specs = build_match_specs([a, b], [0, 1], swap_sides=True)
    assert len(specs) == 4
    assert all(isinstance(s, ClassicMatchSpec) for s in specs)
    seeds = {s.seed for s in specs}
    assert seeds == {0, 1}


def test_classic_game_record_round_trip(tmp_path: Path):
    record = ClassicGameRecord(
        game_id="test_game",
        seed=0,
        mode="classic",
        bot_a="a",
        bot_b="b",
        bot_a_commit_or_tag="abc",
        bot_b_commit_or_tag="abc",
        winner="draw",
        winner_player_id=-1,
        turns=100,
        terminated=False,
        truncated=True,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
        grid_size=24,
        truncation_limit=5000,
    )
    path = save_classic_game(record, tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["mode"] == "classic"
    assert ClassicGameRecord.from_dict(loaded).game_id == "test_game"
