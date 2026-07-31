"""Light tests for arena/tournaments/competition.py seed and pair helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.tournaments.competition import (
    ALTERNATING_SEATS,
    RANDOM_SEATS,
    _pair_rng,
    bot_pairs,
    expand_pair_seeds,
    parse_seeds,
    write_round_manifest,
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("0", [0]),
        ("0,1,2", [0, 1, 2]),
        ("0-3", [0, 1, 2, 3]),
        ("0-2,5,7-8", [0, 1, 2, 5, 7, 8]),
        ("2,0,2", [0, 2]),
    ],
    ids=["single", "list", "range", "mixed", "dedupe_sort"],
)
def test_parse_seeds(spec, expected):
    assert parse_seeds(spec) == expected


def test_parse_seeds_rejects_empty():
    with pytest.raises(ValueError, match="no seeds"):
        parse_seeds("")


def test_parse_seeds_rejects_inverted_range():
    with pytest.raises(ValueError, match="invalid seed range"):
        parse_seeds("5-2")


def test_bot_pairs_unique_unordered(tmp_path: Path):
    a = tmp_path / "a" / "run.sh"
    b = tmp_path / "b" / "run.sh"
    c = tmp_path / "c" / "run.sh"
    for p in (a, b, c):
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/bash\n")
    pairs = bot_pairs([a, b, c])
    assert len(pairs) == 3
    assert (a.resolve(), b.resolve()) in pairs
    assert (a.resolve(), c.resolve()) in pairs
    assert (b.resolve(), c.resolve()) in pairs


def test_bot_pairs_include_self(tmp_path: Path):
    """Unordered pairs plus one self-play pair each — not the ordered product."""
    a = tmp_path / "a" / "run.sh"
    b = tmp_path / "b" / "run.sh"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/bash\n")
    pairs = bot_pairs([a, b], include_self=True)
    assert len(pairs) == 3
    assert (a.resolve(), a.resolve()) in pairs
    assert (b.resolve(), b.resolve()) in pairs


def test_expand_pair_seeds_random_reproducible(tmp_path: Path):
    a = tmp_path / "smoke" / "run.sh"
    b = tmp_path / "rush" / "run.sh"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/bash\n")
    pairs = [(a.resolve(), b.resolve())]
    s1 = expand_pair_seeds(pairs, games_per_pair=5, round_seed=42)
    s2 = expand_pair_seeds(pairs, games_per_pair=5, round_seed=42)
    s3 = expand_pair_seeds(pairs, games_per_pair=5, round_seed=43)
    assert s1 == s2
    assert len(s1) == 5
    assert len({seed for _, _, seed in s1}) == 5
    assert {seed for _, _, seed in s1} != {seed for _, _, seed in s3}


def test_expand_pair_seeds_fixed_list(tmp_path: Path):
    a = tmp_path / "a" / "run.sh"
    b = tmp_path / "b" / "run.sh"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/bash\n")
    specs = expand_pair_seeds(
        [(a.resolve(), b.resolve())],
        games_per_pair=50,
        round_seed=0,
        fixed_seeds=[10, 20],
    )
    assert [(s[2]) for s in specs] == [10, 20]


def test_write_round_manifest(tmp_path: Path):
    a = tmp_path / "bots" / "smoke" / "run.sh"
    b = tmp_path / "bots" / "rush" / "run.sh"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/bash\n")
    games_dir = tmp_path / "round1"
    path = write_round_manifest(
        games_dir,
        round_name="round1",
        round_seed=7,
        games_per_pair=2,
        jobs=4,
        bots=["smoke", "rush"],
        specs=[(a.resolve(), b.resolve(), 11), (a.resolve(), b.resolve(), 22)],
        seat_policy=RANDOM_SEATS,
        fixed_seeds=None,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["round"] == "round1"
    assert data["round_seed"] == 7
    assert data["match_count"] == 2
    assert data["jobs"] == 4
    assert data["seat_policy"] == RANDOM_SEATS
    assert len(data["assignments"]) == 2


# --- T16: seat randomization, orientation-independent RNG, matched seeds ---


@pytest.fixture
def roster(tmp_path: Path):
    """Ten bot run.sh paths, named so roster order and id order disagree."""
    names = [f"bot{i:02d}" for i in range(10)]
    paths = []
    for name in names:
        p = tmp_path / name / "run.sh"
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/bash\n")
        paths.append(p.resolve())
    return paths


def test_pair_rng_is_orientation_independent():
    """The stream chooses the seat, so it must not depend on the orientation."""
    forward = _pair_rng(7, "aegis", "blitz")
    backward = _pair_rng(7, "blitz", "aegis")
    assert [forward.random() for _ in range(5)] == [backward.random() for _ in range(5)]


def test_both_orientations_draw_the_same_map_seeds(roster):
    """The old oriented key is why --swap-sides never actually mirrored."""
    a, b = roster[0], roster[1]
    forward = expand_pair_seeds([(a, b)], games_per_pair=5, round_seed=3)
    mirrored = expand_pair_seeds([(b, a)], games_per_pair=5, round_seed=3)
    assert {s for _, _, s in forward} == {s for _, _, s in mirrored}
    assert forward == mirrored  # including the seat each seed lands in


def test_random_seat_policy_costs_no_extra_games(roster):
    specs = expand_pair_seeds(bot_pairs(roster), games_per_pair=6, round_seed=1)
    assert len(specs) == 6 * len(bot_pairs(roster))


def test_random_seats_split_near_evenly_over_a_large_round(roster):
    """Balanced in expectation — which is why decision arms do not rely on it."""
    specs = expand_pair_seeds(bot_pairs(roster), games_per_pair=40, round_seed=11)
    first_seat = sum(1 for a, b, _ in specs if a.parent.name < b.parent.name)
    share = first_seat / len(specs)
    assert 0.45 < share < 0.55, share


def test_alternating_seats_are_exactly_fifty_fifty(roster):
    """Decision arms balance by construction, not by expectation."""
    specs = expand_pair_seeds(
        bot_pairs(roster), games_per_pair=6, round_seed=1, seat_policy=ALTERNATING_SEATS
    )
    first_seat = sum(1 for a, b, _ in specs if a.parent.name < b.parent.name)
    assert first_seat * 2 == len(specs)


def test_alternating_seats_play_every_map_seed_both_ways(roster):
    """Matched pairs: map difficulty cancels within each seed."""
    a, b = roster[0], roster[1]
    specs = expand_pair_seeds(
        [(a, b)], games_per_pair=6, round_seed=1, seat_policy=ALTERNATING_SEATS
    )
    assert len(specs) == 6
    by_seed: dict[int, set[str]] = {}
    for seat_a, _, seed in specs:
        by_seed.setdefault(seed, set()).add(seat_a.parent.name)
    assert len(by_seed) == 3
    assert all(sides == {"bot00", "bot01"} for sides in by_seed.values())


def test_alternating_seats_round_an_odd_count_up_to_stay_balanced(roster):
    specs = expand_pair_seeds(
        [(roster[0], roster[1])],
        games_per_pair=5,
        round_seed=1,
        seat_policy=ALTERNATING_SEATS,
    )
    assert len(specs) == 6


def test_self_play_is_one_game_per_seed_under_either_policy(roster):
    a = roster[0]
    for policy in (RANDOM_SEATS, ALTERNATING_SEATS):
        specs = expand_pair_seeds(
            [(a, a)], games_per_pair=4, round_seed=1, seat_policy=policy
        )
        assert len(specs) == 4, policy


def test_expand_pair_seeds_rejects_an_unknown_seat_policy(roster):
    with pytest.raises(ValueError, match="unknown seat policy"):
        expand_pair_seeds(
            [(roster[0], roster[1])],
            games_per_pair=2,
            round_seed=0,
            seat_policy="coin-flip",
        )


def test_alternating_seats_honour_a_fixed_seed_list(roster):
    """Every fixed seed still gets both orientations."""
    specs = expand_pair_seeds(
        [(roster[0], roster[1])],
        games_per_pair=50,
        round_seed=0,
        fixed_seeds=[10, 20],
        seat_policy=ALTERNATING_SEATS,
    )
    assert sorted(seed for _, _, seed in specs) == [10, 10, 20, 20]
    first_seat = sum(1 for a, b, _ in specs if a.parent.name < b.parent.name)
    assert first_seat * 2 == len(specs)
