"""Light tests for arena/tournament.py seed and pair helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.tournament import (
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
    a = tmp_path / "a" / "run.sh"
    b = tmp_path / "b" / "run.sh"
    for p in (a, b):
        p.parent.mkdir(parents=True)
        p.write_text("#!/bin/bash\n")
    pairs = bot_pairs([a, b], include_self=True)
    assert len(pairs) == 4


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
        swap_sides=False,
        fixed_seeds=None,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["round"] == "round1"
    assert data["round_seed"] == 7
    assert data["match_count"] == 2
    assert data["jobs"] == 4
    assert len(data["assignments"]) == 2
