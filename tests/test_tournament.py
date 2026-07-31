"""Light tests for arena/tournament.py seed and pair helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from arena.tournament import bot_pairs, parse_seeds


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
