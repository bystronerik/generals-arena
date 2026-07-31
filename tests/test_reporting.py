"""Tests for arena/records/reporting.py: shared aggregation and table rendering."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from arena.records.reporting import (
    EXPANDER_PYTHON,
    aggregate_stats,
    bot_run_sh,
    matchup_table,
    winner_bot_id,
    winrate_table_lines,
)


def game(bot_a: str, bot_b: str, winner: str, turns: int):
    return SimpleNamespace(bot_a=bot_a, bot_b=bot_b, winner=winner, turns=turns)


@pytest.mark.parametrize(
    ("winner", "expected"),
    [("a", "blitz"), ("b", "smoke"), ("draw", "draw")],
    ids=["seat_a", "seat_b", "draw"],
)
def test_winner_bot_id(winner, expected):
    assert winner_bot_id(game("blitz", "smoke", winner, 100)) == expected


def test_aggregate_stats_counts_both_seats():
    stats = aggregate_stats(
        [
            game("blitz", "smoke", "a", 300),
            game("smoke", "blitz", "b", 400),
            game("blitz", "smoke", "draw", 1200),
        ]
    )
    assert stats["total_games"] == 3
    assert stats["draw_rate"] == pytest.approx(0.333, abs=0.001)
    by_bot = {row["bot_id"]: row for row in stats["by_bot"]}
    assert by_bot["blitz"] == {
        "bot_id": "blitz",
        "games": 3,
        "wins": 2,
        "losses": 0,
        "draws": 1,
        "winrate": pytest.approx(0.667, abs=0.001),
        "draw_rate": pytest.approx(0.333, abs=0.001),
        "mean_turns": pytest.approx(633.3, abs=0.1),
    }
    assert by_bot["smoke"]["wins"] == 0
    assert by_bot["smoke"]["losses"] == 2
    # sorted by winrate descending
    assert [row["bot_id"] for row in stats["by_bot"]] == ["blitz", "smoke"]


def test_aggregate_stats_empty():
    stats = aggregate_stats([])
    assert stats == {"total_games": 0, "draw_rate": 0.0, "mean_turns": 0.0, "by_bot": []}


def test_matchup_table_groups_unordered_pairs():
    rows = matchup_table(
        [
            game("blitz", "smoke", "a", 300),
            game("smoke", "blitz", "a", 400),  # smoke wins from seat a
            game("blitz", "smoke", "draw", 1200),
        ]
    )
    assert len(rows) == 1
    row = rows[0]
    assert (row["bot_a"], row["bot_b"]) == ("blitz", "smoke")
    assert row["games"] == 3
    assert row["wins_a"] == 1  # blitz
    assert row["wins_b"] == 1  # smoke
    assert row["draws"] == 1


def test_winrate_table_lines_renders_header_and_percentages():
    stats = aggregate_stats([game("blitz", "smoke", "a", 300)])
    lines = winrate_table_lines(stats["by_bot"])
    assert lines[0].startswith("| Bot | Games |")
    assert lines[1].startswith("| --- |")
    assert "| `blitz` | 1 | 1 | 0 | 0 | 100.0% | 0.0% | 300.0 |" in lines


@pytest.mark.parametrize(
    ("ref", "expected_suffix"),
    [
        ("blitz", "bots/blitz/run.sh"),
        ("bots/blitz", "bots/blitz/run.sh"),
        ("bots/blitz/run.sh", "bots/blitz/run.sh"),
    ],
    ids=["name", "directory", "sh_path"],
)
def test_bot_run_sh_accepts_names_dirs_and_paths(ref, expected_suffix):
    assert str(bot_run_sh(ref)).endswith(expected_suffix)


def test_bot_run_sh_special_cases_expander_python():
    """expander_python lives in competition-module, not bots/."""
    assert bot_run_sh("expander_python") == EXPANDER_PYTHON
