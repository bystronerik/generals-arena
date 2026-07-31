"""Tests for arena.remote.report aggregation helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.remote.report import (
    aggregate_remote_games,
    format_markdown_report,
    load_remote_records,
    star_band,
    wilson_lower_bound,
)


@pytest.mark.parametrize(
    ("stars", "expected"),
    [
        (None, "unrated"),
        (0, "0-49"),
        (49, "0-49"),
        (50, "50-99"),
        (99, "50-99"),
        (100, "100-149"),
        (200, "150+"),
    ],
    ids=["unrated", "zero", "low", "mid_low", "mid_high", "high", "very_high"],
)
def test_star_band(stars, expected):
    assert star_band(stars) == expected


def test_wilson_lower_bound_empty():
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_lower_bound_95_of_100():
    lb = wilson_lower_bound(95, 100)
    assert 0.87 <= lb <= 0.89


def test_aggregate_human_games_only(tmp_path: Path):
    human_win = {
        "counts_toward_block": True,
        "opponent_is_bot": False,
        "result": "win",
        "opponent_stars": 60,
        "room_mode": "1v1",
        "bot_id": "classic_duel",
    }
    bot_opp = {
        "counts_toward_block": True,
        "opponent_is_bot": True,
        "result": "win",
    }
    null_opp = {
        "counts_toward_block": True,
        "opponent_is_bot": None,
        "result": "win",
    }
    human_loss = {
        "counts_toward_block": True,
        "opponent_is_bot": False,
        "result": "loss",
        "replay_id": "abc123",
        "opponent_username": "human1",
        "opponent_stars": None,
        "room_mode": "lobby",
    }
    for name, payload in [
        ("a.json", human_win),
        ("b.json", bot_opp),
        ("c.json", null_opp),
        ("d.json", human_loss),
    ]:
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    records = load_remote_records(tmp_path)
    stats = aggregate_remote_games(records)
    assert stats.total_files == 4
    assert stats.counted_human_games == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.bot_opponent_games == 1
    assert stats.lobby_games == 1
    assert stats.queue_games == 1
    assert stats.loss_replays == [("abc123", "human1")]

    report = format_markdown_report(stats, log_dir=tmp_path)
    assert "Wilson 95% lower bound" in report
    assert "50-99" in report
