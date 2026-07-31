"""Remote human-block counter (T2): opponent_is_bot filter."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from arena.remote_block import count_human_block_games, counts_as_human_block_game


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"counts_toward_block": True, "opponent_is_bot": False}, True),
        ({"counts_toward_block": True, "opponent_is_bot": True}, False),
        ({"counts_toward_block": True, "opponent_is_bot": None}, False),
        ({"counts_toward_block": False, "opponent_is_bot": False}, False),
        ({}, False),
    ],
    ids=["human", "bot", "null_opponent", "not_counted", "empty"],
)
def test_counts_as_human_block_game(record, expected):
    assert counts_as_human_block_game(record) is expected


def test_count_human_block_games_filters_and_mtime(tmp_path: Path):
    now = time.time()
    human = tmp_path / "human_win.json"
    bot = tmp_path / "bot_win.json"
    null_opp = tmp_path / "unknown_opp.json"
    old = tmp_path / "old_human.json"

    for path, payload in [
        (human, {"counts_toward_block": True, "opponent_is_bot": False}),
        (bot, {"counts_toward_block": True, "opponent_is_bot": True}),
        (null_opp, {"counts_toward_block": True, "opponent_is_bot": None}),
    ]:
        path.write_text(json.dumps(payload), encoding="utf-8")

    old.write_text(
        json.dumps({"counts_toward_block": True, "opponent_is_bot": False}),
        encoding="utf-8",
    )
    # Backdate so mtime filter excludes it.
    import os

    os.utime(old, (now - 3600, now - 3600))

    assert count_human_block_games(tmp_path) == 2
    assert count_human_block_games(tmp_path, since_mtime=now - 60) == 1
    assert count_human_block_games(tmp_path, since_mtime=now + 3600) == 0
