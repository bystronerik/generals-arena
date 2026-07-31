"""Tests for remote result fidelity (human-95-plan §5.4)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from generals_client.bot import BotError, GameResult

from arena.remote_bridge import ensure_bot_username, register_username_safe
from arena.remote_client import (
    DECIDED_REASONS,
    FidelityRemoteSession,
    opponent_is_bot,
    result_from_reason,
)


def test_ensure_bot_username_adds_prefix():
    assert ensure_bot_username("arena_army_convey") == "[Bot] arena_army_convey"
    assert ensure_bot_username("[Bot] arena_army_convey") == "[Bot] arena_army_convey"


def test_result_from_reason_mapping():
    assert result_from_reason("game_won") == "win"
    assert result_from_reason("game_lost") == "loss"
    assert result_from_reason("disconnect") == "disconnect"
    assert result_from_reason("receive_error") == "error"
    assert result_from_reason("stall") == "error"


def test_opponent_is_bot():
    assert opponent_is_bot("[Bot] foo") is True
    assert opponent_is_bot("human") is False
    assert opponent_is_bot(None) is None


def test_null_opponent_is_bot_does_not_count_toward_human_block():
    from arena.remote_block import counts_as_human_block_game

    record = {
        "counts_toward_block": True,
        "result": "win",
        "opponent_username": None,
        "opponent_is_bot": opponent_is_bot(None),
    }
    assert record["opponent_is_bot"] is None
    assert counts_as_human_block_game(record) is False


@pytest.fixture
def fidelity_session(tmp_path: Path):
    from arena.remote_bridge import UnifiedBot

    bot = UnifiedBot("smoke")
    session = FidelityRemoteSession(
        bot,
        "test-user-id",
        bot_name="smoke",
        room_mode="1v1",
        log_dir=tmp_path,
    )
    session.client = MagicMock()
    return session


def test_disconnect_is_not_a_win(fidelity_session, tmp_path: Path):
    fidelity_session.client.play_1v1 = MagicMock(
        side_effect=BotError("server disconnected before the game finished")
    )
    fidelity_session.play_1v1()

    logs = list(tmp_path.glob("*.json"))
    assert len(logs) == 1
    record = json.loads(logs[0].read_text())
    assert record["result"] == "disconnect"
    assert record["result_reason"] == "disconnect"
    assert record["counts_toward_block"] is False
    assert fidelity_session.wins == 0


def test_game_won_counts_toward_block(fidelity_session, tmp_path: Path):
    fidelity_session.client.play_1v1 = MagicMock(
        return_value=GameResult(won=True, replay_url="https://bot.generals.io/replays/abc", turns=10)
    )
    fidelity_session.play_1v1()

    assert fidelity_session.wins == 1
    record = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
    assert record["result"] == "win"
    assert record["result_reason"] == "game_won"
    assert record["counts_toward_block"] is True
    assert record["replay_id"] == "abc"


def test_game_lost_counts_toward_block(fidelity_session, tmp_path: Path):
    fidelity_session.client.play_1v1 = MagicMock(
        return_value=GameResult(won=False, replay_url="https://bot.generals.io/replays/xyz", turns=8)
    )
    fidelity_session.play_1v1()

    assert fidelity_session.losses == 1
    record = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
    assert record["result"] == "loss"
    assert record["counts_toward_block"] is True


def test_register_username_continues_when_username_already_bound():
    client = MagicMock()
    client.register_username = MagicMock(
        side_effect=BotError(
            "server rejected username '[Bot] BOBTHEAGENT': "
            "You already have a username! Only Supporters can change usernames."
        )
    )
    register_username_safe(client, "BOBTHEAGENT")


def test_decided_reasons_set():
    assert DECIDED_REASONS == frozenset({"game_won", "game_lost"})
