"""Tests for remote result fidelity (human-95-plan §5.4)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arena.remote_client import (
    DECIDED_REASONS,
    FidelityGeneralsIOClient,
    normalize_bot_endpoint_username,
    opponent_is_bot,
    result_from_reason,
)


def test_normalize_bot_endpoint_username_strips_prefix():
    assert normalize_bot_endpoint_username("[Bot] arena_army_convey") == "arena_army_convey"
    assert normalize_bot_endpoint_username("plain_name") == "plain_name"


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


class _FakeAgent:
    bot_name = "smoke"

    def session_stats(self):
        return {
            "builds_dropped": 0,
            "faults": 0,
            "timeouts": 0,
            "saw_enemy_general_at": None,
            "peak_land": 1,
            "peak_army": 5,
            "final_land": 1,
            "final_army": 5,
            "server_turns": 10,
        }

    def reset(self):
        pass


@pytest.fixture
def fidelity_client(tmp_path: Path):
    with patch.object(FidelityGeneralsIOClient, "__init__", lambda self, *a, **k: None):
        client = FidelityGeneralsIOClient.__new__(FidelityGeneralsIOClient)
        client.public_server = False
        client._replay_id = "testreplay"
        client._status = "game"
        client._score_wins = 0
        client._score_losses = 0
        client._arena_agent = _FakeAgent()
        client.bot_name = "smoke"
        client.room_mode = "1v1"
        client.log_dir = tmp_path
        client._opponent_username = "human_player"
        client.game_state = MagicMock(opponent_index=0, stars=[3])
        client.emit = MagicMock()
        return client


def test_malformed_receive_is_not_a_win(fidelity_client, tmp_path: Path):
    fidelity_client.receive = MagicMock(side_effect=ValueError("bad frame"))
    fidelity_client._play_game()

    assert fidelity_client._score_wins == 0
    assert fidelity_client._score_losses == 0
    logs = list(tmp_path.glob("*.json"))
    assert len(logs) == 1
    record = json.loads(logs[0].read_text())
    assert record["result"] == "disconnect"
    assert record["result_reason"] == "disconnect"
    assert record["counts_toward_block"] is False


def test_short_receive_tuple_is_not_a_win(fidelity_client, tmp_path: Path):
    fidelity_client.receive = MagicMock(return_value=("chat_message", {"text": "hi"}))
    fidelity_client._play_game()

    assert fidelity_client._score_wins == 0
    record = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
    assert record["result_reason"] == "receive_error"
    assert record["counts_toward_block"] is False


def test_game_won_counts_toward_block(fidelity_client, tmp_path: Path):
    fidelity_client.receive = MagicMock(return_value=("game_won", {}, None))
    fidelity_client._play_game()

    assert fidelity_client._score_wins == 1
    record = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
    assert record["result"] == "win"
    assert record["result_reason"] == "game_won"
    assert record["counts_toward_block"] is True


def test_game_lost_counts_toward_block(fidelity_client, tmp_path: Path):
    fidelity_client.receive = MagicMock(return_value=("game_lost", {}, None))
    fidelity_client._play_game()

    assert fidelity_client._score_losses == 1
    record = json.loads(list(tmp_path.glob("*.json"))[0].read_text())
    assert record["result"] == "loss"
    assert record["counts_toward_block"] is True


def test_decided_reasons_set():
    assert DECIDED_REASONS == frozenset({"game_won", "game_lost"})
