"""Tests for remote session runners and queue timeout detection."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from generals_client.bot import BotError, GameResult

from arena.remote_client import (
    is_queue_timeout,
    run_1v1_session,
    run_lobby_session,
)


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ("game did not finish within 600s; giving up", True),
        ("server disconnected before the game finished", False),
        (None, False),
    ],
)
def test_is_queue_timeout(detail, expected):
    assert is_queue_timeout(detail) is expected


def test_run_1v1_session_requeues_on_timeout():
    from arena.remote_bridge import UnifiedBot

    bot = UnifiedBot("smoke")
    session = MagicMock()
    session.bot_name = "smoke"
    session.room_mode = "1v1"
    session.endpoint = "botws.generals.io"
    session.wins = 0
    session.losses = 0
    session.client = MagicMock()
    session.client.game_timeout = 60.0
    session.register = MagicMock()

    calls = {"n": 0}

    def play_1v1():
        calls["n"] += 1
        if calls["n"] == 1:
            session.last_finish_detail = "game did not finish within 60s; giving up"
            return "receive_error"
        session.last_finish_detail = None
        return "game_won"

    session.play_1v1 = play_1v1

    with patch("arena.remote_client.time.sleep"):
        summary = run_1v1_session(
            session,
            "[Bot] test",
            max_games=1,
            queue_timeout_seconds=60.0,
        )

    assert summary.queue_timeouts == 1
    assert summary.games_played == 1
    assert summary.wins == 0
    assert session.register.called


def test_run_lobby_session_counts_games():
    session = MagicMock()
    session.bot_name = "smoke"
    session.room_mode = "lobby"
    session.endpoint = "botws.generals.io"
    session.wins = 1
    session.losses = 0
    session.register = MagicMock()
    session.play_private = MagicMock(return_value="game_won")
    session.last_finish_detail = None

    summary = run_lobby_session(session, "arena-test", "[Bot] test", max_games=2)
    assert summary.games_played == 2
    assert session.play_private.call_count == 2
