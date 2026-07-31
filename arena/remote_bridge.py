"""
Live generals.io wire bridge via ``generals_client``.

Wraps :class:`~generals_client.bot.GameClient` and :class:`~generals_client.bot.BaseBot`
so arena bots use the unified API without duplicating socket logic.
"""
from __future__ import annotations

import logging
from typing import Any

from generals_client.bot import BaseBot, BotError, GameClient
from generals_client.state import GameState
from generals_client.transport import DEFAULT_SERVER

from arena.bot_api import StrategySession, from_game_state, to_client_move

logger = logging.getLogger(__name__)


class UnifiedBot(BaseBot):
    """``generals_client`` bot that runs ``bots/<name>/agent.py`` strategy code."""

    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self.session = StrategySession(bot_name)
        self.last_state: GameState | None = None

    def on_game_start(self, state: GameState) -> None:
        self.session.reset()
        self.last_state = state

    def move(self, state: GameState) -> tuple[int, int] | tuple[int, int, bool] | None:
        self.last_state = state
        obs = from_game_state(state)
        action = self.session.act(obs)
        return to_client_move(action, state)


class ArenaGameClient(GameClient):
    """GameClient that forwards split moves and notifies the bot on game start."""

    def _on_game_start(self, data: dict[str, Any], *_: Any) -> None:
        super()._on_game_start(data, *_)
        bot = self._bot
        if isinstance(bot, UnifiedBot) and self._state is not None:
            bot.on_game_start(self._state)

    def _on_game_update(self, data: dict[str, Any], *_: Any) -> None:
        """Apply update, ask bot, emit attack with optional 50/50 split."""
        with self._lock:
            if self._finished.is_set():
                return
            state = self._state
            if state is None:
                logger.warning("game_update received before game_start; ignoring")
                return
            try:
                state.apply_update(data)
            except Exception:
                self._fail(BotError("failed to apply game_update; state is unreliable"))
                logger.exception("apply_update failed")
                return

            try:
                move = self._bot.move(state)
            except Exception:
                logger.exception("bot.move() raised on turn %d; passing this tick", state.turn)
                return

            if move is None:
                return

            validated = self._coerce_move(move, state)
            if validated is None:
                return
            start, end, is50 = validated
            self._transport.attack(start, end, is50)

    @staticmethod
    def _coerce_move(
        move: Any,
        state: GameState,
    ) -> tuple[int, int, bool] | None:
        is50 = False
        try:
            if len(move) == 3:
                start, end, is50 = move
                is50 = bool(is50)
            else:
                start, end = move
        except (TypeError, ValueError):
            logger.error("bot returned malformed move %r; passing", move)
            return None
        start, end = int(start), int(end)
        if not (0 <= start < state.size and 0 <= end < state.size):
            logger.error(
                "bot returned out-of-range move %d -> %d (map has %d tiles); passing",
                start,
                end,
                state.size,
            )
            return None
        return start, end, is50


def opponent_username(state: GameState | None) -> str | None:
    if state is None or not state.usernames:
        return None
    if len(state.usernames) == 2:
        opp = 1 - state.player_index
        if 0 <= opp < len(state.usernames):
            return state.usernames[opp]
    for i, name in enumerate(state.usernames):
        if i != state.player_index:
            return name
    return None


def opponent_stars(state: GameState | None) -> int | None:
    if state is None or not state.stars:
        return None
    if len(state.usernames) == 2:
        opp = 1 - state.player_index
        if 0 <= opp < len(state.stars):
            return int(state.stars[opp])
    return None


def ensure_bot_username(username: str) -> str:
    """
    generals_client requires usernames to start with ``[Bot]``.

    Adds the prefix when missing; leaves an existing prefix unchanged.
    """
    name = username.strip()
    if not name.startswith("[Bot]"):
        name = f"[Bot] {name}"
    return name


def register_username_safe(client: GameClient, username: str) -> None:
    """Register username; continue when the user id is already bound."""
    normalized = ensure_bot_username(username)
    if normalized != username.strip():
        print(
            f"Note: generals_client requires a [Bot] prefix (registering as {normalized!r}).",
        )
    try:
        client.register_username(normalized)
    except BotError as exc:
        message = str(exc)
        if "already have a username" in message.lower():
            print(
                "Note: user id already has a bound username on generals.io; "
                "continuing with existing registration.",
            )
            return
        raise


def make_unified_bot(bot_name: str) -> UnifiedBot:
    return UnifiedBot(bot_name=bot_name)


def make_remote_client(bot_name: str, user_id: str, server_url: str = DEFAULT_SERVER) -> ArenaGameClient:
    return ArenaGameClient(make_unified_bot(bot_name), user_id, server_url=server_url)
