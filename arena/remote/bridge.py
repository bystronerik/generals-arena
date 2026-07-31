"""
Live generals.io wire bridge via ``generals_client``.

Wraps :class:`~generals_client.bot.GameClient` and :class:`~generals_client.bot.BaseBot`
so arena bots use the unified API without duplicating socket logic.
"""
from __future__ import annotations

from generals_client.bot import BaseBot, BotError, GameClient
from generals_client.state import GameState

from arena.bot_api import StrategySession, from_game_state, to_client_move


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
    """GameClient alias used by the arena remote harness (split moves via ``_emit_move``)."""


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
