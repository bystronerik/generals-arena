"""
Bridge competition-module remote observations to unified bot strategies.

``StdioStrategyAdapter`` wraps ``bots/<name>/agent.py`` for legacy in-process
play through ``generals.agents.Agent``. Live remote play uses
``arena.remote_bridge.UnifiedBot`` instead.

See ``docs/engine/remote-eval-heuristics.md`` and ``docs/engine/unified-bot-api.md``.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from arena.bot_api import (
    PASS,
    StrategySession,
    from_competition_remote_obs,
    list_bots,
    to_competition_action_array,
    translate_action_for_remote,
)

if TYPE_CHECKING:
    from generals.agents import Agent as RemoteAgent
    from generals.core.observation import Observation as RemoteObservation

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

REMOTE_RECOMMENDED_BOTS: tuple[str, ...] = (
    "classic_duel",
    "army_convey",
    "late_rush",
    "fog_scout",
    "expand_plus",
)

REMOTE_BUILD_BOTS: frozenset[str] = frozenset({"castle_builder", "castle_rush", "phase_switch"})

# Backward-compatible alias kept for scripts/remote_play.py.
list_remote_bots = list_bots

_AdapterClass: type[Any] | None = None


def _adapter_class() -> type[Any]:
    """Lazy import of competition-module Agent (JAX path not needed for live play)."""
    global _AdapterClass
    if _AdapterClass is not None:
        return _AdapterClass

    from generals.agents import Agent as RemoteAgent
    from generals.core.observation import Observation as RemoteObservation

    class StdioStrategyAdapter(RemoteAgent):
        """Wrap ``bots/<name>/agent.py`` for legacy competition-module remote play."""

        def __init__(self, bot_name: str, bot_id: str | None = None):
            super().__init__(id=bot_id or bot_name)
            self.bot_name = bot_name
            self.session = StrategySession(bot_name)
            self.strategy = None
            self.builds_dropped = 0
            self.faults = 0
            self.timeouts = 0
            self.saw_enemy_general_at: int | None = None
            self._peak_land = 0
            self._peak_army = 0
            self._last_land = 0
            self._last_army = 0
            self._last_turn = 0

        def act(self, observation: RemoteObservation, key=None) -> np.ndarray:
            """Remote client calls ``act(observation)`` with one argument; key is optional."""
            action = self.session.act(from_competition_remote_obs(observation))
            self._sync_from_session()
            return to_competition_action_array(action)

        def _sync_from_session(self) -> None:
            stats = self.session.session_stats()
            self.builds_dropped = stats["builds_dropped"]
            self.faults = stats["faults"]
            self.timeouts = stats["timeouts"]
            self.saw_enemy_general_at = stats["saw_enemy_general_at"]
            self._peak_land = stats["peak_land"]
            self._peak_army = stats["peak_army"]
            self._last_land = stats["final_land"]
            self._last_army = stats["final_army"]
            self._last_turn = stats["server_turns"]
            self.strategy = self.session.strategy

        def session_stats(self) -> dict:
            self._sync_from_session()
            return self.session.session_stats()

        def reset(self) -> None:
            self.session.reset()
            self.strategy = None
            self.builds_dropped = 0
            self.faults = 0
            self.timeouts = 0
            self.saw_enemy_general_at = None
            self._peak_land = 0
            self._peak_army = 0
            self._last_land = 0
            self._last_army = 0
            self._last_turn = 0

    _AdapterClass = StdioStrategyAdapter
    return _AdapterClass


def __getattr__(name: str) -> Any:
    if name == "StdioStrategyAdapter":
        return _adapter_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_remote_agent(bot_name: str) -> Any:
    adapter_cls = _adapter_class()
    if bot_name in REMOTE_BUILD_BOTS:
        print(
            f"Warning: {bot_name} uses build actions that generals.io ignores remotely. "
            "Prefer army_convey, late_rush, fog_scout, or expand_plus.",
            file=sys.stderr,
        )
    return adapter_cls(bot_name=bot_name)


def verify_adapter_offline() -> list[str]:
    """
    Offline checks required before any live run. Returns error strings;
    empty list means all checks passed.
    """
    from generals.core.observation import Observation as RemoteObservation

    errors: list[str] = []
    adapter_cls = _adapter_class()

    H, W = 3, 3
    armies = np.zeros((H, W), dtype=np.int32)
    armies[1, 1] = 5
    generals = np.zeros((H, W), dtype=bool)
    generals[1, 1] = True
    castles = np.zeros((H, W), dtype=bool)
    mountains = np.zeros((H, W), dtype=bool)
    mountains[0, 1] = True
    neutral = np.zeros((H, W), dtype=bool)
    neutral[1, 0] = True
    owned = np.zeros((H, W), dtype=bool)
    owned[1, 1] = True
    opponent = np.zeros((H, W), dtype=bool)
    fog = np.zeros((H, W), dtype=bool)
    fog[2, 2] = True
    structures_in_fog = np.zeros((H, W), dtype=bool)

    remote_obs = RemoteObservation(
        armies=armies,
        generals=generals,
        castles=castles,
        mountains=mountains,
        neutral_cells=neutral,
        owned_cells=owned,
        opponent_cells=opponent,
        fog_cells=fog,
        structures_in_fog=structures_in_fog,
        owned_land_count=1,
        owned_army_count=5,
        opponent_land_count=0,
        opponent_army_count=0,
        timestep=7,
    )
    stdio = from_competition_remote_obs(remote_obs)
    if stdio.H != 3 or stdio.W != 3 or stdio.turn != 7:
        errors.append("scalar mapping failed on synthetic observation")
    if stdio.type_grid[0][1] != 2:
        errors.append("mountain type_grid mapping failed")
    if stdio.type_grid[1][1] != 4:
        errors.append("general type_grid mapping failed")
    if stdio.type_grid[2][2] != 0:
        errors.append("fog type_grid mapping failed")
    if stdio.owner_grid[1][1] != 1:
        errors.append("owned owner_grid mapping failed")
    if stdio.army_grid[1][1] != 5:
        errors.append("army_grid mapping failed")

    rewritten = translate_action_for_remote((2, 1, 2, 0, 0))
    if rewritten != PASS:
        errors.append("build action was not rewritten to pass")

    adapter = adapter_cls("smoke")
    action = adapter.act(remote_obs)
    if int(action[0]) not in (0, 1) or len(action) != 5:
        errors.append("smoke adapter did not return a valid action array")

    for bot in REMOTE_RECOMMENDED_BOTS:
        try:
            agent = adapter_cls(bot)
            out = agent.act(remote_obs)
            if len(out) != 5:
                errors.append(f"{bot}: action length != 5")
        except Exception as exc:
            errors.append(f"{bot}: import/act failed: {exc}")

    return errors
