"""
Bridge generals.core.observation.Observation to stdio-bot agent logic for live play.

See docs/engine/remote-eval-heuristics.md and docs/engine/remote-play-setup.md.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Type

import numpy as np

from generals.agents import Agent as RemoteAgent
from generals.core.observation import Observation as RemoteObservation

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top round-1 heuristics plus expand_plus and the classic champion.
REMOTE_RECOMMENDED_BOTS: tuple[str, ...] = (
    "classic_duel",
    "army_convey",
    "late_rush",
    "fog_scout",
    "expand_plus",
)

# Build-economy bots silently pass every build tick on generals.io.
REMOTE_BUILD_BOTS: frozenset[str] = frozenset({"castle_builder", "castle_rush", "phase_switch"})


@dataclass
class StdioObservation:
    """Plain observation shape used by bots/<name>/agent.py (matches main.py)."""

    H: int
    W: int
    turn: int
    my_land: int
    my_army: int
    opp_land: int
    opp_army: int
    type_grid: List[List[int]]
    owner_grid: List[List[int]]
    army_grid: List[List[int]]


def list_remote_bots() -> list[str]:
    """All bots with agent.py under bots/."""
    bots_dir = REPO_ROOT / "bots"
    return sorted(
        p.name
        for p in bots_dir.iterdir()
        if p.is_dir() and (p / "agent.py").is_file()
    )


def load_strategy_class(bot_name: str) -> Type:
    """Import bots/<name>/agent.py Agent class without copying strategy code."""
    bot_dir = REPO_ROOT / "bots" / bot_name
    agent_path = bot_dir / "agent.py"
    if not agent_path.is_file():
        known = ", ".join(list_remote_bots())
        raise ValueError(f"Unknown bot {bot_name!r}. Known bots: {known}")

    module_name = f"_arena_remote_{bot_name}_agent"
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {agent_path}")

    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(bot_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        if str(bot_dir) in sys.path:
            sys.path.remove(str(bot_dir))

    agent_cls = getattr(module, "Agent", None)
    if agent_cls is None:
        raise ImportError(f"{agent_path} has no Agent class")
    return agent_cls


def to_stdio_observation(obs: RemoteObservation) -> StdioObservation:
    """Map remote NamedTuple observation to the stdio dataclass our bots expect."""
    armies = np.asarray(obs.armies)
    H, W = armies.shape

    turn = int(obs.timestep)
    my_land = int(obs.owned_land_count)
    my_army = int(obs.owned_army_count)
    opp_land = int(obs.opponent_land_count)
    opp_army = int(obs.opponent_army_count)

    fog = np.asarray(obs.fog_cells, dtype=bool)
    mountains = np.asarray(obs.mountains, dtype=bool)
    castles = np.asarray(obs.castles, dtype=bool)
    generals = np.asarray(obs.generals, dtype=bool)
    structures_in_fog = np.asarray(obs.structures_in_fog, dtype=bool)
    owned = np.asarray(obs.owned_cells, dtype=bool)
    opponent = np.asarray(obs.opponent_cells, dtype=bool)

    type_grid: List[List[int]] = []
    owner_grid: List[List[int]] = []
    army_grid: List[List[int]] = []

    for r in range(H):
        type_row: List[int] = []
        owner_row: List[int] = []
        army_row: List[int] = []
        for c in range(W):
            army_row.append(int(armies[r, c]))
            if fog[r, c]:
                cell_type = 0
            elif mountains[r, c]:
                cell_type = 2
            elif castles[r, c]:
                cell_type = 3
            elif generals[r, c]:
                cell_type = 4
            elif structures_in_fog[r, c]:
                cell_type = 5
            else:
                cell_type = 1
            type_row.append(cell_type)
            if owned[r, c]:
                owner_row.append(1)
            elif opponent[r, c]:
                owner_row.append(2)
            else:
                owner_row.append(0)
        type_grid.append(type_row)
        owner_grid.append(owner_row)
        army_grid.append(army_row)

    return StdioObservation(
        H=H,
        W=W,
        turn=turn,
        my_land=my_land,
        my_army=my_army,
        opp_land=opp_land,
        opp_army=opp_army,
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )


def translate_action(action: Sequence[int]) -> tuple[int, ...]:
    """Rewrite unsupported build actions to pass; return a 5-tuple."""
    p, r, c, d, s = (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
    if p == 2:
        return (1, 0, 0, 0, 0)
    return (p, r, c, d, s)


class StdioStrategyAdapter(RemoteAgent):
    """Wrap bots/<name>/agent.py for in-process generals.io play."""

    def __init__(self, bot_name: str, bot_id: str | None = None):
        super().__init__(id=bot_id or bot_name)
        self.bot_name = bot_name
        self._strategy_class = load_strategy_class(bot_name)
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
        """Remote client calls act(observation) with one argument; key is optional."""
        try:
            stdio_obs = to_stdio_observation(observation)
            self._track_scalars(stdio_obs)
            self._track_enemy_general(stdio_obs)

            if self.strategy is None:
                self.strategy = self._strategy_class(player_id=0, H=stdio_obs.H, W=stdio_obs.W)

            raw = self.strategy.act(stdio_obs)
            action = translate_action(raw)
            if int(raw[0]) == 2:
                self.builds_dropped += 1
            return np.array(action, dtype=int)
        except Exception:
            self.faults += 1
            return np.array([1, 0, 0, 0, 0], dtype=int)

    def _track_scalars(self, obs: StdioObservation) -> None:
        self._last_turn = obs.turn
        self._last_land = obs.my_land
        self._last_army = obs.my_army
        self._peak_land = max(self._peak_land, obs.my_land)
        self._peak_army = max(self._peak_army, obs.my_army)

    def _track_enemy_general(self, obs: StdioObservation) -> None:
        if self.saw_enemy_general_at is not None:
            return
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.type_grid[r][c] == 4 and obs.owner_grid[r][c] == 2:
                    self.saw_enemy_general_at = obs.turn
                    return

    def session_stats(self) -> dict:
        return {
            "builds_dropped": self.builds_dropped,
            "faults": self.faults,
            "timeouts": self.timeouts,
            "saw_enemy_general_at": self.saw_enemy_general_at,
            "peak_land": self._peak_land,
            "peak_army": self._peak_army,
            "final_land": self._last_land,
            "final_army": self._last_army,
            "server_turns": self._last_turn,
        }

    def reset(self) -> None:
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


def make_remote_agent(bot_name: str) -> StdioStrategyAdapter:
    if bot_name in REMOTE_BUILD_BOTS:
        print(
            f"Warning: {bot_name} uses build actions that generals.io ignores remotely. "
            "Prefer army_convey, late_rush, fog_scout, or expand_plus.",
            file=sys.stderr,
        )
    return StdioStrategyAdapter(bot_name=bot_name)


def verify_adapter_offline() -> list[str]:
    """
    Offline checks required before any live run. Returns a list of error strings;
    empty list means all checks passed.
    """
    errors: list[str] = []

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
    stdio = to_stdio_observation(remote_obs)
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

    rewritten = translate_action((2, 1, 2, 0, 0))
    if rewritten != (1, 0, 0, 0, 0):
        errors.append("build action was not rewritten to pass")

    adapter = StdioStrategyAdapter("smoke")
    action = adapter.act(remote_obs)
    if int(action[0]) not in (0, 1) or len(action) != 5:
        errors.append("smoke adapter did not return a valid action array")

    for bot in REMOTE_RECOMMENDED_BOTS:
        try:
            agent = StdioStrategyAdapter(bot)
            out = agent.act(remote_obs)
            if len(out) != 5:
                errors.append(f"{bot}: action length != 5")
        except Exception as exc:
            errors.append(f"{bot}: import/act failed: {exc}")

    return errors
