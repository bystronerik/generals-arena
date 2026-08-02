"""
Unified observation and action types for arena bots.

Strategy code in ``bots/<name>/agent.py`` uses :class:`UnifiedObservation` and
returns a five-int :data:`UnifiedAction`. Stdio competition play and live
generals.io both map into this shape through wire bridges — bots do not import
competition-module or generals_client directly.

See ``docs/engine/unified-bot-api.md``.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol, Sequence, Type, runtime_checkable

import numpy as np

from arena.paths import REPO_ROOT

logger = logging.getLogger(__name__)

# Action tuple: (pass_flag, row, col, direction, split)
# pass_flag: 0 = move, 1 = pass, 2 = build (competition only; rewritten for remote)
UnifiedAction = tuple[int, int, int, int, int]

PASS: UnifiedAction = (1, 0, 0, 0, 0)

# (dr, dc) for direction codes 0..3 — matches competition protocol and generals.io
DIRECTIONS: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class UnifiedObservation:
    """Plain observation shape shared by stdio bots and both wire bridges."""

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


@runtime_checkable
class ArenaAgent(Protocol):
    """Contract for ``bots/<name>/agent.py`` strategy classes."""

    def __init__(self, player_id: int, H: int, W: int) -> None: ...

    def act(self, obs: UnifiedObservation) -> UnifiedAction: ...


def list_bots() -> list[str]:
    """All bots with ``agent.py`` under ``bots/``."""
    bots_dir = REPO_ROOT / "bots"
    return sorted(
        p.name
        for p in bots_dir.iterdir()
        if p.is_dir() and (p / "agent.py").is_file()
    )


def _module_locations(module) -> list[Path]:
    file = getattr(module, "__file__", None)
    if file:
        return [Path(file)]
    # Namespace packages (e.g. ``yankee`` imported as ``yankee.search``) have
    # no __file__, only __path__.
    return [Path(p) for p in list(getattr(module, "__path__", None) or [])]


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _sibling_module_names(bot_name: str, bot_dir: Path) -> set[str]:
    """Top-level module names a bot's plain imports can bind to."""
    names = {bot_name}  # the bot dir itself, as a namespace package
    for p in bot_dir.iterdir():
        if p.name in ("agent.py", "__init__.py"):
            continue
        if p.suffix == ".py":
            names.add(p.stem)
        elif p.is_dir() and not p.name.startswith(("_", ".")):
            names.add(p.name)
    return names


def _guard_sibling_collisions(bot_name: str, bot_dir: Path) -> None:
    """Refuse to exec a bot whose sibling imports would hit a foreign module.

    ``import blitz_core`` inside a bot is satisfied from ``sys.modules`` before
    ``sys.path`` is consulted, so a same-named module loaded from anywhere else
    would silently alias this bot's code to the other implementation.
    """
    for name in _sibling_module_names(bot_name, bot_dir):
        existing = sys.modules.get(name)
        if existing is None:
            continue
        locations = _module_locations(existing)
        if any(_is_under(loc, bot_dir) for loc in locations):
            continue
        origin = ", ".join(str(loc) for loc in locations) or "<no file>"
        raise ImportError(
            f"Cannot load bot {bot_name!r}: sys.modules[{name!r}] is already "
            f"loaded from {origin}, which is outside {bot_dir}. The bot's "
            f"'import {name}' would silently reuse that module instead of "
            f"{bot_dir / name}. Rename the sibling module or import it as "
            f"'{bot_name}.{name}' (as bots/yankee does)."
        )


def _purge_private_modules(before: set[str], bots_dir: Path) -> None:
    """Drop bot-private modules registered during one bot's exec.

    Sibling modules under ``bots/<name>/`` stay reachable through the agent
    module's own references, but must not linger in ``sys.modules`` where the
    next bot's identically named imports would pick them up. Shared code under
    ``bots/_common/`` is deliberately kept.
    """
    common_dir = bots_dir / "_common"
    for name in set(sys.modules) - before:
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = _module_locations(module)
        if any(
            _is_under(loc, bots_dir) and not _is_under(loc, common_dir)
            for loc in locations
        ):
            del sys.modules[name]


def load_strategy_class(bot_name: str) -> Type:
    """Import ``bots/<name>/agent.py`` ``Agent`` class without copying strategy code.

    Each load is self-contained: bot-private sibling modules (``params``,
    ``search``, …) are removed from ``sys.modules`` afterwards, so two bots
    that ship identically named siblings each get their own implementation.
    """
    bot_dir = REPO_ROOT / "bots" / bot_name
    agent_path = bot_dir / "agent.py"
    if not agent_path.is_file():
        known = ", ".join(list_bots())
        raise ValueError(f"Unknown bot {bot_name!r}. Known bots: {known}")

    _guard_sibling_collisions(bot_name, bot_dir)

    module_name = f"_arena_bot_{bot_name}_agent"
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {agent_path}")

    module = importlib.util.module_from_spec(spec)
    bots_dir = REPO_ROOT / "bots"
    before = set(sys.modules)
    # Register before exec (importlib recipe) — dataclasses and other
    # introspection in the agent module need sys.modules[module.__module__].
    sys.modules[module_name] = module
    sys.path.insert(0, str(bots_dir))
    sys.path.insert(0, str(bot_dir))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if str(bot_dir) in sys.path:
            sys.path.remove(str(bot_dir))
        if str(bots_dir) in sys.path:
            sys.path.remove(str(bots_dir))
        _purge_private_modules(before | {module_name}, bots_dir)

    agent_cls = getattr(module, "Agent", None)
    if agent_cls is None:
        raise ImportError(f"{agent_path} has no Agent class")
    return agent_cls


def translate_action_for_remote(action: Sequence[int]) -> UnifiedAction:
    """Rewrite unsupported build actions to pass; return a five-int tuple."""
    p, r, c, d, s = (int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
    if p == 2:
        return PASS
    return (p, r, c, d, s)


def to_competition_action_array(action: UnifiedAction) -> np.ndarray:
    """Encode a unified action for competition-module remote ``Agent`` callers."""
    return np.array(action, dtype=int)


def from_competition_remote_obs(obs) -> UnifiedObservation:
    """Map ``generals.core.observation.Observation`` (remote NamedTuple) to unified obs."""
    armies = np.asarray(obs.armies)
    H, W = armies.shape

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

    return UnifiedObservation(
        H=H,
        W=W,
        turn=int(obs.timestep),
        my_land=int(obs.owned_land_count),
        my_army=int(obs.owned_army_count),
        opp_land=int(obs.opponent_land_count),
        opp_army=int(obs.opponent_army_count),
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )


def from_game_state(state) -> UnifiedObservation:
    """Map ``generals_client.state.GameState`` to unified observation."""
    from generals_client.state import (
        TILE_FOG,
        TILE_FOG_OBSTACLE,
        TILE_MOUNTAIN,
    )

    H = state.height
    W = state.width
    if H <= 0 or W <= 0:
        return UnifiedObservation(
            H=max(H, 0),
            W=max(W, 0),
            turn=int(state.turn),
            my_land=0,
            my_army=0,
            opp_land=0,
            opp_army=0,
            type_grid=[],
            owner_grid=[],
            army_grid=[],
        )

    player = state.player_index
    opp_idx = _opponent_index(state)
    my_land, my_army = _score_for_player(state, player)
    opp_land, opp_army = _score_for_player(state, opp_idx) if opp_idx is not None else (0, 0)

    general_tiles = _general_tile_set(state)
    city_set = set(state.cities)

    type_grid: List[List[int]] = []
    owner_grid: List[List[int]] = []
    army_grid: List[List[int]] = []

    for r in range(H):
        type_row: List[int] = []
        owner_row: List[int] = []
        army_row: List[int] = []
        for c in range(W):
            idx = r * W + c
            terrain = state.terrain[idx] if idx < len(state.terrain) else TILE_FOG
            army = int(state.armies[idx]) if idx < len(state.armies) else 0

            if terrain == TILE_FOG:
                cell_type = 0
                army = 0
            elif terrain == TILE_FOG_OBSTACLE:
                cell_type = 5
                army = 0
            elif terrain == TILE_MOUNTAIN:
                cell_type = 2
            elif idx in city_set:
                cell_type = 3
            elif idx in general_tiles:
                cell_type = 4
            else:
                cell_type = 1

            if terrain == TILE_FOG or terrain == TILE_FOG_OBSTACLE:
                owner = 0
            elif terrain == player:
                owner = 1
            elif terrain >= 0:
                owner = 2
            else:
                owner = 0

            type_row.append(cell_type)
            owner_row.append(owner)
            army_row.append(army)
        type_grid.append(type_row)
        owner_grid.append(owner_row)
        army_grid.append(army_row)

    return UnifiedObservation(
        H=H,
        W=W,
        turn=int(state.turn),
        my_land=my_land,
        my_army=my_army,
        opp_land=opp_land,
        opp_army=opp_army,
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )


def to_client_move(action: UnifiedAction, state) -> tuple[int, int] | tuple[int, int, bool] | None:
    """
    Convert a unified action to generals_client tile indices.

    Returns ``None`` to pass. When split is set, returns ``(start, end, is50)``.
    """
    remote = translate_action_for_remote(action)
    p, r, c, d, split = remote
    if p != 0:
        return None
    if d < 0 or d >= len(DIRECTIONS):
        return None
    dr, dc = DIRECTIONS[d]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < state.height and 0 <= nc < state.width):
        return None
    start = r * state.width + c
    end = nr * state.width + nc
    if split:
        return (start, end, True)
    return (start, end)


def _opponent_index(state) -> int | None:
    if len(state.usernames) == 2:
        return 1 - int(state.player_index)
    for i in range(len(state.usernames)):
        if i != state.player_index:
            return i
    return None


def _score_for_player(state, player_index: int | None) -> tuple[int, int]:
    if player_index is None:
        return 0, 0
    for entry in state.scores:
        if int(entry.get("i", -1)) == player_index:
            return int(entry.get("tiles", 0)), int(entry.get("total", 0))
    return 0, 0


def _general_tile_set(state) -> set[int]:
    tiles: set[int] = set()
    for g in state.generals:
        if int(g) >= 0:
            tiles.add(int(g))
    for tile in state.known_generals.values():
        if int(tile) >= 0:
            tiles.add(int(tile))
    return tiles


class StrategySession:
    """Load one ``bots/<name>/agent.py`` strategy and track session stats."""

    def __init__(self, bot_name: str):
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
        self._first_fault_traceback: str | None = None

    def act(self, obs: UnifiedObservation) -> UnifiedAction:
        """Run strategy and rewrite build actions for remote play."""
        try:
            self._track_scalars(obs)
            self._track_enemy_general(obs)

            if self.strategy is None:
                self.strategy = self._strategy_class(player_id=0, H=obs.H, W=obs.W)

            raw = self.strategy.act(obs)
            action = translate_action_for_remote(raw)
            if int(raw[0]) == 2:
                self.builds_dropped += 1
            return action
        except Exception:
            self.faults += 1
            if self._first_fault_traceback is None:
                self._first_fault_traceback = traceback.format_exc()
                logger.exception("StrategySession fault in %s", self.bot_name)
            return PASS

    def _track_scalars(self, obs: UnifiedObservation) -> None:
        self._last_turn = obs.turn
        self._last_land = obs.my_land
        self._last_army = obs.my_army
        self._peak_land = max(self._peak_land, obs.my_land)
        self._peak_army = max(self._peak_army, obs.my_army)

    def _track_enemy_general(self, obs: UnifiedObservation) -> None:
        if self.saw_enemy_general_at is not None:
            return
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.type_grid[r][c] == 4 and obs.owner_grid[r][c] == 2:
                    self.saw_enemy_general_at = obs.turn
                    return

    def session_stats(self) -> dict:
        stats = {
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
        if self._first_fault_traceback is not None:
            stats["first_fault_traceback"] = self._first_fault_traceback
        return stats

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
        self._first_fault_traceback = None
