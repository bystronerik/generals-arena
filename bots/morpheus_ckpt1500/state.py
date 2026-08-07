"""Canonical in-memory competition state for Morpheus.

Bundle-safe: no competition-module import. Field layout mirrors the engine
GameState so differential tests and the training adapter can compare cell-wise.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

Array = np.ndarray


class GameState(NamedTuple):
    """Complete competition board state.

    Attributes match competition-module/generals/core/game.py GameState.
    """

    armies: Array  # (H, W) int32
    ownership: Array  # (2, H, W) bool
    ownership_neutral: Array  # (H, W) bool
    generals: Array  # (H, W) bool
    castles: Array  # (H, W) bool
    mountains: Array  # (H, W) bool
    passable: Array  # (H, W) bool
    general_positions: Array  # (2, 2) int32 [row, col]
    time: int  # pre-step timestep
    winner: int  # -1 ongoing, 0 or 1 winner
    pool_idx: int  # engine pool index; kept for parity


class GameInfo(NamedTuple):
    """Per-turn totals and terminal flag after one transition."""

    army: Array  # (2,) int
    land: Array  # (2,) int
    is_done: bool
    winner: int
    time: int


def create_initial_state(grid: Array) -> GameState:
    """Build a GameState from a numeric grid.

    Cell codes: -2 mountain, 0 empty, 1/2 generals, >2 castle with that army.
    """
    grid = np.asarray(grid, dtype=np.int32)
    is_general_0 = grid == 1
    is_general_1 = grid == 2
    generals = is_general_0 | is_general_1
    mountains = grid == -2
    passable = grid != -2
    castles = grid > 2

    ownership = np.stack([is_general_0, is_general_1])
    ownership_neutral = passable & ~is_general_0 & ~is_general_1

    armies = np.where(is_general_0 | is_general_1, 1, 0).astype(np.int32)
    armies = np.where(castles, grid, armies)

    pos0 = np.argwhere(is_general_0)
    pos1 = np.argwhere(is_general_1)
    g0 = pos0[0] if len(pos0) else np.array([-1, -1], dtype=np.int32)
    g1 = pos1[0] if len(pos1) else np.array([-1, -1], dtype=np.int32)
    general_positions = np.stack([g0, g1]).astype(np.int32)

    return GameState(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        generals=generals,
        castles=castles,
        mountains=mountains,
        passable=passable,
        general_positions=general_positions,
        time=0,
        winner=-1,
        pool_idx=0,
    )


def from_engine(engine_state) -> GameState:
    """Copy a JAX/engine GameState into the Morpheus NamedTuple."""
    return GameState(
        armies=np.asarray(engine_state.armies, dtype=np.int32).copy(),
        ownership=np.asarray(engine_state.ownership, dtype=bool).copy(),
        ownership_neutral=np.asarray(engine_state.ownership_neutral, dtype=bool).copy(),
        generals=np.asarray(engine_state.generals, dtype=bool).copy(),
        castles=np.asarray(engine_state.castles, dtype=bool).copy(),
        mountains=np.asarray(engine_state.mountains, dtype=bool).copy(),
        passable=np.asarray(engine_state.passable, dtype=bool).copy(),
        general_positions=np.asarray(engine_state.general_positions, dtype=np.int32).copy(),
        time=int(engine_state.time),
        winner=int(engine_state.winner),
        pool_idx=int(engine_state.pool_idx),
    )


def states_equal(a: GameState, b: GameState) -> bool:
    """True when every field matches (used by differential tests)."""
    return (
        np.array_equal(a.armies, b.armies)
        and np.array_equal(a.ownership, b.ownership)
        and np.array_equal(a.ownership_neutral, b.ownership_neutral)
        and np.array_equal(a.generals, b.generals)
        and np.array_equal(a.castles, b.castles)
        and np.array_equal(a.mountains, b.mountains)
        and np.array_equal(a.passable, b.passable)
        and np.array_equal(a.general_positions, b.general_positions)
        and a.time == b.time
        and a.winner == b.winner
        and a.pool_idx == b.pool_idx
    )


def infos_equal(a: GameInfo, b: GameInfo) -> bool:
    return (
        np.array_equal(a.army, b.army)
        and np.array_equal(a.land, b.land)
        and a.is_done == b.is_done
        and a.winner == b.winner
        and a.time == b.time
    )
