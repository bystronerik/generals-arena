"""Wave assembly: gather targets the front toward the hunt target.

The old target (largest movable stack) moved every time any cell's army
changed, so gather flows chased it and the army stayed dribbled ~4/cell
while blitz-style bots massed and struck.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from memory import empty_memory, update_memory
from observe import emit_observation
from state import create_initial_state
from tactics import king_cell, wave_assembly_cell


def _board(*, latched: bool):
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[9, 0] = 1
    grid[0, 9] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    # Own column toward the enemy corner; big pile in the REAR.
    for r in range(4, 10):
        ownership[0, r, 0] = True
        neut[r, 0] = False
        armies[r, 0] = 3
    armies[8, 0] = 40  # rear pile (old king)
    armies[4, 0] = 5   # front tip
    if latched:
        ownership[1, 0, 9] = True
        neut[0, 9] = False
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=200
    )
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(10, 10), obs)
    if latched:
        kg = np.zeros((10, 10), dtype=bool)
        kg[0, 9] = True
        mem = mem._replace(known_enemy_general=kg)
    return obs, mem


def test_assembly_is_front_cell_not_rear_pile():
    obs, mem = _board(latched=True)
    # Raw king is the 40-army rear pile; the assembly point is the own cell
    # nearest the latched general — the front tip.
    assert king_cell(obs) == (8, 0)
    cell = wave_assembly_cell(obs, mem)
    assert cell is not None and cell[0] <= 5


def test_assembly_falls_back_to_king_without_hunt_target():
    obs, mem = _board(latched=False)
    assert wave_assembly_cell(obs, mem) == king_cell(obs)


def test_assembly_is_deterministic_and_stable():
    obs, mem = _board(latched=True)
    a = wave_assembly_cell(obs, mem)
    b = wave_assembly_cell(obs, mem)
    assert a == b
