"""Kill-window planner: exploit an underdefended visible general.

Field-observed miss: macaria's general sat at 2 army while morpheus had a
1-stack adjacent, a 2-stack beside it, and a 7-stack three cells away — no
score term or 2-ply search can represent the collecting march, so macaria
recovered and won. The planner forces the march step by step.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from memory import empty_memory, update_memory
from observe import emit_observation
from state import create_initial_state
from tactics import constrain_nn_action, winning_kill_move
from transition import DEATHTOUCH_TURN, DIRECTIONS


def _board(*, gen_army: int, stacks: dict, time: int = 300):
    """Enemy general at (0,4); own column below it per ``stacks`` {row: army}."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 4] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r, a in stacks.items():
        ownership[0, r, 4] = True
        neut[r, 4] = False
        armies[r, 4] = a
    armies[0, 4] = gen_army
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=time
    )
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(8, 8), obs)


def _up() -> int:
    return next(i for i in range(4) if tuple(DIRECTIONS[i]) == (-1, 0))


def test_field_scenario_marches_the_seven_stack():
    # gen=2; own: 1 adjacent, 2 at dist 2, 7 at dist 3.
    obs, mem = _board(gen_army=2, stacks={1: 1, 2: 2, 3: 7})
    step = winning_kill_move(obs, mem)
    assert step is not None
    # First move: the 7-stack at (3,4) marches up, collecting the 2-stack.
    assert (int(step[1]), int(step[2])) == (3, 4)
    assert int(step[3]) == _up() and int(step[4]) == 0
    # And constrain forces it over whatever the NN wanted.
    out = constrain_nn_action(obs, mem, (1, 0, 0, 0, 0))
    assert tuple(int(x) for x in out) == tuple(int(x) for x in step)


def test_no_plan_when_garrison_too_big():
    obs, mem = _board(gen_army=60, stacks={1: 1, 2: 2, 3: 7})
    assert winning_kill_move(obs, mem) is None


def test_growth_is_priced_in():
    # 3-turn march, arrival moved=7: needs > 2 + 2 growth = 4 -> wins.
    # With gen=7 the same march needs > 9 -> no plan.
    obs, mem = _board(gen_army=7, stacks={1: 1, 2: 2, 3: 7})
    assert winning_kill_move(obs, mem) is None


def test_deathtouch_march_needs_no_surplus():
    obs, mem = _board(
        gen_army=60, stacks={1: 1, 2: 2, 3: 7}, time=DEATHTOUCH_TURN + 10
    )
    step = winning_kill_move(obs, mem)
    assert step is not None
    # Any arrival wins post-800, so the FASTEST plan (the 2-stack, two turns)
    # beats the bigger 7-stack march.
    assert (int(step[1]), int(step[2])) == (2, 4)


def test_prefers_the_fastest_winning_plan():
    # A big stack at dist 5 and a sufficient stack at dist 2: take the fast
    # one. Row 1 holds a scout so the general is actually visible.
    obs, mem = _board(gen_army=2, stacks={1: 1, 2: 8, 5: 40})
    step = winning_kill_move(obs, mem)
    assert step is not None
    assert (int(step[1]), int(step[2])) == (2, 4)


def test_invisible_general_yields_no_plan():
    # No enemy cell visible at all -> planner stays quiet.
    obs, mem = _board(gen_army=2, stacks={5: 40})
    # Rebuild board without the enemy general in vision: move our stacks away.
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 4] = 2
    state = create_initial_state(grid)
    obs2 = emit_observation(state, 0)
    mem2 = update_memory(empty_memory(8, 8), obs2)
    assert winning_kill_move(obs2, mem2) is None
