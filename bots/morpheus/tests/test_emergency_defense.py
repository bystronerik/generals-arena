"""Emergency defense: reinforce or intercept when the general is threatened.

Field-observed loss: an enemy stack bigger than the garrison two steps from
our general, a field stack adjacent to the general able to save it, and the
bot ignored the threat and died.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from memory import empty_memory, update_memory
from observe import emit_observation
from state import create_initial_state
from tactics import (
    DEFENSE_RADIUS,
    constrain_nn_action,
    defend_general_move,
    general_threat,
)
from transition import DEATHTOUCH_TURN, DIRECTIONS


def _board(*, garrison: int, threat_army: int, threat_dist: int = 2,
           helper_army: int = 8, time: int = 300, helper: bool = True):
    """Our general (7,0); enemy stack approaching down column 0; own helper
    at (7,1), adjacent to the general."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 7] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    armies[7, 0] = garrison
    tr = 7 - threat_dist
    ownership[1, tr, 0] = True
    neut[tr, 0] = False
    armies[tr, 0] = threat_army
    # Scout line keeps the approach visible (1-army cells: no accidental
    # interceptors or reinforcements).
    for r in range(tr + 1, 7):
        ownership[0, r, 0] = True
        neut[r, 0] = False
        armies[r, 0] = 1
    if helper:
        ownership[0, 7, 1] = True
        neut[7, 1] = False
        armies[7, 1] = helper_army
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=time
    )
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(8, 8), obs)


def test_threat_detected_and_helper_reinforces():
    obs, mem = _board(garrison=5, threat_army=20)
    threat = general_threat(obs, mem)
    assert threat is not None and threat[1] == 2
    move = defend_general_move(obs, mem, threat)
    assert move is not None
    # The helper at (7,1) moves onto the general (7,0).
    assert (int(move[1]), int(move[2])) == (7, 1)
    d = int(move[3])
    assert (7 + int(DIRECTIONS[d, 0]), 1 + int(DIRECTIONS[d, 1])) == (7, 0)
    out = constrain_nn_action(obs, mem, (1, 0, 0, 0, 0))
    assert tuple(int(x) for x in out) == tuple(int(x) for x in move)


def test_intercept_beats_reinforcement_when_a_kill_is_available():
    from state import create_initial_state as _cis  # local rebuild

    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 7] = 2
    state = _cis(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    armies[7, 0] = 5
    ownership[1, 5, 0] = True; neut[5, 0] = False; armies[5, 0] = 12
    ownership[0, 6, 0] = True; neut[6, 0] = False; armies[6, 0] = 1
    ownership[0, 7, 1] = True; neut[7, 1] = False; armies[7, 1] = 8
    # Interceptor adjacent to the threat, big enough to kill it outright.
    ownership[0, 5, 1] = True; neut[5, 1] = False; armies[5, 1] = 20
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=300
    )
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(8, 8), obs)
    threat = general_threat(obs, mem)
    assert threat is not None and threat[0] == (5, 0)
    move = defend_general_move(obs, mem, threat)
    assert move is not None
    # The interceptor kills the threat stack rather than huddling.
    assert (int(move[1]), int(move[2])) == (5, 1)
    tr = 5 + int(DIRECTIONS[int(move[3]), 0])
    tc = 1 + int(DIRECTIONS[int(move[3]), 1])
    assert (tr, tc) == (5, 0)


def test_no_threat_no_defense():
    obs, mem = _board(garrison=30, threat_army=10)
    assert general_threat(obs, mem) is None


def test_arrival_math_prices_garrison_growth():
    # army - d must reach garrison + d//2: 12 - 2 = 10 vs 9 + 1 -> exactly
    # equal counts as a threat (defense errs conservative).
    obs, mem = _board(garrison=9, threat_army=12)
    assert general_threat(obs, mem) is not None
    obs2, mem2 = _board(garrison=10, threat_army=12)
    assert general_threat(obs2, mem2) is None


def test_deathtouch_any_reachable_stack_is_a_threat():
    obs, mem = _board(garrison=50, threat_army=4, time=DEATHTOUCH_TURN + 5)
    assert general_threat(obs, mem) is not None


def test_garrison_split_is_illegal_under_a_nearby_stack():
    """Field-observed loss: 44 garrison, 40-army enemy two steps away, a
    LEGAL half-split to 22, dead. The stack arrives with 38, so the mask
    must require 39 to stay — every move off the general becomes illegal."""
    from action import encode_action
    from tactics import max_threat_arrival, play_mask

    obs, mem = _board(garrison=44, threat_army=40, threat_dist=2)
    assert max_threat_arrival(obs, mem) == 38
    # 44 survives 38: correctly NOT an emergency...
    assert general_threat(obs, mem) is None
    # ...but shipping half of it would not survive, so the mask forbids it.
    mask = play_mask(obs, mem)
    for d in range(4):
        for s in (0, 1):
            assert not mask[encode_action((0, 7, 0, d, s))]


def test_big_garrison_may_still_ship_past_the_threat_floor():
    from action import encode_action
    from tactics import play_mask

    obs, mem = _board(garrison=90, threat_army=40, threat_dist=2)
    # Half of 90 leaves 45 > 39: the release valve stays available.
    mask = play_mask(obs, mem)
    assert any(mask[encode_action((0, 7, 0, d, 1))] for d in range(4))


def test_out_of_radius_is_not_a_threat():
    obs, mem = _board(garrison=5, threat_army=40,
                      threat_dist=DEFENSE_RADIUS + 2)
    assert general_threat(obs, mem) is None
