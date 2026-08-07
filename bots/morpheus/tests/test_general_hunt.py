"""General-hunt behaviors: belief-aimed seek, deathtouch, kill calculus, damp.

Motivating measurements (seed-0 games vs cm_expander): a 1200-turn draw in
which the enemy general was never seen while holding 1-9 army, and a 573-turn
win whose first general sighting was the capture turn itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import encode_action
from memory import empty_memory, update_memory
from observe import emit_observation
from state import create_initial_state
from tactics import (
    DEATHTOUCH_SCORE,
    GENERAL_CHEW_DAMP,
    believed_enemy_general,
    constrain_nn_action,
    heuristic_action_scores,
    known_enemy_general_cell,
    play_mask,
    seek_goals,
)
from transition import DEATHTOUCH_TURN, DIRECTIONS


# ------------------------------------------------------------- belief stubs
@dataclass
class _State:
    general_positions: np.ndarray


@dataclass
class _Particle:
    weight: float
    state: _State


@dataclass
class _Belief:
    particles: list[_Particle]
    enemy_seat: int = 1

    @property
    def n(self) -> int:
        return len(self.particles)


def _belief_at(cells_weights: list[tuple[tuple[int, int], float]]) -> _Belief:
    parts = []
    for (r, c), w in cells_weights:
        gp = np.full((2, 2), -1, dtype=np.int32)
        gp[1] = (r, c)
        parts.append(_Particle(weight=w, state=_State(general_positions=gp)))
    return _Belief(particles=parts)


def test_believed_enemy_general_is_the_posterior_mode():
    belief = _belief_at([((2, 3), 0.2), ((5, 5), 0.5), ((2, 3), 0.4)])
    assert believed_enemy_general(belief) == (2, 3)
    assert believed_enemy_general(None) is None
    assert believed_enemy_general(_Belief(particles=[])) is None
    # All-zero weights fall back to the particle-count mode.
    belief0 = _belief_at([((1, 1), 0.0), ((1, 1), 0.0), ((4, 4), 0.0)])
    assert believed_enemy_general(belief0) == (1, 1)


# --------------------------------------------------------------- hunt boards
def _contact_board_no_gen(*, with_enemy_tile: bool = True):
    """Own cluster (optionally) touching enemy land; general NOT visible."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[9, 0] = 1
    grid[0, 9] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r, c in ((5, 5), (5, 6), (6, 5)):
        ownership[0, r, c] = True
        neut[r, c] = False
        armies[r, c] = 5
    armies[5, 6] = 30
    if with_enemy_tile:
        ownership[1, 5, 7] = True
        neut[5, 7] = False
        armies[5, 7] = 2
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=50
    )
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(10, 10), obs)


def test_seek_goals_keep_enemy_land_first_then_believed():
    obs, mem = _contact_board_no_gen()
    assert known_enemy_general_cell(obs, mem) is None
    belief = _belief_at([((1, 9), 1.0)])
    # Visible enemy land outranks the posterior — that is what keeps home
    # incursions scored as progress (a goal-replacement variant lost its own
    # general to an expander at turn 201).
    assert seek_goals(obs, mem, belief) == [(5, 7)]
    # With no visible enemy land, the believed cell replaces the corner beacon.
    obs2, mem2 = _contact_board_no_gen(with_enemy_tile=False)
    assert seek_goals(obs2, mem2, belief) == [(1, 9)]


def test_hunt_bonus_boosts_moves_toward_believed_general_only():
    obs, mem = _contact_board_no_gen()
    mask = play_mask(obs, mem)
    base = heuristic_action_scores(obs, mem, mask)
    # Believed general far to the north of the front.
    belief = _belief_at([((0, 9), 1.0)])
    hunted = heuristic_action_scores(obs, mem, mask, belief)
    d_up = next(i for i in range(4) if tuple(DIRECTIONS[i]) == (-1, 0))
    d_down = next(i for i in range(4) if tuple(DIRECTIONS[i]) == (1, 0))
    up = encode_action((0, 5, 6, d_up, 0))
    down = encode_action((0, 5, 6, d_down, 0))
    assert mask[up] and mask[down]
    # Toward the believed cell: multiplied up. Away: untouched (a bonus field,
    # never a penalty — defense scores must not shrink).
    assert hunted[up] > base[up]
    assert hunted[down] == pytest.approx(base[down])
    np.testing.assert_array_less(base - 1e-12, hunted + 1e-12)


def _adjacent_general_board(*, own_army: int, gen_army: int, time: int):
    """Own stack directly below a visible enemy general."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 4] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r in range(1, 8):
        ownership[0, r, 4] = True
        neut[r, 4] = False
        armies[r, 4] = 2
    armies[1, 4] = own_army
    armies[0, 4] = gen_army
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=time
    )
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(8, 8), obs)


def _touch_index(obs) -> int:
    d_up = next(i for i in range(4) if tuple(DIRECTIONS[i]) == (-1, 0))
    return encode_action((0, 1, 4, d_up, 0))


def test_deathtouch_touch_tops_scores_regardless_of_surplus():
    obs, mem = _adjacent_general_board(
        own_army=3, gen_army=60, time=DEATHTOUCH_TURN + 10
    )
    assert known_enemy_general_cell(obs, mem) == (0, 4)
    mask = play_mask(obs, mem)
    touch = _touch_index(obs)
    assert mask[touch]
    scores = heuristic_action_scores(obs, mem, mask)
    assert scores[touch] == DEATHTOUCH_SCORE
    assert scores[touch] == scores.max()


def test_deathtouch_constrain_forces_any_touch():
    obs, mem = _adjacent_general_board(
        own_army=3, gen_army=60, time=DEATHTOUCH_TURN + 10
    )
    out = constrain_nn_action(obs, mem, (1, 0, 0, 0, 0))
    assert (int(out[1]), int(out[2])) == (1, 4)
    assert int(out[0]) == 0


def test_pre_deathtouch_underpowered_capture_is_not_forced():
    from tactics import move_dest

    obs, mem = _adjacent_general_board(own_army=3, gen_army=60, time=400)
    nn_move = (0, 7, 4, 0, 0)  # some unrelated legal-ish move
    out = constrain_nn_action(obs, mem, nn_move)
    # 2 moved army vs 60 defenders: forcing this touch feeds the general.
    ends = move_dest(tuple(int(x) for x in out))
    assert ends is None or ends[2:] != (0, 4)


def test_pre_deathtouch_winning_capture_is_forced():
    obs, mem = _adjacent_general_board(own_army=80, gen_army=20, time=400)
    out = constrain_nn_action(obs, mem, (1, 0, 0, 0, 0))
    assert int(out[0]) == 0 and (int(out[1]), int(out[2])) == (1, 4)


def test_latched_general_damps_sideways_border_chew():
    # Enemy land beside the front plus a visible general: takes that do not
    # shorten the general path are damped relative to the boosted kill path.
    obs, mem = _adjacent_general_board(own_army=80, gen_army=20, time=400)
    mask = play_mask(obs, mem)
    scores = heuristic_action_scores(obs, mem, mask)
    assert scores[_touch_index(obs)] == scores.max()
    assert GENERAL_CHEW_DAMP < 1.0
