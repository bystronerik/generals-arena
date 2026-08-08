"""Garrison floor: the general never empties while a rush can still come.

Every benchmarked loss vs blitz/macaria was the own general falling; the
garrison marched to the front (probe: 26->7 with an enemy 4 cells away).
Reactive visible-threat defense measured no better because rushes arrive
through fog. The floor is enforced in ``play_mask``, so search candidates,
the shaped prior, fallbacks, and constrain all inherit it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import encode_action
from memory import empty_memory, update_memory
from observe import emit_observation
from state import create_initial_state
from tactics import (
    GARRISON_FLOOR_CAP,
    GARRISON_FLOOR_FROM,
    GARRISON_FLOOR_MIN,
    garrison_floor,
    play_mask,
)
from transition import DEATHTOUCH_TURN, DIRECTIONS


def _board(*, gen_army: int, time: int, enemy_adjacent_gen_army: int = 0):
    """Own general bottom-left with a corridor; optional adjacent enemy gen."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[9, 0] = 1
    grid[0, 9] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r in range(6, 9):
        ownership[0, r, 0] = True
        neut[r, 0] = False
        armies[r, 0] = 4
    armies[9, 0] = gen_army
    if enemy_adjacent_gen_army:
        # Put the ENEMY general right next to ours (exemption test).
        gp = np.asarray(state.general_positions).copy()
        generals = np.asarray(state.generals).copy()
        generals[0, 9] = False
        generals[9, 1] = True
        gp[1] = (9, 1)
        ownership[1, 0, 9] = False
        ownership[1, 9, 1] = True
        neut[9, 1] = False
        armies[9, 1] = enemy_adjacent_gen_army
        state = state._replace(generals=generals, general_positions=gp)
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=time
    )
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(10, 10), obs)


def _move_up_from_gen(split: int = 0) -> int:
    d_up = next(i for i in range(4) if tuple(DIRECTIONS[i]) == (-1, 0))
    return encode_action((0, 9, 0, d_up, split))


def test_floor_formula():
    assert garrison_floor(0) == GARRISON_FLOOR_MIN
    assert garrison_floor(100) == GARRISON_FLOOR_MIN
    assert garrison_floor(500) == GARRISON_FLOOR_CAP
    assert garrison_floor(10_000) == GARRISON_FLOOR_CAP
    # The release valve (half-move) must open at a modest stack size: the
    # first cut capped at 40 and hoarded to ~79 before anything could ship.
    assert 2 * GARRISON_FLOOR_CAP <= 40


def test_full_move_off_general_banned_after_floor_activates():
    obs, mem = _board(gen_army=15, time=GARRISON_FLOOR_FROM + 10)
    mask = play_mask(obs, mem)
    assert not mask[_move_up_from_gen(split=0)]  # would leave 1 < floor
    # Half split of 15 leaves 8 < 10: also banned.
    assert not mask[_move_up_from_gen(split=1)]


def test_half_split_ships_once_garrison_doubles():
    obs, mem = _board(gen_army=25, time=GARRISON_FLOOR_FROM + 10)
    mask = play_mask(obs, mem)
    # Half of 25 leaves 13 >= 10: the release valve opens.
    assert mask[_move_up_from_gen(split=1)]
    assert not mask[_move_up_from_gen(split=0)]


def test_inactive_before_activation_turn_and_in_deathtouch():
    early_obs, early_mem = _board(gen_army=20, time=GARRISON_FLOOR_FROM - 20)
    assert play_mask(early_obs, early_mem)[_move_up_from_gen(split=0)]
    late_obs, late_mem = _board(gen_army=20, time=DEATHTOUCH_TURN + 5)
    assert play_mask(late_obs, late_mem)[_move_up_from_gen(split=0)]


def test_king_and_commitment_exclude_the_floored_general():
    """Gather must assemble FORWARD, not feed the locked garrison, and tips
    must not be thrash-crushed against an immovable stack."""
    from tactics import _movable_exclude_cell, army_concentration, king_cell

    obs, mem = _board(gen_army=50, time=GARRISON_FLOOR_FROM + 10)
    # Add a forward stack smaller than the garrison.
    excl = _movable_exclude_cell(obs, mem)
    assert excl == (9, 0)
    assert king_cell(obs) == (9, 0)  # raw: general is the biggest
    # Forward stack (6,0)=4 is the biggest MOVABLE stack.
    assert king_cell(obs, exclude=excl) != (9, 0)
    _share, max_own, _tot = army_concentration(obs, exclude=excl)
    assert max_own == 4  # not 50 — tips are judged against movable armies


def test_exclusion_ends_when_floor_ends():
    from tactics import _movable_exclude_cell

    obs, mem = _board(gen_army=50, time=GARRISON_FLOOR_FROM - 10)
    assert _movable_exclude_cell(obs, mem) is None
    obs2, mem2 = _board(gen_army=50, time=DEATHTOUCH_TURN + 1)
    assert _movable_exclude_cell(obs2, mem2) is None


def test_winning_enemy_general_capture_is_exempt():
    obs, mem = _board(
        gen_army=20, time=GARRISON_FLOOR_FROM + 10, enemy_adjacent_gen_army=5
    )
    d_right = next(i for i in range(4) if tuple(DIRECTIONS[i]) == (0, 1))
    cap = encode_action((0, 9, 0, d_right, 0))  # 19 moved > 5 defenders
    mask = play_mask(obs, mem)
    assert mask[cap]
    # Ordinary corridor move off the general stays banned meanwhile.
    assert not mask[_move_up_from_gen(split=0)]


def test_release_is_forced_once_garrison_doubles():
    from tactics import GARRISON_RELEASE_FACTOR, constrain_nn_action, garrison_floor

    obs, mem = _board(gen_army=25, time=GARRISON_FLOOR_FROM + 10)
    floor = garrison_floor(25 + 12)
    assert 25 >= GARRISON_RELEASE_FACTOR * floor
    # NN picks a front tip move; the release hard rule overrides with a
    # general half-move (the NN never plays splits, so scores cannot do it).
    d_up = next(i for i in range(4) if tuple(DIRECTIONS[i]) == (-1, 0))
    nn_choice = (0, 6, 0, d_up, 0)
    out = constrain_nn_action(obs, mem, nn_choice)
    assert (int(out[1]), int(out[2])) == (9, 0)
    assert int(out[4]) == 1  # half split


def test_release_not_forced_below_threshold_or_when_already_shipping():
    from tactics import constrain_nn_action

    d_up = next(i for i in range(4) if tuple(DIRECTIONS[i]) == (-1, 0))
    # Below 2x floor: keep the NN choice.
    obs, mem = _board(gen_army=15, time=GARRISON_FLOOR_FROM + 10)
    nn_choice = (0, 6, 0, d_up, 0)
    assert constrain_nn_action(obs, mem, nn_choice) == nn_choice
    # Already shipping from the general: no double-override.
    obs2, mem2 = _board(gen_army=25, time=GARRISON_FLOOR_FROM + 10)
    ship = (0, 9, 0, d_up, 1)
    assert constrain_nn_action(obs2, mem2, ship) == ship


def test_ban_skipped_when_general_is_only_move_source():
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 7] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    armies[7, 0] = 20
    state = state._replace(armies=armies, time=GARRISON_FLOOR_FROM + 10)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(8, 8), obs)
    mask = play_mask(obs, mem)
    # Only the general has army: banning it would leave pass-only. The ban
    # must yield to protocol safety and keep some general move legal.
    nonpass = np.asarray(mask, dtype=bool).copy()
    from action import PASS_INDEX

    nonpass[PASS_INDEX] = False
    assert np.any(nonpass)
