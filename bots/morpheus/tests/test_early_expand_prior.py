"""Play-mask and expand-prior rules: no pass, no pre-contact general stack."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import PASS_INDEX, decode_action, legal_mask
from memory import empty_memory, update_memory
from observe import emit_observation
from state import create_initial_state
from tactics import (
    apply_pre_contact_prior,
    enemy_is_visible,
    frontier_expand_indices,
    play_mask,
)
from transition import DIRECTIONS


def _corridor_obs():
    """Three-tile north corridor with army stacked off the tip."""
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 7] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    ownership[0, 5, 0] = True
    ownership[0, 6, 0] = True
    armies[7, 0] = 3
    armies[6, 0] = 40
    armies[5, 0] = 2
    state = state._replace(armies=armies, ownership=ownership)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(8, 8), obs)
    return obs, mem


def test_play_mask_bans_pass_when_alternatives_exist():
    obs, mem = _corridor_obs()
    mask = play_mask(obs, mem)
    assert not mask[PASS_INDEX]
    assert int(mask.sum()) >= 1


def test_play_mask_bans_structure_reinforce_before_contact():
    obs, mem = _corridor_obs()
    assert not enemy_is_visible(obs, mem)
    mask = play_mask(obs, mem)
    own_struct = np.asarray(mem.own_general, dtype=bool)
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        action = decode_action(idx)
        if action[0] != 0:
            continue
        sr, sc, d = action[1], action[2], action[3]
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        # May leave a structure onto own land or fog; never stack onto it.
        assert not own_struct[tr, tc]


def test_play_mask_allows_evacuate_general_onto_own_land():
    obs, mem = _corridor_obs()
    mask = play_mask(obs, mem)
    # Gen at (7,0) with army; north own cell (6,0) exists — evacuate must be legal.
    gen_r, gen_c = 7, 0
    found = False
    for idx in np.flatnonzero(mask):
        action = decode_action(int(idx))
        if action[0] != 0:
            continue
        if int(action[1]) != gen_r or int(action[2]) != gen_c:
            continue
        tr = gen_r + int(DIRECTIONS[int(action[3]), 0])
        tc = gen_c + int(DIRECTIONS[int(action[3]), 1])
        owners = np.asarray(obs.owner_grid)
        if int(owners[tr, tc]) == 1:
            found = True
            break
    assert found

def test_frontier_expand_indices_lists_unowned_destinations():
    obs, mem = _corridor_obs()
    mask = play_mask(obs, mem)
    frontier = frontier_expand_indices(obs, mem, mask)
    assert frontier
    owners = np.asarray(obs.owner_grid)
    for idx in frontier:
        action = decode_action(idx)
        assert action[0] == 0
        tr = action[1] + int(DIRECTIONS[action[3], 0])
        tc = action[2] + int(DIRECTIONS[action[3], 1])
        assert int(owners[tr, tc]) == 0


def test_fog_urgency_rises_until_contact():
    from tactics import castle_timing_weight, explore_wave_weight, fog_urgency

    assert fog_urgency(0, enemy_seen=False) == pytest.approx(1.0)
    assert fog_urgency(200, enemy_seen=False) > fog_urgency(50, enemy_seen=False)
    assert fog_urgency(200, enemy_seen=True) == pytest.approx(1.0)
    assert explore_wave_weight(20) > explore_wave_weight(2)
    assert castle_timing_weight(40) > castle_timing_weight(250)
    assert castle_timing_weight(100) > 1.0
    assert castle_timing_weight(400) < 1.0


def test_prior_scores_castle_builds_with_timing():
    from tactics import apply_pre_contact_prior, castle_timing_weight

    obs, mem = _corridor_obs()
    mask = play_mask(obs, mem)
    build_idxs = [
        int(i)
        for i in np.flatnonzero(mask)
        if int(decode_action(int(i))[0]) == 2
    ]
    if not build_idxs:
        # Timing curve still holds even without a legal build on this board.
        assert castle_timing_weight(50) > castle_timing_weight(300)
        return
    prior = np.full(mask.shape, 1.0 / max(int(mask.sum()), 1), dtype=np.float64)
    shaped = apply_pre_contact_prior(prior, obs, mem, mask=mask)
    assert shaped[build_idxs[0]] > 0.0


def test_blocks_own_land_oscillation():
    from tactics import blocks_oscillation, is_reverse_move

    obs, _mem = _corridor_obs()
    # Tip at (6,0) moved north to (5,0) last turn; reverse is (5,0)->(6,0).
    prev = (0, 6, 0, 0, 0)  # north
    reverse = (0, 5, 0, 1, 0)  # south back
    assert is_reverse_move(reverse, prev)
    assert blocks_oscillation(reverse, prev, obs)
    other = (0, 5, 0, 0, 0)  # continue north
    assert not is_reverse_move(other, prev)


def test_blocks_delayed_corridor_retreat():
    """A→B then B→C then C→B is oscillation even with a gap step."""
    from tactics import blocks_oscillation

    obs, _mem = _corridor_obs()
    a_to_b = (0, 6, 0, 0, 0)  # (6,0)->(5,0)
    b_to_c = (0, 5, 0, 0, 0)  # (5,0)->(4,0)
    c_to_b = (0, 4, 0, 1, 0)  # (4,0)->(5,0) retreat
    recent = (a_to_b, b_to_c)
    assert blocks_oscillation(
        c_to_b, None, obs, recent_actions=recent
    )
    # Continuing forward is fine.
    c_to_d = (0, 4, 0, 0, 0)  # (4,0)->(3,0)
    assert not blocks_oscillation(
        c_to_d, None, obs, recent_actions=recent
    )


def test_prior_ranks_enemy_take_above_fog_after_contact():
    from action import encode_action
    from tactics import apply_pre_contact_prior

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
    ownership[1, 5, 7] = True
    neut[5, 7] = False
    armies[5, 7] = 2
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=50
    )
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(10, 10), obs)
    mask = play_mask(obs, mem)
    prior = np.full(mask.shape, 1.0 / max(int(mask.sum()), 1), dtype=np.float64)
    shaped = apply_pre_contact_prior(prior, obs, mem, mask=mask)
    enemy_idx = encode_action((0, 5, 6, 3, 0))  # east onto enemy (5,7)
    # Fog/neutral north from (5,6) if legal.
    fog_idx = encode_action((0, 5, 6, 0, 0))  # north
    if mask[enemy_idx] and mask[fog_idx]:
        assert shaped[enemy_idx] > shaped[fog_idx]


def test_constrain_nn_keeps_non_oscillating_move():
    from tactics import constrain_nn_action

    obs, mem = _corridor_obs()
    # Expand off the tip into fog — positive reveal, keep NN choice.
    nn_move = (0, 5, 0, 0, 0)
    out = constrain_nn_action(obs, mem, nn_move, prev_action=None)
    assert out == nn_move


def test_overstack_gather_is_punished():
    from tactics import STACK_GATHER_BAN, stack_gather_factor, wave_weight

    assert wave_weight(400) == pytest.approx(wave_weight(20))
    # Lateral / default progress: fat pile dumps stay crushed.
    assert stack_gather_factor(400, 50, 1) < 0.01
    assert stack_gather_factor(10, 2, 1) > stack_gather_factor(400, 50, 1)
    assert stack_gather_factor(25, 1, 2) == pytest.approx(1.0)
    assert stack_gather_factor(200, 2, 1) > 0.5
    # Toward-enemy consolidation is allowed (soft friction only).
    assert stack_gather_factor(400, 50, 1, progress=1.0) > 0.2
    assert stack_gather_factor(400, 50, 1, progress=-1.0) < 0.05
    assert STACK_GATHER_BAN <= 16


def test_direction_bias_ranks_enemy_above_toward_own_above_retreat():
    from tactics import direction_bias

    enemy = direction_bias(1.0, 2)
    toward_own = direction_bias(1.0, 1)
    lateral_own = direction_bias(0.0, 1)
    retreat = direction_bias(-1.0, 1)
    assert enemy > toward_own > lateral_own > retreat


def test_prior_rewards_toward_enemy_own_march_over_retreat():
    from action import encode_action
    from tactics import apply_pre_contact_prior

    grid = np.zeros((12, 12), dtype=np.int32)
    grid[11, 5] = 1
    grid[0, 5] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r in range(4, 12):
        ownership[0, r, 5] = True
        neut[r, 5] = False
        armies[r, 5] = 2
    armies[8, 5] = 40
    ownership[1, 3, 5] = True
    neut[3, 5] = False
    armies[3, 5] = 2
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=60
    )
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(12, 12), obs)
    mask = play_mask(obs, mem)
    prior = np.full(mask.shape, 1.0 / max(int(mask.sum()), 1), dtype=np.float64)
    shaped = apply_pre_contact_prior(prior, obs, mem, mask=mask)
    toward = encode_action((0, 8, 5, 0, 0))  # north toward enemy
    away = encode_action((0, 8, 5, 1, 0))  # south away
    assert mask[toward] and mask[away]
    assert shaped[toward] > shaped[away]


def test_constrain_replaces_retreat_with_seek():
    from tactics import apply_pre_contact_prior, constrain_nn_action, play_mask

    grid = np.zeros((12, 12), dtype=np.int32)
    grid[11, 5] = 1
    grid[0, 5] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r in range(4, 12):
        ownership[0, r, 5] = True
        neut[r, 5] = False
        armies[r, 5] = 2
    armies[8, 5] = 60
    ownership[1, 3, 5] = True
    neut[3, 5] = False
    armies[3, 5] = 2
    mem = empty_memory(12, 12)
    kg = np.zeros((12, 12), dtype=bool)
    kg[0, 5] = True
    og = np.zeros((12, 12), dtype=bool)
    og[11, 5] = True
    mem = mem._replace(known_enemy_general=kg, own_general=og)
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=90
    )
    obs = emit_observation(state, 0)
    mem = update_memory(mem, obs)
    mem = mem._replace(known_enemy_general=kg)
    mask = play_mask(obs, mem)
    prior = np.full(mask.shape, 1.0 / max(int(mask.sum()), 1), dtype=np.float64)
    shaped = apply_pre_contact_prior(prior, obs, mem, mask=mask)
    retreat = (0, 8, 5, 1, 0)  # south, away from enemy general
    out = constrain_nn_action(obs, mem, retreat, prev_action=None, prior=shaped)
    assert out[3] == 0  # prior re-rank north toward enemy
    assert (out[1], out[2]) == (8, 5)


def test_constrain_forces_evacuate_when_nn_idles_structure():
    from tactics import apply_pre_contact_prior, constrain_nn_action, play_mask

    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 7] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    ownership[0, 6, 0] = True
    ownership[0, 5, 0] = True
    armies[7, 0] = 50
    armies[6, 0] = 2
    armies[5, 0] = 2
    state = state._replace(armies=armies, ownership=ownership, time=10)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(8, 8), obs)
    mask = play_mask(obs, mem)
    prior = np.full(mask.shape, 1.0 / max(int(mask.sum()), 1), dtype=np.float64)
    shaped = apply_pre_contact_prior(prior, obs, mem, mask=mask)
    # Tip thrash between own cells while the king stack sits.
    nn_idle = (0, 5, 0, 1, 0)  # (5,0) → south toward (6,0)
    out = constrain_nn_action(obs, mem, nn_idle, prev_action=None, prior=shaped)
    assert (out[1], out[2]) == (7, 0)


def test_pre_contact_prior_prefers_frontier_over_pass_and_reinforce():
    obs, mem = _corridor_obs()
    mask = play_mask(obs, mem)
    prior = np.zeros(mask.shape, dtype=np.float64)
    leave_gen_own = None
    frontier_side = None
    owners = np.asarray(obs.owner_grid)
    gen = np.asarray(mem.own_general)
    for idx in np.flatnonzero(legal_mask(obs, mem)):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        action = decode_action(idx)
        if action[0] != 0:
            continue
        tr = action[1] + int(DIRECTIONS[action[3], 0])
        tc = action[2] + int(DIRECTIONS[action[3], 1])
        if action[1] == 7 and action[2] == 0 and int(owners[tr, tc]) == 1:
            leave_gen_own = idx
            prior[idx] = 0.7
        if action[1] == 6 and action[2] == 0 and int(owners[tr, tc]) == 0:
            frontier_side = idx
            prior[idx] = 0.05
    assert frontier_side is not None
    prior[PASS_INDEX] = 0.25
    prior = prior / prior.sum()

    shaped = apply_pre_contact_prior(prior, obs, mem, mask=mask)
    assert shaped[PASS_INDEX] == pytest.approx(0.0)
    assert shaped[frontier_side] > 0.0
    # Moves onto the general stay illegal / zero under the play mask.
    for idx in np.flatnonzero(legal_mask(obs, mem)):
        action = decode_action(int(idx))
        if action[0] != 0:
            continue
        tr = action[1] + int(DIRECTIONS[action[3], 0])
        tc = action[2] + int(DIRECTIONS[action[3], 1])
        if gen[tr, tc]:
            assert shaped[int(idx)] == pytest.approx(0.0)
    top = int(np.argmax(shaped))
    top_action = decode_action(top)
    tr = top_action[1] + int(DIRECTIONS[top_action[3], 0])
    tc = top_action[2] + int(DIRECTIONS[top_action[3], 1])
    assert int(owners[tr, tc]) == 0
    del leave_gen_own


def test_path_progress_routes_around_mountains():
    """Manhattan can point into a wall; path progress must go around."""
    from tactics import path_distance_field, path_progress

    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 7] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    mountains = np.asarray(state.mountains, dtype=bool).copy()
    for r in range(2, 6):
        for c in range(0, 3):
            ownership[0, r, c] = True
            neut[r, c] = False
            armies[r, c] = 2
    armies[4, 1] = 80
    mountains[2:6, 3] = True
    ownership[1, 4, 5] = True
    neut[4, 5] = False
    armies[4, 5] = 3
    state = state._replace(
        armies=armies,
        ownership=ownership,
        ownership_neutral=neut,
        mountains=mountains,
        time=80,
    )
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(8, 8), obs)
    dist = path_distance_field(obs, [(4, 5)])
    # South shortens the true path; west lengthens it.
    assert path_progress(4, 1, 5, 1, dist) > 0.0
    assert path_progress(4, 1, 4, 0, dist) < 0.0
    del mem


def test_structure_idle_detects_fat_general_pile():
    from tactics import STRUCTURE_IDLE_ARMY, structure_idle_army

    grid = np.zeros((8, 8), dtype=np.int32)
    grid[7, 0] = 1
    grid[0, 7] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    ownership[0, 6, 0] = True
    ownership[0, 5, 0] = True
    armies[7, 0] = 40  # fat pile on general
    armies[6, 0] = 1
    armies[5, 0] = 1
    state = state._replace(armies=armies, ownership=ownership, time=8)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(8, 8), obs)
    assert structure_idle_army(obs, mem) >= STRUCTURE_IDLE_ARMY
