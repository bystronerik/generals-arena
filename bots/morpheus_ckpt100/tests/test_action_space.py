"""Part 03 — action codec, legal masks, transition agreement."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from _common.wire import Observation
from action import (
    N_ACTIONS,
    PASS_INDEX,
    action_effects_match_pass,
    decode_action,
    encode_action,
    legal_mask,
    live_build_cost,
)
from memory import empty_memory, update_memory
from state import GameState, create_initial_state


def _obs_from_state_perspective(state: GameState, player: int) -> Observation:
    """Build a fully visible perspective observation (test helper)."""
    H, W = state.armies.shape
    types = np.ones((H, W), dtype=np.int32)
    types[state.mountains] = 2
    types[state.castles] = 3
    types[state.generals] = 4
    owners = np.zeros((H, W), dtype=np.int32)
    owners[state.ownership[player]] = 1
    owners[state.ownership[1 - player]] = 2
    armies = state.armies.copy()
    my_land = int(state.ownership[player].sum())
    opp_land = int(state.ownership[1 - player].sum())
    my_army = int((state.armies * state.ownership[player]).sum())
    opp_army = int((state.armies * state.ownership[1 - player]).sum())
    return Observation(
        H=H,
        W=W,
        turn=int(state.time),
        my_land=my_land,
        my_army=my_army,
        opp_land=opp_land,
        opp_army=opp_army,
        type_grid=types.tolist(),
        owner_grid=owners.tolist(),
        army_grid=armies.tolist(),
    )


def test_codec_round_trip_and_pass():
    assert N_ACTIONS == 3970
    assert encode_action((1, 0, 0, 0, 0)) == PASS_INDEX
    assert decode_action(PASS_INDEX) == (1, 0, 0, 0, 0)
    samples = [
        (0, 3, 4, 0, 0),
        (0, 3, 4, 1, 1),
        (0, 0, 0, 3, 0),
        (2, 5, 6, 0, 0),
        (0, 20, 20, 2, 1),
    ]
    for action in samples:
        assert decode_action(encode_action(action)) == action


def test_half_masked_when_army_is_two():
    grid = np.zeros((6, 6), dtype=np.int32)
    grid[0, 0] = 1
    grid[0, 5] = 2
    state = create_initial_state(grid)
    # Give player 0 a size-2 stack with a clear right move.
    armies = state.armies.copy()
    ownership = state.ownership.copy()
    ownership_neutral = state.ownership_neutral.copy()
    armies[2, 2] = 2
    ownership[0, 2, 2] = True
    ownership_neutral[2, 2] = False
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=ownership_neutral
    )
    obs = _obs_from_state_perspective(state, 0)
    mem = update_memory(empty_memory(obs.H, obs.W), obs)
    mask = legal_mask(obs, mem)
    full = encode_action((0, 2, 2, 3, 0))
    half = encode_action((0, 2, 2, 3, 1))
    assert mask[full]
    assert not mask[half]
    assert mask[PASS_INDEX]


def test_mountain_and_structure_fog_destinations_masked():
    H, W = 4, 4
    types = [
        [4, 2, 1, 1],
        [1, 5, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 4],
    ]
    owners = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 2],
    ]
    armies = [
        [5, 0, 0, 0],
        [5, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]
    obs = Observation(
        H=H,
        W=W,
        turn=0,
        my_land=2,
        my_army=10,
        opp_land=1,
        opp_army=1,
        type_grid=types,
        owner_grid=owners,
        army_grid=armies,
    )
    mem = update_memory(empty_memory(H, W), obs)
    mask = legal_mask(obs, mem)
    # From (0,0) right onto mountain — illegal.
    assert not mask[encode_action((0, 0, 0, 3, 0))]
    # From (1,0) right onto type 5 — illegal.
    assert not mask[encode_action((0, 1, 0, 3, 0))]
    # From (1,0) down onto plain — legal.
    assert mask[encode_action((0, 1, 0, 1, 0))]


def test_build_uses_exact_live_cost():
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[0, 0] = 1
    grid[0, 9] = 2
    state = create_initial_state(grid)
    armies = state.armies.copy()
    ownership = state.ownership.copy()
    ownership_neutral = state.ownership_neutral.copy()
    # Far cell with exactly base cost 35.
    armies[8, 8] = 35
    ownership[0, 8, 8] = True
    ownership_neutral[8, 8] = False
    # Near general: cost > 35, give only 35 army → illegal.
    armies[0, 1] = 35
    ownership[0, 0, 1] = True
    ownership_neutral[0, 1] = False
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=ownership_neutral
    )
    obs = _obs_from_state_perspective(state, 0)
    mem = update_memory(empty_memory(obs.H, obs.W), obs)
    costs = live_build_cost(obs, mem)
    assert int(costs[8, 8]) == 35
    assert int(costs[0, 1]) > 35
    mask = legal_mask(obs, mem, cost_grid=costs)
    assert mask[encode_action((2, 8, 8, 0, 0))]
    assert not mask[encode_action((2, 0, 1, 0, 0))]


def test_unmasked_actions_effect_under_transition():
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[0, 0] = 1
    grid[0, 7] = 2
    state = create_initial_state(grid)
    armies = state.armies.copy()
    ownership = state.ownership.copy()
    ownership_neutral = state.ownership_neutral.copy()
    armies[3, 3] = 10
    ownership[0, 3, 3] = True
    ownership_neutral[3, 3] = False
    armies[6, 6] = 40
    ownership[0, 6, 6] = True
    ownership_neutral[6, 6] = False
    state = state._replace(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        time=0,
    )
    obs = _obs_from_state_perspective(state, 0)
    mem = update_memory(empty_memory(obs.H, obs.W), obs)
    mask = legal_mask(obs, mem)

    # Every unmasked non-pass action must change the board vs double-pass.
    for idx in np.flatnonzero(mask):
        action = decode_action(int(idx))
        if action[0] == 1:
            assert action_effects_match_pass(state, 0, action)
            continue
        assert not action_effects_match_pass(state, 0, action), action


def test_every_legal_wire_action_is_reachable():
    grid = np.zeros((6, 6), dtype=np.int32)
    grid[1, 1] = 1
    grid[1, 4] = 2
    state = create_initial_state(grid)
    armies = state.armies.copy()
    ownership = state.ownership.copy()
    ownership_neutral = state.ownership_neutral.copy()
    armies[2, 2] = 5
    ownership[0, 2, 2] = True
    ownership_neutral[2, 2] = False
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=ownership_neutral
    )
    obs = _obs_from_state_perspective(state, 0)
    mem = update_memory(empty_memory(obs.H, obs.W), obs)
    mask = legal_mask(obs, mem)

    # Enumerate protocol-legal moves/builds from the observation rules and
    # require each to appear in the mask.
    H, W = obs.H, obs.W
    types = np.asarray(obs.type_grid)
    owners = np.asarray(obs.owner_grid)
    armies_g = np.asarray(obs.army_grid)
    costs = live_build_cost(obs, mem)
    expected = {PASS_INDEX}
    for r in range(H):
        for c in range(W):
            if owners[r, c] != 1:
                continue
            if armies_g[r, c] >= 2:
                for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if types[nr, nc] in (2, 5):
                        continue
                    expected.add(encode_action((0, r, c, d, 0)))
                    if armies_g[r, c] > 2:
                        expected.add(encode_action((0, r, c, d, 1)))
            if (
                mem.known_passable_base[r, c]
                and not mem.known_castle[r, c]
                and not mem.own_general[r, c]
                and not mem.known_mountain[r, c]
                and armies_g[r, c] >= costs[r, c]
            ):
                expected.add(encode_action((2, r, c, 0, 0)))
    assert expected == set(np.flatnonzero(mask).tolist())
