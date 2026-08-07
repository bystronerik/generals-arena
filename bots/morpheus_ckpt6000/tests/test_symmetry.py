"""Part 03 — dihedral symmetry round-trips."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from _common.wire import Observation
from action import N_ACTIONS, PASS_INDEX, decode_action, encode_action
from memory import empty_memory
from symmetry import (
    all_symmetry_names,
    get_symmetry,
    inverse_symmetry,
    transform_action_tuple,
    transform_policy,
    transform_tensor,
)
from tensor import observation_tensor


def _square_obs():
    """Fully interior content on a 21×21 board so every symmetry stays in-bounds."""
    H = W = 21
    types = np.ones((H, W), dtype=np.int32)
    owners = np.zeros((H, W), dtype=np.int32)
    armies = np.zeros((H, W), dtype=np.int32)
    types[5, 5] = 4
    owners[5, 5] = 1
    armies[5, 5] = 12
    types[5, 6] = 1
    owners[5, 6] = 1
    armies[5, 6] = 8
    types[10, 10] = 2
    types[8, 8] = 3
    owners[8, 8] = 2
    armies[8, 8] = 20
    types[15, 15] = 0
    return Observation(
        H=H,
        W=W,
        turn=42,
        my_land=2,
        my_army=20,
        opp_land=1,
        opp_army=20,
        type_grid=types.tolist(),
        owner_grid=owners.tolist(),
        army_grid=armies.tolist(),
    )


def test_eight_symmetries_named():
    names = all_symmetry_names()
    assert len(names) == 8
    assert "id" in names and "rot90" in names


def test_tensor_symmetry_round_trip():
    obs = _square_obs()
    prev = (0, 5, 5, 3, 0)  # right from general
    tensor, _ = observation_tensor(obs, empty_memory(21, 21), previous_action=prev)
    for name in all_symmetry_names():
        sym = get_symmetry(name)
        inv = inverse_symmetry(name)
        rotated = transform_tensor(tensor, sym)
        restored = transform_tensor(rotated, inv)
        np.testing.assert_allclose(restored, tensor, atol=1e-6, err_msg=name)


def test_policy_and_action_round_trip():
    policy = np.arange(N_ACTIONS, dtype=np.float32)
    actions = [
        (1, 0, 0, 0, 0),
        (0, 5, 5, 0, 0),
        (0, 5, 5, 3, 1),
        (2, 8, 8, 0, 0),
        (0, 10, 12, 2, 0),
    ]
    for name in all_symmetry_names():
        sym = get_symmetry(name)
        inv = inverse_symmetry(name)
        for action in actions:
            mid = transform_action_tuple(action, sym)
            back = transform_action_tuple(mid, inv)
            assert back == action, (name, action, mid, back)
        mid_p = transform_policy(policy, sym)
        back_p = transform_policy(mid_p, inv)
        np.testing.assert_array_equal(back_p, policy, err_msg=name)
        assert mid_p[PASS_INDEX] == policy[PASS_INDEX]


def test_pass_channel_fixed():
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    policy[PASS_INDEX] = 7.0
    policy[encode_action((0, 1, 1, 0, 0))] = 3.0
    for name in all_symmetry_names():
        out = transform_policy(policy, get_symmetry(name))
        assert out[PASS_INDEX] == 7.0


def test_direction_follows_destination():
    sym = get_symmetry("rot90")
    # Move up from (10, 10) → dest (9, 10). After rot90 source (10, 10)→(10, 10)?
    # rot90: (r,c)->(c, PAD-1-r); (10,10)->(10, 10). up dest (9,10)->(10, 11).
    # New direction should be right.
    action = (0, 10, 10, 0, 0)
    got = transform_action_tuple(action, sym)
    assert got == (0, 10, 10, 3, 0)
    sr, sc = got[1], got[2]
    dr = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[got[3]]
    dest = (sr + dr[0], sc + dr[1])
    assert dest == (10, 11)
