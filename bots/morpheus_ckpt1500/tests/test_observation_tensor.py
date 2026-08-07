"""Part 03 — observation tensor plane contracts and memory updates."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from _common.wire import Observation
from memory import empty_memory, update_memory
from tensor import (
    ARMY_SCALE,
    N_PLANES,
    PAD,
    PLANE_NAMES,
    BeliefSummary,
    army_value,
    binary_entropy_bits,
    build_tensor,
    observation_tensor,
)


def _obs(
    *,
    H=4,
    W=5,
    turn=0,
    my_land=1,
    my_army=1,
    opp_land=1,
    opp_army=1,
    types=None,
    owners=None,
    armies=None,
):
    if types is None:
        types = [[1] * W for _ in range(H)]
        types[0][0] = 4
        types[0][W - 1] = 0  # fog
    if owners is None:
        owners = [[0] * W for _ in range(H)]
        owners[0][0] = 1
    if armies is None:
        armies = [[0] * W for _ in range(H)]
        armies[0][0] = 1
    return Observation(
        H=H,
        W=W,
        turn=turn,
        my_land=my_land,
        my_army=my_army,
        opp_land=opp_land,
        opp_army=opp_army,
        type_grid=types,
        owner_grid=owners,
        army_grid=armies,
    )


def test_plane_count_and_names():
    assert N_PLANES == 49
    assert len(PLANE_NAMES) == 49
    assert PLANE_NAMES[0] == "board_mask"
    assert PLANE_NAMES[48] == "belief_ess"


def test_army_value_endpoints():
    assert float(army_value(0)) == 0.0
    assert float(army_value(ARMY_SCALE)) == pytest.approx(1.0)
    assert float(army_value(ARMY_SCALE * 10)) == 1.0
    assert float(army_value(-5)) == 0.0


def test_padding_zero_and_board_mask():
    obs = _obs(H=3, W=4)
    mem = empty_memory(3, 4)
    tensor, mem = observation_tensor(obs, mem)
    assert tensor.shape == (49, PAD, PAD)
    assert np.all(tensor[:, 3:, :] == 0)
    assert np.all(tensor[:, :, 4:] == 0)
    assert np.all(tensor[0, :3, :4] == 1.0)
    assert np.all(tensor[0, 3:, :] == 0)
    assert np.all(tensor[0, :, 4:] == 0)


def test_first_frame_type5_is_mountain():
    H, W = 3, 3
    types = [
        [4, 5, 5],
        [5, 1, 5],
        [0, 5, 5],
    ]
    owners = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    armies = [
        [1, 0, 0],
        [0, 3, 0],
        [0, 0, 0],
    ]
    obs = _obs(H=H, W=W, types=types, owners=owners, armies=armies, my_land=2, my_army=4)
    mem = update_memory(empty_memory(H, W), obs)
    # Unseen type-5 cells are mountains; type 0 is passable base.
    assert mem.known_mountain[0, 1]
    assert mem.known_mountain[1, 0]
    assert not mem.known_castle[0, 1]
    assert mem.known_passable_base[2, 0]
    assert not mem.known_mountain[2, 0]
    assert mem.own_general[0, 0]


def test_type0_then_type5_marks_castle():
    H, W = 2, 2
    # First see fog type 0 at (0,1).
    obs1 = _obs(
        H=H,
        W=W,
        types=[[4, 0], [1, 1]],
        owners=[[1, 0], [1, 1]],
        armies=[[1, 0], [2, 2]],
        my_land=3,
        my_army=5,
    )
    mem = update_memory(empty_memory(H, W), obs1)
    assert mem.known_passable_base[0, 1]
    # Later structure-in-fog on that cell → castle.
    obs2 = _obs(
        H=H,
        W=W,
        turn=10,
        types=[[4, 5], [1, 1]],
        owners=[[1, 0], [1, 1]],
        armies=[[5, 0], [2, 2]],
        my_land=3,
        my_army=9,
    )
    mem = update_memory(mem, obs2)
    assert mem.known_castle[0, 1]
    assert not mem.known_mountain[0, 1]


def test_golden_planes_small_board():
    H, W = 3, 3
    types = [
        [4, 1, 2],
        [1, 0, 1],
        [3, 1, 5],
    ]
    owners = [
        [1, 1, 0],
        [1, 0, 2],
        [2, 0, 0],
    ]
    armies = [
        [10, 4, 0],
        [2, 0, 7],
        [9, 0, 0],
    ]
    obs = _obs(
        H=H,
        W=W,
        turn=100,
        my_land=3,
        my_army=16,
        opp_land=2,
        opp_army=16,
        types=types,
        owners=owners,
        armies=armies,
    )
    belief = BeliefSummary(
        enemy_owner=np.array(
            [[0, 0, 0], [0, 0.25, 1], [1, 0, 0]], dtype=np.float32
        ),
        enemy_army_mean=np.array(
            [[0, 0, 0], [0, 0, 7], [9, 0, 0]], dtype=np.float32
        ),
        enemy_army_std=np.array(
            [[0, 0, 0], [0, 0, 1], [2, 0, 0]], dtype=np.float32
        ),
        enemy_general=np.array(
            [[0, 0, 0], [0, 0, 0], [0.8, 0, 0]], dtype=np.float32
        ),
        enemy_castle_owner=np.zeros((H, W), dtype=np.float32),
        enemy_visibility=np.zeros((H, W), dtype=np.float32),
        ess_fraction=0.5,
    )
    prev = (0, 1, 0, 3, 0)  # all-but-one right from (1,0) → (1,1)
    mem = empty_memory(H, W)
    tensor, mem = observation_tensor(
        obs, mem, belief=belief, previous_action=prev
    )

    # board_mask
    assert tensor[0, 0, 0] == 1.0 and tensor[0, 2, 2] == 1.0 and tensor[0, 3, 0] == 0.0
    # visible_now: fog 0 and structure-fog 5 are invisible
    assert tensor[1, 1, 1] == 0.0
    assert tensor[1, 2, 2] == 0.0
    assert tensor[1, 0, 0] == 1.0
    # fog planes
    assert tensor[2, 1, 1] == 1.0
    assert tensor[3, 2, 2] == 1.0
    # mountain / castle / general
    assert tensor[4, 0, 2] == 1.0
    assert tensor[6, 2, 0] == 1.0
    assert tensor[7, 0, 0] == 1.0
    # owned / enemy / armies
    assert tensor[9, 0, 1] == 1.0
    assert tensor[10, 1, 2] == 1.0
    assert tensor[12, 0, 0] == pytest.approx(float(army_value(10)))
    assert tensor[13, 1, 2] == pytest.approx(float(army_value(7)))
    # belief mean/std transform-after-aggregate
    assert tensor[23, 1, 2] == pytest.approx(float(army_value(7)))
    assert tensor[24, 2, 0] == pytest.approx(float(army_value(2)))
    assert tensor[28, 1, 1] == pytest.approx(float(binary_entropy_bits(0.25)))
    # previous move
    assert tensor[29, 1, 0] == 1.0
    assert tensor[30, 1, 1] == 1.0
    assert tensor[31, 1, 1] == 1.0
    # constants on board only
    assert tensor[37, 0, 0] == pytest.approx(100 / 1200)
    assert tensor[37, 5, 5] == 0.0
    assert tensor[48, 1, 1] == pytest.approx(0.5)
    # growth helpers
    assert tensor[40, 0, 0] == 0.0  # (100+1) % 2 != 0
    assert tensor[41, 0, 0] == pytest.approx(((50 - ((100 + 1) % 50)) % 50) / 49)


def test_perspective_previous_action_is_seat_local():
    """Enemy-perspective tensors use the enemy's last action, not ours."""
    obs = _obs()
    mem = update_memory(empty_memory(obs.H, obs.W), obs)
    ours = (0, 0, 0, 1, 0)
    enemy = (2, 1, 1, 0, 0)
    t_ours = build_tensor(obs, mem, previous_action=ours)
    t_enemy = build_tensor(obs, mem, previous_action=enemy)
    assert t_ours[29, 0, 0] == 1.0
    assert t_enemy[29, 0, 0] == 0.0
    assert t_enemy[32, 1, 1] == 1.0


def test_zero_belief_default():
    obs = _obs()
    mem = empty_memory(obs.H, obs.W)
    tensor, _ = observation_tensor(obs, mem)
    assert np.all(tensor[22:29] == 0)
    assert np.all(tensor[48] == 0)
