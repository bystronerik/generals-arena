"""prune_by_clock: wave closes on the objective, gather feeds the muster.

One corridor row gives a unique BFS direction, so "east" always closes on the
objective and "west" never does.
"""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import T_GENERAL, T_PLAIN, corridor, make_obs

ROW = 1
SRC = (ROW, 2)
EAST = (ROW, 3)
WEST = (ROW, 1)
GOAL = (ROW, 6)
SRC_ARMY = 10


def _scene(dst, dst_owner, dst_army, src_type=T_PLAIN):
    types, owner, army = corridor(3, 7, row=ROW)
    types[SRC[0]][SRC[1]] = src_type
    owner[SRC[0]][SRC[1]] = 1
    army[SRC[0]][SRC[1]] = SRC_ARMY
    owner[dst[0]][dst[1]] = dst_owner
    army[dst[0]][dst[1]] = dst_army
    return make_obs(types, owner, army, turn=100)


@pytest.mark.parametrize(
    "label, dst, dst_owner, dst_army, expect_kept",
    [
        ("own_step_closes", EAST, 1, 1, True),
        ("own_step_away", WEST, 1, 1, False),
        ("enemy_win_off_path", WEST, 2, 3, True),
        ("neutral_off_path", WEST, 0, 0, False),
        ("enemy_loss_off_path", WEST, 2, SRC_ARMY, False),
    ],
)
def test_wave_keeps_closing_and_winning_captures(
    label, dst, dst_owner, dst_army, expect_kept
):
    with sosipolis_imports():
        from components.army import move_action
        from components.conveyor import prune_by_clock

        obs = _scene(dst, dst_owner, dst_army)
        action = move_action(SRC[0], SRC[1], dst[0], dst[1], 0)
        kept = prune_by_clock(obs, [(action, 1.0)], "wave", GOAL, None)
        assert bool(kept) is expect_kept, label


@pytest.mark.parametrize(
    "label, dst, dst_owner, src_type, muster, expect_kept",
    [
        ("own_closes_on_muster", EAST, 1, T_PLAIN, GOAL, True),
        ("own_away_from_muster", WEST, 1, T_PLAIN, GOAL, False),
        ("structure_drain_away", WEST, 1, T_GENERAL, GOAL, True),
        ("capture_never_kept", EAST, 2, T_GENERAL, GOAL, False),
        ("no_muster_keeps_own", WEST, 1, T_PLAIN, None, True),
    ],
)
def test_gather_stays_on_own_land_toward_muster(
    label, dst, dst_owner, src_type, muster, expect_kept
):
    with sosipolis_imports():
        from components.army import move_action
        from components.conveyor import prune_by_clock

        obs = _scene(dst, dst_owner, 1, src_type=src_type)
        action = move_action(SRC[0], SRC[1], dst[0], dst[1], 0)
        kept = prune_by_clock(obs, [(action, 1.0)], "gather", GOAL, muster)
        assert bool(kept) is expect_kept, label
