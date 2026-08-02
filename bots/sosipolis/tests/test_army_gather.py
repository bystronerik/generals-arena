"""gather_toward and step_toward on a single-row corridor.

gather_toward ranks sources by army * 100 - distance, so mass beats proximity
until the distance term can close a 1-army gap.
"""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import T_MOUNTAIN, action_dst, action_src, corridor, make_obs

ROW = 1
RALLY = (ROW, 0)
NEAR = (ROW, 1)
FAR = (ROW, 6)
WALL = (ROW, 3)


def _obs(stacks, wall=None, rally_owner=1):
    types, owner, army = corridor(3, 9, row=ROW)
    if wall is not None:
        types[wall[0]][wall[1]] = T_MOUNTAIN
    owner[RALLY[0]][RALLY[1]] = rally_owner
    army[RALLY[0]][RALLY[1]] = 10
    for (r, c), a, o in stacks:
        owner[r][c] = o
        army[r][c] = a
    return make_obs(types, owner, army, turn=150)


@pytest.mark.parametrize(
    "label, stacks, wall, expected",
    [
        ("far_mass_beats_near", ((NEAR, 2, 1), (FAR, 5, 1)), None, (FAR, (ROW, 5))),
        ("near_mass_beats_far", ((NEAR, 9, 1), (FAR, 5, 1)), None, (NEAR, RALLY)),
        ("only_rally_has_army", (), None, None),
        ("all_stacks_army_one", ((NEAR, 1, 1), (FAR, 1, 1)), None, None),
        ("stack_behind_wall", ((FAR, 5, 1),), WALL, None),
    ],
)
def test_gather_toward(label, stacks, wall, expected):
    with sosipolis_imports():
        from components.army import gather_toward, is_wall

        obs = _obs(stacks, wall=wall)
        blocked = lambda r, c: is_wall(obs.type_grid, r, c)
        action = gather_toward(obs, RALLY, blocked)
        if expected is None:
            assert action is None, label
        else:
            assert (action_src(action), action_dst(action)) == expected, label


@pytest.mark.parametrize(
    "label, src_army, src_owner, wall, expected",
    [
        ("steps_toward_goal", 5, 1, None, (FAR, (ROW, 5))),
        ("army_one_cannot_move", 1, 1, None, None),
        ("unowned_source", 5, 0, None, None),
        ("goal_behind_wall", 5, 1, WALL, None),
    ],
)
def test_step_toward(label, src_army, src_owner, wall, expected):
    with sosipolis_imports():
        from components.army import is_wall, step_toward

        obs = _obs(((FAR, src_army, src_owner),), wall=wall)
        blocked = lambda r, c: is_wall(obs.type_grid, r, c)
        action = step_toward(obs, FAR, RALLY, blocked)
        if expected is None:
            assert action is None, label
        else:
            assert (action_src(action), action_dst(action)) == expected, label
