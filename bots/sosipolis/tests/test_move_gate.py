"""legal_moves, bfs_dist, and has_leave1_move boundaries.

Leave-1 means a stack needs army > 1 to move, and mountains remove directions.
"""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import T_MOUNTAIN, T_PLAIN, corridor, grid, make_obs, plain

CORRIDOR_ROW = 1


def _boards(kind):
    if kind == "plain":
        return plain(3, 3)
    if kind == "corridor":
        return corridor(3, 5, row=CORRIDOR_ROW)
    if kind == "island":
        types = grid(3, 3, T_MOUNTAIN)
        types[1][1] = T_PLAIN
        return types, grid(3, 3), grid(3, 3)
    raise AssertionError(kind)


def _obs(kind, cell, army_value):
    types, owner, army = _boards(kind)
    owner[cell[0]][cell[1]] = 1
    army[cell[0]][cell[1]] = army_value
    return make_obs(types, owner, army, turn=100)


@pytest.mark.parametrize(
    "label, kind, cell, army_value, expected",
    [
        ("center_army_one_cannot_move", "plain", (1, 1), 1, 0),
        ("center_army_two_moves", "plain", (1, 1), 2, 4),
        ("corner_has_two_directions", "plain", (0, 0), 5, 2),
        ("corridor_walls_block_two", "corridor", (1, 1), 5, 2),
        ("walled_in_stack_has_none", "island", (1, 1), 5, 0),
    ],
)
def test_legal_moves_leave_one(label, kind, cell, army_value, expected):
    with sosipolis_imports():
        from components.army import legal_moves

        obs = _obs(kind, cell, army_value)
        assert len(legal_moves(obs, leave=1)) == expected, label


@pytest.mark.parametrize(
    "label, kind, cell, army_value, expected",
    [
        ("movable_stack", "plain", (1, 1), 2, True),
        ("army_one_only", "plain", (1, 1), 1, False),
        ("walled_in_stack", "island", (1, 1), 5, False),
        ("corridor_stack", "corridor", (1, 1), 5, True),
    ],
)
def test_has_leave1_move(label, kind, cell, army_value, expected):
    with sosipolis_imports():
        from components.conveyor import has_leave1_move

        obs = _obs(kind, cell, army_value)
        assert has_leave1_move(obs) is expected, label


@pytest.mark.parametrize(
    "label, kind, starts, expected_size, probes",
    [
        ("blocked_start_is_empty", "corridor", [(0, 0)], 0, {}),
        ("single_start_fills_plain", "plain", [(1, 1)], 9, {(0, 0): 2, (1, 1): 0}),
        ("multi_start_takes_minimum", "plain", [(0, 0), (2, 2)], 9, {(1, 1): 2, (2, 2): 0}),
        ("walls_shrink_reachable", "corridor", [(1, 0)], 5, {(1, 4): 4}),
    ],
)
def test_bfs_dist(label, kind, starts, expected_size, probes):
    with sosipolis_imports():
        from components.army import bfs_dist, is_wall

        obs = _obs(kind, (1, 1), 1)
        blocked = lambda r, c: is_wall(obs.type_grid, r, c)
        dist = bfs_dist(obs.H, obs.W, starts, blocked)
        assert len(dist) == expected_size, label
        for cell, d in probes.items():
            assert dist[cell] == d, (label, cell)
