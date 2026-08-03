"""Marching cost: what a step takes out of the stack that makes it.

Hop distance treats a snake lying across the route as free. It is not: each
enemy tile costs the unit left behind plus the garrison, which is what drained
the assault tip to nothing on the way to the general.
"""
from __future__ import annotations

from _imports import sosipolis_imports
from boards import T_MOUNTAIN, corridor, make_obs, plain


def _board(H, W, *, own=(), neutral=(), enemy=()):
    types, owner, army = plain(H, W)
    for (r, c), a in own:
        owner[r][c] = 1
        army[r][c] = a
    for r, c in neutral:
        owner[r][c] = 0
        army[r][c] = 0
    for (r, c), a in enemy:
        owner[r][c] = 2
        army[r][c] = a
    return types, owner, army


def _blocked(obs):
    from components.army import is_wall

    return lambda r, c: is_wall(obs.type_grid, r, c)


# ---------------------------------------------------------------------------
# per-step cost
# ---------------------------------------------------------------------------


def test_march_cost_by_tile_owner():
    with sosipolis_imports():
        from components.army import march_cost

        types, owner, army = _board(
            5, 5, own=[((0, 0), 9)], neutral=[(0, 1)], enemy=[((0, 2), 4)]
        )
        obs = make_obs(types, owner, army, turn=100)

        assert march_cost(obs, 0, 0, 12) == 0  # own land is absorbed
        assert march_cost(obs, 0, 1, 12) == 1  # the unit left behind
        assert march_cost(obs, 0, 2, 12) == 5  # that, plus the garrison


def test_march_cost_caps_one_garrison():
    with sosipolis_imports():
        from components.army import march_cost

        types, owner, army = _board(5, 5, enemy=[((0, 2), 40)])
        obs = make_obs(types, owner, army, turn=100)

        assert march_cost(obs, 0, 2, 3) == 4
        assert march_cost(obs, 0, 2, 12) == 13


# ---------------------------------------------------------------------------
# route choice
# ---------------------------------------------------------------------------


def test_march_dist_prefers_walking_round_a_snake():
    """Two routes of equal length; the one through enemy tiles costs more.

    Row 1 is ours all the way. Row 3 is theirs. Both reach column 5 in the
    same number of hops from `(2,0)`, so hop distance cannot tell them apart.
    """
    with sosipolis_imports():
        from components.army import march_dist

        own = [((1, c), 1) for c in range(6)] + [((2, 0), 30)]
        enemy = [((3, c), 2) for c in range(6)]
        types, owner, army = _board(6, 7, own=own, enemy=enemy)
        obs = make_obs(types, owner, army, turn=100)

        cost = march_dist(obs, [(1, 6)], _blocked(obs), 3)
        # Stepping up into our own row is cheaper than stepping down into theirs.
        assert cost[(1, 3)] < cost[(3, 3)]


def test_march_dist_still_reaches_a_target_behind_enemy_land():
    """Cost is a price, not a wall — the far side stays reachable."""
    with sosipolis_imports():
        from components.army import march_dist

        enemy = [((r, 3), 5) for r in range(6)]
        types, owner, army = _board(6, 7, enemy=enemy)
        obs = make_obs(types, owner, army, turn=100)

        cost = march_dist(obs, [(0, 6)], _blocked(obs), 3)
        assert (0, 0) in cost
        assert cost[(0, 0)] > cost[(0, 4)]


def test_march_dist_respects_walls_and_reports_unreachable():
    with sosipolis_imports():
        from components.army import march_dist

        types, owner, army = corridor(5, 7, row=2)
        obs = make_obs(types, owner, army, turn=100)

        cost = march_dist(obs, [(2, 0)], _blocked(obs), 3)
        assert (2, 6) in cost
        assert (0, 0) not in cost  # mountain rows never enter the map
        assert obs.type_grid[0][0] == T_MOUNTAIN


def test_march_dist_grows_along_a_neutral_corridor():
    with sosipolis_imports():
        from components.army import march_dist

        types, owner, army = corridor(5, 7, row=2)
        obs = make_obs(types, owner, army, turn=100)

        cost = march_dist(obs, [(2, 0)], _blocked(obs), 3)
        # Neutral plain: one for the hop, one for the unit left behind.
        assert cost[(2, 1)] == 2
        assert cost[(2, 2)] == 4
