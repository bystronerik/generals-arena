"""build_cost: base price plus a per-structure surcharge that decays with distance."""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import T_CASTLE, T_GENERAL, make_obs, plain

SITE = (4, 4)


def _obs(structures=(), enemy_structures=()):
    types, owner, army = plain(9, 9)
    for (r, c), t in structures:
        types[r][c] = t
        owner[r][c] = 1
    for (r, c), t in enemy_structures:
        types[r][c] = t
        owner[r][c] = 2
    return make_obs(types, owner, army, turn=200)


def _surcharge(params, dist):
    return max(0, params.BUILD_SURCHARGE_CAP - params.BUILD_SURCHARGE_PER_STEP * dist)


@pytest.mark.parametrize(
    "label, dists",
    [
        ("no_structures", ()),
        ("one_adjacent", (1,)),
        ("one_mid_range", (3,)),
        ("one_out_of_range", (7,)),
        ("two_structures", (1, 3)),
    ],
)
def test_build_cost_from_explicit_structures(label, dists):
    with sosipolis_imports():
        from components.economy import build_cost
        from params import PARAMS

        structures = [(SITE[0], SITE[1] + d) for d in dists]
        obs = _obs()
        expected = PARAMS.BUILD_BASE_COST + sum(
            _surcharge(PARAMS, d) for d in dists
        )
        cost = build_cost(obs, SITE[0], SITE[1], PARAMS, structures)
        assert cost == expected, label


def test_out_of_range_structure_is_free():
    """The surcharge clamps at zero, so a distant structure costs nothing."""
    with sosipolis_imports():
        from components.economy import build_cost
        from params import PARAMS

        far = PARAMS.BUILD_SURCHARGE_CAP // PARAMS.BUILD_SURCHARGE_PER_STEP
        assert _surcharge(PARAMS, far) == 0
        structures = [(SITE[0], SITE[1] + far)]
        cost = build_cost(_obs(), SITE[0], SITE[1], PARAMS, structures)
        assert cost == PARAMS.BUILD_BASE_COST


def test_structures_none_derives_own_structures_from_board():
    """Own general and own castle count; enemy structures do not."""
    with sosipolis_imports():
        from components.economy import build_cost, own_structures
        from params import PARAMS

        own = [((SITE[0], SITE[1] - 1), T_GENERAL), ((SITE[0], SITE[1] + 1), T_CASTLE)]
        obs = _obs(
            structures=own,
            enemy_structures=(((SITE[0], SITE[1] - 2), T_GENERAL),),
        )
        assert sorted(own_structures(obs)) == sorted(cell for cell, _ in own)
        derived = build_cost(obs, SITE[0], SITE[1], PARAMS)
        explicit = build_cost(
            obs, SITE[0], SITE[1], PARAMS, [cell for cell, _ in own]
        )
        assert derived == explicit
        assert derived == PARAMS.BUILD_BASE_COST + 2 * _surcharge(PARAMS, 1)
