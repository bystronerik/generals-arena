"""select_mass_tip hold rules: keep a substantial tip, else take the best mass.

Hold needs four facts at once: the cached cell is still owned and movable, the
cached army is at least max(STRIKE_MIN_TIP // 2, 12), the cached army is not
dominated (cached * 5 >= best * 4), and the hold is younger than
STRIKE_TIP_HOLD turns.
"""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import corridor, make_obs

ROW = 1
CACHED = (ROW, 1)
OTHER = (ROW, 3)
GOAL = (ROW, 8)
TURN = 200


def _min_hold(params):
    return max(params.STRIKE_MIN_TIP // 2, 12)


def _obs(cached_army, other_army, cached_owner=1):
    types, owner, army = corridor(3, 9, row=ROW)
    owner[CACHED[0]][CACHED[1]] = cached_owner
    army[CACHED[0]][CACHED[1]] = cached_army
    owner[OTHER[0]][OTHER[1]] = 1
    army[OTHER[0]][OTHER[1]] = other_army
    return make_obs(types, owner, army, turn=TURN)


@pytest.mark.parametrize(
    "label, cached_of, other_of, expired, cached_owner, expect_hold",
    [
        ("hold_when_comparable", lambda m: m + 8, lambda c: c + 2, False, 1, True),
        ("switch_when_dominated", lambda m: m + 8, lambda c: c * 2, False, 1, False),
        ("switch_below_min_hold", lambda m: m - 1, lambda c: c + 1, False, 1, False),
        ("switch_when_hold_expired", lambda m: m + 8, lambda c: c + 2, True, 1, False),
        ("switch_when_tip_lost", lambda m: m + 8, lambda c: 5, False, 2, False),
    ],
)
def test_select_mass_tip_hold(
    label, cached_of, other_of, expired, cached_owner, expect_hold
):
    with sosipolis_imports():
        from components.tip import select_mass_tip
        from params import PARAMS

        cached_army = cached_of(_min_hold(PARAMS))
        obs = _obs(cached_army, other_of(cached_army), cached_owner=cached_owner)
        age = PARAMS.STRIKE_TIP_HOLD if expired else 2
        tip = select_mass_tip(obs, GOAL, PARAMS, CACHED, TURN - age)
        assert tip == (CACHED if expect_hold else OTHER), label


@pytest.mark.parametrize(
    "label, cached",
    [("no_cache", None), ("stale_cache", CACHED)],
)
def test_select_mass_tip_none_without_movable_stacks(label, cached):
    with sosipolis_imports():
        from components.tip import select_mass_tip
        from params import PARAMS

        types, owner, army = corridor(3, 9, row=ROW)
        for cell in (CACHED, OTHER):
            owner[cell[0]][cell[1]] = 1
            army[cell[0]][cell[1]] = 1
        obs = make_obs(types, owner, army, turn=TURN)
        assert select_mass_tip(obs, GOAL, PARAMS, cached, TURN - 1) is None, label
