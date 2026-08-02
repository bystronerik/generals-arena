"""tip_is_ready and tip_below_sight_floor thresholds.

The goal is one hop away, so the path cost stays small and STRIKE_MIN_TIP is
the term that decides readiness.
"""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import corridor, make_obs

ROW = 1
TIP = (ROW, 0)
GOAL = (ROW, 1)


def _obs(tip_army):
    types, owner, army = corridor(3, 4, row=ROW)
    owner[TIP[0]][TIP[1]] = 1
    army[TIP[0]][TIP[1]] = tip_army
    return make_obs(types, owner, army, turn=200)


def test_min_tip_dominates_short_path():
    with sosipolis_imports():
        from components.tip import path_finish_need, tip_mass_target
        from params import PARAMS

        obs = _obs(5)
        assert path_finish_need(obs, TIP, GOAL, PARAMS) < PARAMS.STRIKE_MIN_TIP
        assert tip_mass_target(obs, TIP, GOAL, PARAMS) == PARAMS.STRIKE_MIN_TIP


@pytest.mark.parametrize(
    "label, delta, expect_ready",
    [
        ("one_short", 0, False),
        ("exact", 1, True),
        ("surplus", 10, True),
    ],
)
def test_tip_is_ready_at_min_tip_boundary(label, delta, expect_ready):
    with sosipolis_imports():
        from components.tip import tip_is_ready
        from params import PARAMS

        army = PARAMS.STRIKE_MIN_TIP + delta
        obs = _obs(army)
        assert tip_is_ready(obs, TIP, GOAL, PARAMS) is expect_ready, label


def test_tip_is_ready_rejects_immobile_and_missing_tip():
    with sosipolis_imports():
        from components.tip import tip_is_ready
        from params import PARAMS

        assert tip_is_ready(_obs(1), TIP, GOAL, PARAMS) is False
        assert tip_is_ready(_obs(30), None, GOAL, PARAMS) is False


@pytest.mark.parametrize(
    "label, delta, expect_below",
    [
        ("far_below", -9, True),
        ("one_below", -1, True),
        ("at_floor", 0, False),
        ("above_floor", 20, False),
    ],
)
def test_tip_below_sight_floor(label, delta, expect_below):
    with sosipolis_imports():
        from components.tip import tip_below_sight_floor
        from params import PARAMS

        obs = _obs(PARAMS.TIP_AT_SIGHT_FLOOR + delta)
        assert tip_below_sight_floor(obs, TIP, PARAMS) is expect_below, label


def test_missing_tip_is_below_sight_floor():
    with sosipolis_imports():
        from components.tip import tip_below_sight_floor
        from params import PARAMS

        assert tip_below_sight_floor(_obs(30), None, PARAMS) is True
