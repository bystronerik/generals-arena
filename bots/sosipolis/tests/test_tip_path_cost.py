"""path_finish_need on corridor boards with one forced route.

Expected costs are composed from params constants, never from literals:
base (deathtouch or margin plus defender), per-cell path charges, the path
buffer per hop, and regen slack every second hop.
"""
from __future__ import annotations

import pytest

from _imports import sosipolis_imports
from boards import T_GENERAL, T_MOUNTAIN, corridor, make_obs

ROW = 1
H, W = 3, 7
TIP = (ROW, 0)
DEFENDER = 7
BLOCKER = 3


def _clean_neutral():
    """Tip walks 4 neutral hops to a neutral goal."""
    types, owner, army = corridor(H, W, row=ROW)
    return make_obs(types, owner, army, turn=100), TIP, (ROW, 4)


def _contested(turn):
    """One enemy cell, one own cell, one neutral cell, then the enemy general."""
    types, owner, army = corridor(H, W, row=ROW)
    owner[ROW][1] = 2
    army[ROW][1] = 3
    owner[ROW][2] = 1
    army[ROW][2] = 1
    types[ROW][4] = T_GENERAL
    owner[ROW][4] = 2
    army[ROW][4] = DEFENDER
    return make_obs(types, owner, army, turn=turn), TIP, (ROW, 4)


def _unreachable():
    """A mountain splits the corridor, so the BFS fallback prices the gap."""
    types, owner, army = corridor(H, W, row=ROW)
    types[ROW][BLOCKER] = T_MOUNTAIN
    return make_obs(types, owner, army, turn=100), TIP, (ROW, 5)


def _hop_cost(params, hops):
    return params.STRIKE_PATH_BUFFER * hops + params.STRIKE_REGEN_SLACK * (hops // 2)


SCENES = {
    "clean_neutral": (
        _clean_neutral,
        # 3 neutral cells charged at 1 each; the goal sits in base.
        lambda p: p.FINISH_MARGIN + 3 + _hop_cost(p, 4),
    ),
    "contested_path": (
        lambda: _contested(100),
        # enemy cell 3+1, own cell free, neutral cell 1.
        lambda p: p.FINISH_MARGIN + DEFENDER + (3 + 1) + 1 + _hop_cost(p, 4),
    ),
    "deathtouch": (
        lambda: _contested(1000),
        lambda p: 1 + (3 + 1) + 1 + _hop_cost(p, 4),
    ),
    "unreachable": (
        _unreachable,
        # No path: base plus buffer over the H+W fallback distance.
        lambda p: p.FINISH_MARGIN + p.STRIKE_PATH_BUFFER * (H + W),
    ),
}


@pytest.mark.parametrize("label", sorted(SCENES))
def test_path_finish_need(label):
    with sosipolis_imports():
        from components.tip import path_finish_need
        from params import PARAMS

        build, expected = SCENES[label]
        obs, tip, goal = build()
        if label == "deathtouch":
            assert obs.turn >= PARAMS.DEATHTOUCH_TURN
        assert path_finish_need(obs, tip, goal, PARAMS) == expected(PARAMS), label
