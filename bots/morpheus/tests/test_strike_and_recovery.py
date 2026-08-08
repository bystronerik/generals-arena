"""Scoreboard helpers + the record of two failed experiments.

The forced strike march (three gating variants) and scoreboard expansion
recovery both failed the bad-change filter repeatedly (blitz 0-5 probes);
only ``opponent_mobile`` survives, for future sizing logic. The regression
test pins that scoring is scoreboard-independent again.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from memory import empty_memory, update_memory
from observe import emit_observation
from state import create_initial_state
from tactics import opponent_mobile


class _Obs:
    def __init__(self, opp_army, opp_land):
        self.opp_army = opp_army
        self.opp_land = opp_land


def test_opponent_mobile():
    assert opponent_mobile(_Obs(100, 40)) == 60
    assert opponent_mobile(_Obs(30, 40)) == 0


def _board():
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[9, 0] = 1
    grid[0, 9] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r in range(4, 10):
        ownership[0, r, 1] = True
        neut[r, 1] = False
        armies[r, 1] = 2
    armies[5, 1] = 10
    ownership[1, 3, 1] = True
    neut[3, 1] = False
    armies[3, 1] = 4
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut, time=300
    )
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(10, 10), obs)


def test_scoreboard_deficit_does_not_change_scores():
    from tactics import heuristic_action_scores, play_mask

    obs, mem = _board()
    mask = play_mask(obs, mem)
    base = heuristic_action_scores(obs, mem, mask)

    class _Deficit:
        def __init__(self, inner):
            self._inner = inner
            self.opp_land = 200

        def __getattr__(self, k):
            return getattr(self._inner, k)

    scores = heuristic_action_scores(_Deficit(obs), mem, mask)
    np.testing.assert_allclose(scores, base)
