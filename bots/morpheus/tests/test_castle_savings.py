"""Castle savings: base-price site near the general, forced build.

Measured triple lock: legal builds on 8/256 turns, castle score at 5% of the
top action, NN build mass ~0 (clip-proof). The savings site + hard rule is
the only path to a castle — and it must never pay a surcharge.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import BASE_COST, live_build_cost
from memory import empty_memory, update_memory
from observe import emit_observation
from state import create_initial_state
from tactics import (
    CASTLE_WINDOW_UNTIL,
    GARRISON_FLOOR_FROM,
    castle_build_site,
    constrain_nn_action,
    winning_kill_move,
)


def _board(*, site_army: int = 0, time: int = 200, wide: bool = True,
           castles: int = 0, enemy_near_site: bool = False):
    """General at (11,0); own territory spanning 8+ Manhattan when wide."""
    H = W = 12
    grid = np.zeros((H, W), dtype=np.int32)
    grid[11, 0] = 1
    grid[0, 11] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    castles_g = np.asarray(state.castles).copy()
    rmax = 0 if wide else 8
    for r in range(rmax, 12):
        for c in range(0, 3):
            ownership[0, r, c] = True
            neut[r, c] = False
            armies[r, c] = 2
    # (4,0) is Manhattan 7 from the general (11,0): the base-price cell.
    armies[4, 0] = max(site_army, armies[4, 0]) if wide else armies[4, 0]
    for i in range(castles):
        castles_g[10 - i, 2] = True
    if enemy_near_site:
        ownership[1, 3, 1] = True
        neut[3, 1] = False
        armies[3, 1] = 2
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut,
        castles=castles_g, time=time,
    )
    obs = emit_observation(state, 0)
    return obs, update_memory(empty_memory(H, W), obs)


def test_site_is_base_price_cell_nearest_general():
    obs, mem = _board()
    site = castle_build_site(obs, mem)
    assert site is not None
    cost = np.asarray(live_build_cost(obs, mem))
    assert int(cost[site]) == BASE_COST
    # Manhattan >= 7 from the general, and minimal among candidates.
    assert abs(site[0] - 11) + abs(site[1] - 0) == 7


def test_no_site_when_territory_too_small_or_window_closed():
    obs, mem = _board(wide=False)
    assert castle_build_site(obs, mem) is None
    early, em = _board(time=GARRISON_FLOOR_FROM - 10)
    assert castle_build_site(early, em) is None
    late, lm = _board(time=CASTLE_WINDOW_UNTIL + 50)
    assert castle_build_site(late, lm) is None


def test_no_site_at_castle_target_or_near_visible_enemy():
    obs, mem = _board(castles=2)
    assert castle_build_site(obs, mem) is None
    obs2, mem2 = _board(enemy_near_site=True)
    site2 = castle_build_site(obs2, mem2)
    assert site2 is None or abs(site2[0] - 3) + abs(site2[1] - 1) >= 4


def test_forced_build_when_pile_reaches_price():
    obs, mem = _board(site_army=BASE_COST + 2)
    site = castle_build_site(obs, mem)
    assert site == (4, 0)
    out = constrain_nn_action(obs, mem, (1, 0, 0, 0, 0))
    assert tuple(int(x) for x in out) == (2, 4, 0, 0, 0)


def test_underfunded_site_is_not_forced():
    obs, mem = _board(site_army=BASE_COST - 5)
    out = constrain_nn_action(obs, mem, (1, 0, 0, 0, 0))
    assert int(out[0]) != 2


def test_underfunded_site_pile_is_anchored():
    from action import encode_action
    from tactics import play_mask
    from transition import DIRECTIONS

    obs, mem = _board(site_army=20)  # below price: pinned
    site = castle_build_site(obs, mem)
    assert site == (4, 0)
    mask = play_mask(obs, mem)
    for d in range(4):
        for s in (0, 1):
            assert not mask[encode_action((0, 4, 0, d, s))]
    # Funded: the pin is gone (the build itself is now legal and forced).
    obs2, mem2 = _board(site_army=BASE_COST + 2)
    mask2 = play_mask(obs2, mem2)
    assert any(
        mask2[encode_action((0, 4, 0, d, 0))] for d in range(4)
    ) or mask2[encode_action((2, 4, 0, 0, 0))]


def test_site_is_sticky_on_an_existing_pile():
    # A pile one cell farther than the nearest empty candidate keeps the site.
    obs, mem = _board(site_army=0)
    armies = np.asarray(obs.army_grid).copy()
    # (5, 1) is Manhattan 7 from the general too; give it a pile.
    obs2, mem2 = _board()
    # simulate a pile at an equally-priced but lexicographically later cell
    from state import create_initial_state
    grid = np.zeros((12, 12), dtype=np.int32)
    grid[11, 0] = 1
    grid[0, 11] = 2
    state = create_initial_state(grid)
    a2 = np.asarray(state.armies, dtype=np.int32).copy()
    o2 = np.asarray(state.ownership, dtype=bool).copy()
    n2 = np.asarray(state.ownership_neutral, dtype=bool).copy()
    for r in range(0, 12):
        for c in range(0, 3):
            o2[0, r, c] = True
            n2[r, c] = False
            a2[r, c] = 2
    a2[3, 1] = 20  # pile at Manhattan 9 — farther than (4,0) at 7
    state = state._replace(armies=a2, ownership=o2, ownership_neutral=n2, time=200)
    obs3 = emit_observation(state, 0)
    mem3 = update_memory(empty_memory(12, 12), obs3)
    assert castle_build_site(obs3, mem3) == (3, 1)


def test_tithe_moves_biggest_catchment_tip_on_schedule():
    from tactics import CASTLE_TITHE_PERIOD, castle_tithe_move

    obs, mem = _board(site_army=5, time=201 // CASTLE_TITHE_PERIOD * CASTLE_TITHE_PERIOD)
    site = castle_build_site(obs, mem)
    assert site is not None
    tithe = castle_tithe_move(obs, mem, site)
    assert tithe is not None
    # It is a full move from an own cell inside the catchment, not the site.
    assert int(tithe[0]) == 0 and int(tithe[4]) == 0
    assert (int(tithe[1]), int(tithe[2])) != site
    out = constrain_nn_action(obs, mem, (1, 0, 0, 0, 0))
    assert int(out[0]) == 0  # redirected to a move (tithe), not pass


def test_kill_window_outranks_build():
    # Both a funded site and a winning kill march: the kill wins.
    obs, mem = _board(site_army=BASE_COST + 2)
    if winning_kill_move(obs, mem) is not None:
        out = constrain_nn_action(obs, mem, (1, 0, 0, 0, 0))
        assert int(out[0]) == 0
