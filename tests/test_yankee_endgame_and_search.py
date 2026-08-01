"""Offline tests for the two things yankee adds to proteus.

Both are pure: the endgame core is a function of one observation plus a
latched belief, and the search's forward model is a function of a board and
two moves. Nothing here starts an engine — which is the point, because what is
being checked is *agreement with RULES.md*, and an engine test would only show
that yankee agrees with itself.

The §07 cases are the ones worth having. Deathtouch inverts three defaults at
once (army on your own general stops defending it, the counter-attack has to
come from a third tile, a 2-stack outranks a 40-stack), and every one of them
is easy to write code that looks right and is exactly backwards.
"""
from __future__ import annotations

import pytest

from test_common_tactics import _grid, make_obs
from yankee.deathtouch import (
    DeathtouchConfig,
    DeathtouchMemory,
    chase_move,
    chase_ring,
    deathtouch_move,
    door_threats,
    touch_field,
    touch_move,
)
from yankee.params import TUNE_ENV, YankeeParams, load_params
from yankee.search import _Sim
from yankee.switcher import DEATHTOUCH, Switcher
from yankee.classifier import AGGRESSOR, ECONOMY, Classification


def board(H=7, W=7):
    return _grid(H, W, 1), _grid(H, W, 0), _grid(H, W, 0)


def obs_with(cells, turn=850, H=7, W=7):
    """`cells` maps (r, c) -> (type, owner, army)."""
    types, owner, army = board(H, W)
    for (r, c), (t, o, a) in cells.items():
        types[r][c], owner[r][c], army[r][c] = t, o, a
    return make_obs(types, owner, army, turn=turn)


def mem_for(obs, enemy_general=None):
    mem = DeathtouchMemory()
    mem.observe(obs)
    if enemy_general is not None:
        mem.belief.enemy_general = enemy_general
    return mem


# --------------------------------------------------------------- §07 the win
def test_touch_spends_the_smallest_stack_that_wins():
    """One unit is lethal, so spending the big stack is pure waste."""
    o = obs_with({
        (0, 0): (4, 1, 30),          # our general
        (3, 3): (4, 2, 99),          # theirs, garrisoned to the teeth
        (3, 2): (1, 1, 40),          # a fat neighbour
        (2, 3): (1, 1, 2),           # a thin one
    })
    mem = mem_for(o, enemy_general=(3, 3))
    move = touch_move(o, mem)
    assert move is not None
    assert (move[1], move[2]) == (2, 3), "should touch with the 2-stack"


def test_touch_ignores_a_one_army_neighbour():
    """A 1-stack moves nothing (§02 always leaves one behind)."""
    o = obs_with({
        (0, 0): (4, 1, 30),
        (3, 3): (4, 2, 5),
        (2, 3): (1, 1, 1),
    })
    assert touch_move(o, mem_for(o, enemy_general=(3, 3))) is None


# ----------------------------------------------------------- §07 the defence
def test_chase_comes_from_a_third_tile_and_never_the_general():
    """General-versus-source is a mutual chase the attacker wins."""
    o = obs_with({
        (3, 3): (4, 1, 50),          # our general, huge and irrelevant
        (3, 4): (1, 2, 6),           # their stack, one step from a touch
        (2, 4): (1, 1, 8),           # a third tile that can take it
    })
    mem = mem_for(o)
    move = chase_move(o, mem)
    assert move is not None
    assert (move[1], move[2]) == (2, 4)


def test_chase_refuses_a_tile_that_cannot_capture_the_source():
    """Reducing the source is not enough: it still sends one unit."""
    o = obs_with({
        (3, 3): (4, 1, 50),
        (3, 4): (1, 2, 6),
        (2, 4): (1, 1, 7),           # 7 - 1 == 6, a tie: defender holds
    })
    assert chase_move(o, mem_for(o)) is None


def test_door_threats_only_counts_stacks_that_can_move():
    o = obs_with({
        (3, 3): (4, 1, 5),
        (3, 4): (1, 2, 1),           # pinned, cannot touch
        (2, 3): (1, 2, 3),           # can
    })
    assert [cell for cell, _ in door_threats(o, (3, 3))] == [(2, 3)]


def test_chase_ring_is_the_distance_two_shell_and_excludes_home():
    o = obs_with({(3, 3): (4, 1, 5)})
    for cell in ((1, 3), (3, 1), (5, 3), (3, 5), (2, 4), (2, 2)):
        o.owner_grid[cell[0]][cell[1]] = 1
    ring = set(chase_ring(o, (3, 3)))
    assert (3, 3) not in ring
    assert {(1, 3), (3, 1), (5, 3), (3, 5), (2, 4), (2, 2)} <= ring
    # Neighbours of the general are not chase tiles for a threat *at* the
    # general — they are the cells the threat stands on.
    assert all(abs(r - 3) + abs(c - 3) == 2 for r, c in ring)


# ------------------------------------------------------------ §07 the routing
def test_touch_field_walks_own_and_empty_ground_only():
    """A 2-stack sends one unit, and §05 needs strictly more than the
    defender — so any enemy cell is a wall, however small."""
    types, owner, army = board()
    owner[0][0], types[0][0], army[0][0] = 1, 4, 5
    for c in range(7):                       # a wall of 1-army enemy cells
        owner[3][c], army[3][c] = 2, 1
    owner[6][6], types[6][6] = 2, 4          # their general, past the wall
    o = make_obs(types, owner, army, turn=850)

    field = touch_field(o, (6, 6))
    assert field[0][0] >= 1 << 20, "the enemy wall must not be walkable"
    assert field[6][5] == 1, "the target itself is always enterable"


# --------------------------------------------------------------- the switcher
def test_switcher_hands_over_on_the_clock_not_on_evidence():
    params = YankeeParams(deathtouch_enabled=True, deathtouch_from=800)
    s = Switcher.from_params(params, "blitz")
    assert s.update(Classification(AGGRESSOR, 1.0), 799) == "blitz"
    assert s.update(Classification(AGGRESSOR, 1.0), 800) == DEATHTOUCH
    # No hysteresis: a confident opposite verdict does not pull it back.
    assert s.update(Classification(ECONOMY, 1.0), 900) == DEATHTOUCH
    assert s.history[-1] == (800, AGGRESSOR, DEATHTOUCH)


def test_switcher_leaves_the_core_unreachable_when_disabled():
    s = Switcher.from_params(YankeeParams(deathtouch_enabled=False), "blitz")
    assert s.update(Classification(AGGRESSOR, 1.0), 1100) == "blitz"


def test_deathtouch_move_prefers_the_win_over_the_chase():
    """Both available: touching is at worst a mutual-touch draw (§07), and
    a chase only postpones."""
    o = obs_with({
        (3, 3): (4, 1, 9),
        (3, 4): (1, 2, 6),           # threatening our general
        (2, 4): (1, 1, 9),           # could chase
        (0, 6): (4, 2, 40),          # their general
        (0, 5): (1, 1, 2),           # ...and a 2-stack beside it
    })
    move = deathtouch_move(o, mem_for(o, enemy_general=(0, 6)), DeathtouchConfig())
    assert (move[1], move[2]) == (0, 5)


# ------------------------------------------------- the search's forward model
def sim_for(cells, seat, turn=100, H=5, W=5):
    types, owner, army = _grid(H, W, 1), _grid(H, W, 0), _grid(H, W, 0)
    for (r, c), (t, o, a) in cells.items():
        types[r][c], owner[r][c], army[r][c] = t, o, a
    sim = _Sim(make_obs(types, owner, army, turn=turn), seat, 800)
    return sim


def test_sim_combat_needs_strictly_more_army():
    """§05: an exact tie leaves the defender in place."""
    sim = sim_for({(0, 0): (1, 1, 5), (0, 1): (1, 2, 4)}, seat=0)
    sim.my_gen, sim.op_gen = 24, 20
    sim.step((0, 0, 3, 0), None)          # move right, all but one -> 4 v 4
    assert sim.owner[1] == 2, "attacker with equal army must not take the cell"
    assert sim.army[1] == 0


def test_sim_chase_outranks_reinforce_and_cancels_the_second_move():
    """§02 priority, and the §07 chase defence that falls out of it."""
    # We chase their source; they move onto a cell of their own (reinforce).
    sim = sim_for(
        {(0, 0): (1, 1, 9), (0, 1): (1, 2, 3), (0, 2): (1, 2, 2)}, seat=0
    )
    sim.my_gen, sim.op_gen = 24, 20
    sim.step((0, 0, 3, 0), (0, 1, 3, 0))  # us (0,0)->(0,1); them (0,1)->(0,2)
    assert sim.owner[1] == 1, "the chase should resolve first and capture"
    # Their move is re-validated against the updated board and is now illegal,
    # so (0,2) never received anything.
    assert sim.army[2] == 2


def test_sim_full_tie_goes_to_player_zero():
    """`game._determine_move_order`: same chase, same reinforce, equal army."""
    for seat, winner in ((0, 1), (1, 2)):
        sim = sim_for({(0, 0): (1, 1, 5), (2, 0): (1, 2, 5)}, seat=seat)
        sim.my_gen, sim.op_gen = 24, 20
        # Both attack the neutral cell (1,0) with 4; whoever resolves *second*
        # attacks the other's 4 and cannot beat it, so the first mover holds.
        sim.step((0, 0, 1, 0), (0, 10, 0, 0))
        assert sim.owner[5] == winner


def test_sim_touch_wins_regardless_of_garrison_from_the_touch_turn():
    for turn, expected in ((799, -1), (800, 1)):
        sim = sim_for({(0, 0): (1, 1, 2), (0, 1): (4, 2, 99)}, seat=0, turn=turn)
        sim.my_gen, sim.op_gen = 24, 1
        sim.step((0, 0, 3, 0), None)
        assert sim.winner == expected


def test_sim_growth_matches_the_engine_cadence():
    """§04: structures on even ticks, every cell every 50 — after the move."""
    sim = sim_for({(0, 0): (4, 1, 5), (4, 4): (1, 1, 3)}, seat=0, turn=47)
    sim.my_gen, sim.op_gen = 0, 24
    sim.step(None, None)                   # -> turn 48, even: general only
    assert (int(sim.army[0]), int(sim.army[24])) == (6, 3)
    sim.step(None, None)                   # -> 49, odd: nothing
    assert (int(sim.army[0]), int(sim.army[24])) == (6, 3)
    sim.step(None, None)                   # -> 50: land bonus *and* even tick
    assert (int(sim.army[0]), int(sim.army[24])) == (8, 4)


# ------------------------------------------------------------------- params
def test_tune_env_overrides_and_bad_json_falls_back(monkeypatch):
    monkeypatch.setenv(TUNE_ENV, '{"mcts_window": 3, "not_a_field": 1}')
    assert load_params().mcts_window == 3

    monkeypatch.setenv(TUNE_ENV, "{not json")
    assert load_params().mcts_window == YankeeParams().mcts_window

    monkeypatch.delenv(TUNE_ENV)
    assert load_params() == YankeeParams()


@pytest.mark.parametrize("field", ["mcts_enabled", "deathtouch_enabled"])
def test_shipped_defaults_are_the_measured_ones(field):
    """Both increments ship on; the control arms are the env override."""
    assert getattr(YankeeParams(), field) is True
