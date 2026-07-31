"""Tests for the shared grid tactics and opponent model (bots/_common)."""
from __future__ import annotations

from _common.oppmodel import OpponentModel
from _common.tactics import (
    UNREACHABLE,
    build_cost,
    expansion_step,
    fog_reveal,
    gather_step,
    mirror_tile,
    multi_bfs,
    next_step_toward,
    own_structures,
)
from arena.bot_api import UnifiedObservation


def make_obs(type_grid, owner_grid, army_grid, turn=10, **overrides):
    H, W = len(type_grid), len(type_grid[0])
    stats = {"my_land": 0, "my_army": 0, "opp_land": 0, "opp_army": 0}
    for r in range(H):
        for c in range(W):
            if owner_grid[r][c] == 1:
                stats["my_land"] += 1
                stats["my_army"] += army_grid[r][c]
            elif owner_grid[r][c] == 2:
                stats["opp_land"] += 1
                stats["opp_army"] += army_grid[r][c]
    stats.update(overrides)
    return UnifiedObservation(
        H=H, W=W, turn=turn,
        type_grid=type_grid, owner_grid=owner_grid, army_grid=army_grid,
        **stats,
    )


def _grid(H, W, fill=0):
    return [[fill] * W for _ in range(H)]


def test_multi_bfs_walls_and_multi_source():
    # Mountain wall splits row 1; fog (type 0) is passable.
    types = [
        [1, 1, 1],
        [2, 2, 1],
        [1, 0, 1],
    ]
    obs = make_obs(types, _grid(3, 3), _grid(3, 3))
    dist = multi_bfs(obs, [(0, 0)])
    assert dist[0][2] == 2
    assert dist[1][0] == UNREACHABLE  # mountain
    assert dist[2][1] == 5  # around the right side, through fog
    # Multi-source takes the minimum.
    dist2 = multi_bfs(obs, [(0, 0), (2, 2)])
    assert dist2[2][1] == 1


def test_multi_bfs_source_on_wall_gets_zero():
    types = [[1, 2], [1, 1]]
    obs = make_obs(types, _grid(2, 2), _grid(2, 2))
    dist = multi_bfs(obs, [(0, 1)])
    assert dist[0][1] == 0
    # But nothing propagates through it... except neighbors reached via
    # passable cells only; (0,0) is adjacent to the source itself.
    assert dist[0][0] == 1


def test_next_step_toward_enters_impassable_target():
    types = [
        [1, 1, 5],
        [1, 2, 1],
    ]
    obs = make_obs(types, _grid(2, 3), _grid(2, 3))
    # Target (0,2) is a structure-in-fog (impassable) but still enterable.
    step = next_step_toward(obs, (0, 0), [(0, 2)])
    assert step == (0, 1)


def test_gather_step_moves_best_stack_down_gradient():
    types = _grid(3, 3, fill=1)
    owner = [
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
    army = [
        [10, 1, 1],
        [0, 0, 1],
        [0, 0, 2],
    ]
    obs = make_obs(types, owner, army)
    # Rally at (2,2): the 10-stack at (0,0) dominates (army-1)/(dist+1).
    action = gather_step(obs, (2, 2))
    assert action is not None
    p, r, c, d, s = action
    assert (p, r, c, s) == (0, 0, 0, 0)
    assert d in (1, 3)  # down or right, both descend the gradient


def test_gather_step_respects_exclude():
    types = _grid(2, 2, fill=1)
    owner = [[1, 1], [0, 1]]
    army = [[9, 1], [0, 1]]
    obs = make_obs(types, owner, army)
    assert gather_step(obs, (1, 1), exclude={(0, 0)}) is None


def test_expansion_step_prefers_fog_reveal():
    # Two capturable neutrals; (2,1) borders fog below-left, (0,1) does not.
    types = [
        [1, 1, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
    owner = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    army = _grid(3, 3)
    army[1][1] = 5
    obs = make_obs(types, owner, army)
    action = expansion_step(obs)
    assert action is not None
    p, r, c, d, s = action
    assert (r, c) == (1, 1)
    # Direction 1 (down) targets (2,1), which reveals the fog cell at (2,0).
    assert d == 1


def test_expansion_step_respects_reserve_and_falls_back_to_walk():
    # Only owned stack is reserved -> no move at all.
    types = _grid(2, 2, fill=1)
    owner = [[1, 0], [0, 0]]
    army = [[5, 0], [0, 0]]
    obs = make_obs(types, owner, army)
    assert expansion_step(obs, reserve={(0, 0)}) is None

    # Fallback: no adjacent capturable neutral (all owned around), walk toward
    # the far neutral.
    types2 = _grid(1, 4, fill=1)
    owner2 = [[1, 1, 1, 0]]
    army2 = [[1, 8, 1, 0]]
    obs2 = make_obs(types2, owner2, army2)
    action = expansion_step(obs2, reserve=set())
    assert action is not None
    p, r, c, d, s = action
    # Direct adjacent capture from (0,2) with 1 army is impossible (needs >1),
    # so the 8-stack at (0,1) walks right.
    assert (r, c, d) == (0, 1, 3)


def test_fog_reveal_counts_box():
    types = [
        [0, 0, 1],
        [5, 1, 1],
        [1, 1, 1],
    ]
    obs = make_obs(types, _grid(3, 3), _grid(3, 3))
    assert fog_reveal(obs, 1, 1) == 3  # two fog + one structure-in-fog
    assert fog_reveal(obs, 2, 2) == 0


def test_build_cost_matches_rules_table():
    # General at (0,0); candidate cells at varying Manhattan distances.
    types = _grid(1, 9, fill=1)
    types[0][0] = 4
    owner = [[1] * 9]
    obs = make_obs(types, owner, _grid(1, 9))
    assert own_structures(obs) == [(0, 0)]
    assert build_cost(obs, 0, 1) == 47  # d=1: 35 + 12
    assert build_cost(obs, 0, 2) == 45  # d=2: 35 + 10
    assert build_cost(obs, 0, 7) == 35  # d>=7: no surcharge
    # Surcharges stack: add a castle at (0,4); cell (0,2) is d=2 from both.
    types[0][4] = 3
    assert build_cost(obs, 0, 2) == 55  # 35 + 10 + 10


def test_mirror_tile_reflects_and_snaps():
    types = _grid(3, 3, fill=1)
    obs = make_obs(types, _grid(3, 3), _grid(3, 3))
    assert mirror_tile(obs, 0, 0) == (2, 2)
    # Mirror lands on a mountain -> snaps to an adjacent passable cell.
    types[2][2] = 2
    assert mirror_tile(obs, 0, 0) in ((1, 2), (2, 1))


def test_opponent_model_signals():
    model = OpponentModel()
    types = _grid(3, 3, fill=1)
    types[0][0] = 4
    owner = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 2],
    ]
    army = [
        [5, 0, 0],
        [0, 0, 0],
        [0, 0, 7],
    ]
    obs = make_obs(types, owner, army, turn=10)
    model.update(obs)
    model.update(obs)  # idempotent: same turn recorded once
    assert len(model.turns) == 1
    assert model.first_contact_turn == 10
    assert model.biggest_enemy_stack_now == 7
    assert model.closest_enemy_dist_now == 4
    assert model.under_attack()
    assert model.opponent_mobile() == 6  # 7 army - 1 land

    # Enemy castle visibility is sticky.
    types[2][2] = 3
    obs2 = make_obs(types, owner, army, turn=11)
    model.update(obs2)
    assert model.enemy_castles_seen == 1

    # Army drop over a window.
    army[2][2] = 2
    obs3 = make_obs(types, owner, army, turn=12)
    model.update(obs3)
    assert model.opp_total_drop() == 5
    assert model.army_ratio() == 5 / 2
