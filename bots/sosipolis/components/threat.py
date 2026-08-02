"""Shared home-threat helpers for phase-adjusted defense scoring."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from components.army import is_wall, neighbors
from params import Params


Cell = tuple[int, int]


@dataclass(frozen=True)
class HomeThreat:
    dist: int | None
    nearest: Cell | None
    threat_army: int


def min_enemy_dist_to(obs, cell: Cell, max_dist: int = 20) -> tuple[int | None, Cell | None]:
    """BFS distance from ``cell`` to the nearest visible enemy cell."""
    H, W = obs.H, obs.W
    dist: dict[Cell, int] = {cell: 0}
    q: deque[Cell] = deque([cell])
    while q:
        r, c = q.popleft()
        if obs.owner_grid[r][c] == 2:
            return dist[(r, c)], (r, c)
        d = dist[(r, c)]
        if d >= max_dist:
            continue
        for nr, nc in neighbors(H, W, r, c):
            if (nr, nc) in dist or is_wall(obs.type_grid, nr, nc):
                continue
            dist[(nr, nc)] = d + 1
            q.append((nr, nc))
    return None, None


def home_threat(obs, state, params: Params) -> HomeThreat:
    home = state.memory.own_general
    if home is None:
        return HomeThreat(None, None, 0)
    dist, nearest = min_enemy_dist_to(obs, home, max_dist=params.DEFENSE_RADIUS + 4)
    army = 0
    if nearest is not None:
        army = obs.army_grid[nearest[0]][nearest[1]]
    return HomeThreat(dist, nearest, army)


def home_bank_target(phase: str, params: Params) -> int:
    if phase == "strike":
        return params.HOME_BANK_STRIKE
    if phase == "contact":
        return params.HOME_BANK_CONTACT
    return params.HOME_BANK_SEARCH


def home_bank_deficit(obs, state, params: Params) -> int:
    home = state.memory.own_general
    if home is None:
        return 0
    target = home_bank_target(state.phase, params)
    have = obs.army_grid[home[0]][home[1]]
    return max(0, target - have)


def defense_weight(phase: str, params: Params) -> float:
    if phase == "strike":
        return params.DEFENSE_WEIGHT_STRIKE
    if phase == "contact":
        return params.DEFENSE_WEIGHT_CONTACT
    return params.DEFENSE_WEIGHT_SEARCH


def defense_score_delta(
    obs,
    state,
    params: Params,
    src: Cell,
    dst: Cell,
) -> float:
    """Score adjustment for a move from src to dst under home threat."""
    home = state.memory.own_general
    if home is None:
        return 0.0
    threat = home_threat(obs, state, params)
    if threat.dist is None or threat.dist > params.DEFENSE_RADIUS + 2:
        return 0.0
    w = defense_weight(state.phase, params)
    delta = 0.0

    if threat.nearest is not None and dst == threat.nearest:
        delta += w * 2.0

    home_d = abs(dst[0] - home[0]) + abs(dst[1] - home[1])
    if home_d <= params.DEFENSE_RADIUS and obs.owner_grid[dst[0]][dst[1]] == 2:
        delta += w * 1.2
    elif home_d <= params.DEFENSE_RADIUS and obs.owner_grid[dst[0]][dst[1]] == 0:
        delta += w * 0.4

    if src == home:
        remaining = 1
        target = home_bank_target(state.phase, params)
        if remaining < target and threat.dist <= params.DEFENSE_RADIUS:
            delta -= w * (target - remaining) / max(target, 1)

    if dst == home:
        deficit = home_bank_deficit(obs, state, params)
        target = home_bank_target(state.phase, params)
        if deficit > 0 and threat.dist <= params.DEFENSE_RADIUS + 1:
            delivered = obs.army_grid[src[0]][src[1]] - 1
            delta += w * 0.8 * min(deficit, delivered) / max(target, 1)

    return delta


def imminent_loss_move(obs, state):
    """Only when an adjacent enemy can capture the general this turn."""
    from components.army import move_action

    home = state.memory.own_general
    if home is None:
        return None
    hr, hc = home
    gen_army = obs.army_grid[hr][hc]
    threat = None
    threat_army = 0
    for r, c in neighbors(obs.H, obs.W, hr, hc):
        if obs.owner_grid[r][c] != 2:
            continue
        a = obs.army_grid[r][c]
        if a - 1 >= gen_army and a > threat_army:
            threat_army = a
            threat = (r, c)
    if threat is None:
        return None
    tr, tc = threat
    for r, c in neighbors(obs.H, obs.W, tr, tc):
        if obs.owner_grid[r][c] != 1:
            continue
        if obs.army_grid[r][c] - 1 > obs.army_grid[tr][tc]:
            return move_action(r, c, tr, tc, 0)
    for r, c in neighbors(obs.H, obs.W, hr, hc):
        if obs.owner_grid[r][c] == 1 and obs.army_grid[r][c] > 1:
            return move_action(r, c, hr, hc, 0)
    return None
