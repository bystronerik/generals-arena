"""Rare tip recall and soft defense scoring (Kubic present-but-rare)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from components.army import is_wall, largest_owned_stack, move_action, neighbors, step_toward
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


def manhattan_enemy_to_home(obs, home: Cell) -> tuple[int | None, Cell | None]:
    """Manhattan distance from nearest visible enemy tile to own general."""
    best_d = None
    best_cell = None
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 2:
                continue
            d = abs(r - home[0]) + abs(c - home[1])
            if best_d is None or d < best_d:
                best_d = d
                best_cell = (r, c)
    return best_d, best_cell


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


def strongest_near_home(obs, home: Cell, radius: int) -> tuple[int, Cell | None]:
    """Biggest enemy stack within `radius` (manhattan) of home, and where it is.

    The *nearest* enemy tile is the wrong thing to measure. Once they take a
    cell near our general it stays taken, and a captured cell holding one army
    is not a threat — it is furniture.
    """
    best_army = 0
    best_cell: Cell | None = None
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 2:
                continue
            if abs(r - home[0]) + abs(c - home[1]) > radius:
                continue
            army = obs.army_grid[r][c]
            if army > best_army:
                best_army = army
                best_cell = (r, c)
    return best_army, best_cell


def recall_armed(obs, state, params: Params) -> bool:
    """True when a stack near home could actually take the general.

    This used to arm on *presence*: any visible enemy tile within
    RECALL_PROX_D. Enemy land near our general is permanent, so the gate
    latched the first time macaria captured a cell there and never released —
    measured at 148 consecutive recall turns on seed 7 and 86 on seed 5, the
    bot walking its assault home and shuffling there until it lost. Arm on
    what the stack can do instead: attacking spends one unit to leave, so it
    needs `army - 1` to beat what the general is holding.
    """
    home = state.memory.own_general
    if home is None:
        return False
    d, _ = manhattan_enemy_to_home(obs, home)
    state.home_threat_dist = d
    if d is None or d > params.RECALL_PROX_D:
        return False
    army, _cell = strongest_near_home(obs, home, params.RECALL_PROX_D)
    return army - 1 >= obs.army_grid[home[0]][home[1]]


def recall_move(obs, state, params: Params):
    """Redirect tip home when enemy is near; else reinforce threatened cells."""
    home = state.memory.own_general
    if home is None or not recall_armed(obs, state, params):
        return None
    tip = state.strike_tip or state.chain_head or largest_owned_stack(obs)
    if tip is None:
        return None
    tip_d = abs(tip[0] - home[0]) + abs(tip[1] - home[1])
    blocked = lambda r, c: is_wall(obs.type_grid, r, c)
    if tip_d > 2:
        act = step_toward(obs, tip, home, blocked)
        if act is not None:
            return act
    # Reinforce: capture or step onto threatened cells near home.
    _, nearest = manhattan_enemy_to_home(obs, home)
    if nearest is not None:
        # Prefer capturing the nearby enemy if a win-margin stack is adjacent.
        for r, c in neighbors(obs.H, obs.W, *nearest):
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                continue
            if obs.army_grid[r][c] - 1 > obs.army_grid[nearest[0]][nearest[1]]:
                return move_action(r, c, nearest[0], nearest[1], 0)
        act = step_toward(obs, tip, nearest, blocked)
        if act is not None:
            return act
    # Last: reinforce home.
    for r, c in neighbors(obs.H, obs.W, *home):
        if obs.owner_grid[r][c] == 1 and obs.army_grid[r][c] > 1:
            return move_action(r, c, home[0], home[1], 0)
    return None


def defense_score_delta(
    obs,
    state,
    params: Params,
    src: Cell,
    dst: Cell,
) -> float:
    """Score adjustment only when recall gate is armed (do not fight drain)."""
    if not recall_armed(obs, state, params):
        return 0.0
    home = state.memory.own_general
    if home is None:
        return 0.0
    threat = home_threat(obs, state, params)
    if threat.dist is None:
        return 0.0
    w = defense_weight(state.phase, params)
    delta = 0.0

    if threat.nearest is not None and dst == threat.nearest:
        delta += w * 2.0

    home_d = abs(dst[0] - home[0]) + abs(dst[1] - home[1])
    if home_d <= params.DEFENSE_RADIUS and obs.owner_grid[dst[0]][dst[1]] == 2:
        delta += w * 1.2

    if dst == home:
        delivered = obs.army_grid[src[0]][src[1]] - 1
        delta += w * 0.5 * min(delivered, 8)

    return delta


def imminent_loss_move(obs, state):
    """Only when an adjacent enemy can capture the general this turn."""
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
