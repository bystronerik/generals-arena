"""Castle economy: RULES §03 build cost + Kubic-seeded early programme."""
from __future__ import annotations

from collections import deque

from components.army import Action, gather_toward, is_wall, neighbors
from components.clock import Deadline
from params import Params, T_CASTLE, T_GENERAL, T_PLAIN


Cell = tuple[int, int]


def own_structures(obs) -> list[Cell]:
    out: list[Cell] = []
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1:
                continue
            if obs.type_grid[r][c] in (T_CASTLE, T_GENERAL):
                out.append((r, c))
    return out


def build_cost(obs, r: int, c: int, params: Params, structures: list[Cell] | None = None) -> int:
    """Army cost of building a castle at (r, c) — RULES.md §03."""
    if structures is None:
        structures = own_structures(obs)
    cost = params.BUILD_BASE_COST
    for sr, sc in structures:
        dist = abs(sr - r) + abs(sc - c)
        cost += max(
            0,
            params.BUILD_SURCHARGE_CAP - params.BUILD_SURCHARGE_PER_STEP * dist,
        )
    return cost


def count_owned_castles(obs) -> int:
    n = 0
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] == 1 and obs.type_grid[r][c] == T_CASTLE:
                n += 1
    return n


def build_action(r: int, c: int) -> Action:
    return (2, r, c, 0, 0)


def _enemy_bfs(obs, max_dist: int) -> dict[Cell, int]:
    dist: dict[Cell, int] = {}
    q: deque[Cell] = deque()
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] == 2:
                dist[(r, c)] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        if d >= max_dist:
            continue
        for nr, nc in neighbors(obs.H, obs.W, r, c):
            if (nr, nc) in dist or is_wall(obs.type_grid, nr, nc):
                continue
            dist[(nr, nc)] = d + 1
            q.append((nr, nc))
    return dist


class Economy:
    def __init__(self, params: Params):
        self.params = params

    def decide(self, obs, state, deadline: Deadline) -> Action | None:
        """Return a build or gather-to-site move, or None to leave the turn to MCTS."""
        if deadline.expired():
            return None
        # No new castle projects in strike phase.
        if state.phase == "strike":
            return None
        if obs.turn < self.params.CASTLE_START_TURN:
            return None
        if obs.turn >= self.params.CASTLE_ABORT_TURN:
            return None
        if obs.my_land < self.params.CASTLE_MIN_LAND:
            return None

        castles = count_owned_castles(obs)
        state.castles_owned = castles
        if castles >= self.params.CASTLE_MAX:
            state.build_site = None
            return None

        site = self._pick_site(obs, state)
        if site is None:
            state.build_site = None
            return None
        state.build_site = site

        cost = build_cost(obs, site[0], site[1], self.params)
        need = cost + self.params.CASTLE_KEEP
        sr, sc = site
        if obs.owner_grid[sr][sc] == 1 and obs.army_grid[sr][sc] >= need:
            return build_action(sr, sc)

        blocked = lambda r, c: is_wall(obs.type_grid, r, c)
        return gather_toward(obs, site, blocked)

    def _pick_site(self, obs, state) -> Cell | None:
        home = state.memory.own_general
        if home is None:
            return None
        structures = own_structures(obs)
        enemy_dist = _enemy_bfs(obs, self.params.CASTLE_ENEMY_CLEAR)
        clear = self.params.CASTLE_ENEMY_CLEAR

        # Prefer an existing committed site if still legal.
        if state.build_site is not None:
            r, c = state.build_site
            if (
                obs.owner_grid[r][c] == 1
                and obs.type_grid[r][c] == T_PLAIN
                and enemy_dist.get((r, c), 999) > clear
            ):
                return state.build_site

        best: Cell | None = None
        best_key = None
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.type_grid[r][c] != T_PLAIN:
                    continue
                if enemy_dist.get((r, c), 999) <= clear:
                    continue
                cost = build_cost(obs, r, c, self.params, structures)
                need = cost + self.params.CASTLE_KEEP
                funded = 0 if obs.army_grid[r][c] >= need else 1
                home_d = abs(r - home[0]) + abs(c - home[1])
                # Prefer funded sites, then cheap, then near general.
                key = (funded, cost, home_d, -obs.army_grid[r][c])
                if best_key is None or key < best_key:
                    best_key = key
                    best = (r, c)
        return best
