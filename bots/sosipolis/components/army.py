"""Gather / leave-1 / legal move helpers."""
from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import Iterable

from params import DIRECTIONS, PASS, T_MOUNTAIN, T_STRUCT_FOG


Cell = tuple[int, int]
Action = tuple[int, int, int, int, int]


def in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= r < H and 0 <= c < W


def is_wall(type_grid, r: int, c: int) -> bool:
    t = type_grid[r][c]
    return t == T_MOUNTAIN or t == T_STRUCT_FOG


def dir_index(dr: int, dc: int) -> int:
    for i, (a, b) in enumerate(DIRECTIONS):
        if a == dr and b == dc:
            return i
    return 0


def neighbors(H: int, W: int, r: int, c: int) -> Iterable[Cell]:
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if in_bounds(H, W, nr, nc):
            yield nr, nc


def move_action(r: int, c: int, nr: int, nc: int, split: int = 0) -> Action:
    return (0, r, c, dir_index(nr - r, nc - c), split)


def bfs_dist(
    H: int,
    W: int,
    starts: list[Cell],
    blocked,
) -> dict[Cell, int]:
    dist: dict[Cell, int] = {}
    q: deque[Cell] = deque()
    for s in starts:
        if blocked(s[0], s[1]):
            continue
        dist[s] = 0
        q.append(s)
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for nr, nc in neighbors(H, W, r, c):
            if (nr, nc) in dist or blocked(nr, nc):
                continue
            dist[(nr, nc)] = d + 1
            q.append((nr, nc))
    return dist


def march_cost(obs, r: int, c: int, cap: int) -> int:
    """Army the marching stack gives up to step onto (r, c).

    Own land is free — the stack absorbs whatever sits there. Neutral costs
    the one unit left behind. An enemy tile costs that plus its garrison, so a
    snake lying across the route is expensive to chew and cheap to walk round.
    """
    owner = obs.owner_grid[r][c]
    if owner == 1:
        return 0
    if owner == 0:
        return 1
    return 1 + min(obs.army_grid[r][c], cap)


def march_dist(
    obs,
    starts: list[Cell],
    blocked,
    cap: int,
) -> dict[Cell, int]:
    """Cheapest marching cost from every cell to the nearest start.

    Hop count says a route through eleven enemy tiles is as good as walking
    round them; on seed 2 that route cost the assault tip 22 of its 28 army.
    """
    H, W = obs.H, obs.W
    best: dict[Cell, int] = {}
    heap: list[tuple[int, Cell]] = []
    for s in starts:
        if not in_bounds(H, W, s[0], s[1]) or blocked(s[0], s[1]):
            continue
        if s not in best:
            best[s] = 0
            heappush(heap, (0, s))
    while heap:
        d, cell = heappop(heap)
        if d > best.get(cell, 1 << 30):
            continue
        r, c = cell
        for nr, nc in neighbors(H, W, r, c):
            if blocked(nr, nc):
                continue
            nd = d + 1 + march_cost(obs, nr, nc, cap)
            if nd < best.get((nr, nc), 1 << 30):
                best[(nr, nc)] = nd
                heappush(heap, (nd, (nr, nc)))
    return best


def legal_moves(obs, leave: int = 1) -> list[Action]:
    """All-but-one moves from owned stacks with army > leave."""
    H, W = obs.H, obs.W
    out: list[Action] = []
    for r in range(H):
        for c in range(W):
            if obs.owner_grid[r][c] != 1:
                continue
            army = obs.army_grid[r][c]
            if army <= leave:
                continue
            for nr, nc in neighbors(H, W, r, c):
                if is_wall(obs.type_grid, nr, nc):
                    continue
                out.append(move_action(r, c, nr, nc, 0))
    return out


def step_toward(
    obs,
    src: Cell,
    goal: Cell,
    blocked,
) -> Action | None:
    """One step from src along a shortest path to goal, if movable."""
    H, W = obs.H, obs.W
    if obs.owner_grid[src[0]][src[1]] != 1 or obs.army_grid[src[0]][src[1]] <= 1:
        return None
    dist = bfs_dist(H, W, [goal], blocked)
    if src not in dist:
        return None
    best = None
    best_d = dist[src]
    for nr, nc in neighbors(H, W, *src):
        if (nr, nc) not in dist:
            continue
        if dist[(nr, nc)] < best_d:
            best_d = dist[(nr, nc)]
            best = (nr, nc)
    if best is None:
        return None
    return move_action(src[0], src[1], best[0], best[1], 0)


def largest_owned_stack(obs) -> Cell | None:
    best = None
    best_a = 0
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1:
                continue
            a = obs.army_grid[r][c]
            if a > best_a:
                best_a = a
                best = (r, c)
    return best


def gather_toward(obs, rally: Cell, blocked) -> Action | None:
    """Move the largest off-rally owned stack one step toward rally."""
    H, W = obs.H, obs.W
    dist = bfs_dist(H, W, [rally], blocked)
    best_act = None
    best_score = -1
    for r in range(H):
        for c in range(W):
            if (r, c) == rally:
                continue
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                continue
            if (r, c) not in dist:
                continue
            for nr, nc in neighbors(H, W, r, c):
                if (nr, nc) not in dist:
                    continue
                if dist[(nr, nc)] >= dist[(r, c)]:
                    continue
                if is_wall(obs.type_grid, nr, nc):
                    continue
                score = obs.army_grid[r][c] * 100 - dist[(r, c)]
                if score > best_score:
                    best_score = score
                    best_act = move_action(r, c, nr, nc, 0)
    return best_act


def pass_action() -> Action:
    return PASS
