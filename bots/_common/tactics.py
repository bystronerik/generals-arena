"""Grid-native map tactics shared by the migrated strategy bots.

Ports of the generals-bot ``bots/common/{mapping,expand,gather}.py`` helpers,
rewritten for :class:`UnifiedObservation` grids (see arena/bot_api.py). All
coordinates are ``(row, col)`` tuples; moves are five-int unified actions.

Passability for planning: fog (type 0) is passable land, mountains (2) and
structures-in-fog (5) are walls — the same rule the engine applies.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

from _common.strategy_common import DIRECTIONS, direction_from_to, is_passable

Cell = tuple[int, int]

UNREACHABLE = 1 << 30

# RULES.md §03: base castle cost, plus proximity surcharge per own structure.
BUILD_BASE_COST = 35
BUILD_SURCHARGE_CAP = 14
BUILD_SURCHARGE_PER_STEP = 2


def _default_passable(obs) -> Callable[[int, int], bool]:
    grid = obs.type_grid
    return lambda r, c: is_passable(grid[r][c])


def multi_bfs(
    obs,
    sources: Iterable[Cell],
    passable: Callable[[int, int], bool] | None = None,
) -> list[list[int]]:
    """Multi-source BFS distance map over passable cells.

    Returns an ``H x W`` grid of distances; ``UNREACHABLE`` where not
    reachable. Source cells get distance 0 even if not passable themselves.
    """
    H, W = obs.H, obs.W
    if passable is None:
        passable = _default_passable(obs)
    dist = [[UNREACHABLE] * W for _ in range(H)]
    queue: deque[Cell] = deque()
    for r, c in sources:
        if 0 <= r < H and 0 <= c < W and dist[r][c] != 0:
            dist[r][c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        d = dist[r][c] + 1
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and dist[nr][nc] > d and passable(nr, nc):
                dist[nr][nc] = d
                queue.append((nr, nc))
    return dist


def next_step_toward(
    obs,
    start: Cell,
    targets: Iterable[Cell],
    passable: Callable[[int, int], bool] | None = None,
) -> Cell | None:
    """First step of a shortest path from ``start`` to the nearest target.

    Targets are enterable even if the passability predicate rejects them.
    Ties break toward the lowest ``(row, col)``.
    """
    H, W = obs.H, obs.W
    target_set = {(r, c) for r, c in targets if 0 <= r < H and 0 <= c < W}
    if not target_set or start in target_set:
        return None
    if passable is None:
        passable = _default_passable(obs)
    sr, sc = start
    dist = multi_bfs(
        obs,
        sorted(target_set),
        passable=lambda r, c: passable(r, c) or (r, c) == start,
    )
    if dist[sr][sc] >= UNREACHABLE:
        return None
    best: Cell | None = None
    for dr, dc in DIRECTIONS:
        nr, nc = sr + dr, sc + dc
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if dist[nr][nc] >= dist[sr][sc]:
            continue
        if not (passable(nr, nc) or (nr, nc) in target_set):
            continue
        if best is None or dist[nr][nc] < dist[best[0]][best[1]] or (
            dist[nr][nc] == dist[best[0]][best[1]] and (nr, nc) < best
        ):
            best = (nr, nc)
    return best


def gather_step(
    obs,
    target: Cell,
    min_army: int = 2,
    max_dist: int | None = None,
    exclude: set[Cell] | None = None,
    passable: Callable[[int, int], bool] | None = None,
):
    """One move funneling army toward ``target``; a unified action or None.

    Picks the owned cell with army >= ``min_army`` (excluding ``exclude``)
    that maximises ``(army - 1) / (distance + 1)`` and steps it down the
    distance gradient toward ``target``, preferring merges through other
    owned stacks. The target itself is always enterable.
    """
    H, W = obs.H, obs.W
    tr, tc = target
    if not (0 <= tr < H and 0 <= tc < W):
        return None
    exclude = exclude or set()
    if passable is None:
        passable = _default_passable(obs)
    dist = multi_bfs(
        obs, [target], passable=lambda r, c: passable(r, c) or (r, c) == target
    )
    best: Cell | None = None
    best_key = None
    for r in range(H):
        for c in range(W):
            if obs.owner_grid[r][c] != 1 or (r, c) == target or (r, c) in exclude:
                continue
            a = obs.army_grid[r][c]
            if a < min_army or dist[r][c] >= UNREACHABLE:
                continue
            if max_dist is not None and dist[r][c] > max_dist:
                continue
            key = (a - 1) / (dist[r][c] + 1.0)
            if best_key is None or key > best_key:
                best_key = key
                best = (r, c)
    if best is None:
        return None
    br, bc = best
    step: Cell | None = None
    for dr, dc in DIRECTIONS:
        nr, nc = br + dr, bc + dc
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if dist[nr][nc] < dist[br][bc]:
            if step is None or _step_pref(obs, nr, nc) > _step_pref(obs, *step):
                step = (nr, nc)
    if step is None:
        return None
    d = direction_from_to(br, bc, step[0], step[1])
    if d is None:
        return None
    return (0, br, bc, d, 0)


def _step_pref(obs, r: int, c: int) -> tuple[int, int]:
    """Prefer own cells with large armies as intermediate steps."""
    own = 1 if obs.owner_grid[r][c] == 1 else 0
    return (own, obs.army_grid[r][c])


def expansion_step(
    obs,
    reserve: set[Cell] | None = None,
    bias_dist: list[list[int]] | None = None,
):
    """One move that grows our territory; a unified action or None.

    Priority:
      1. Capture an adjacent visible neutral plain we can take this turn,
         preferring captures that reveal fog and (optionally) lower
         ``bias_dist`` (a distance grid — smaller = more desirable region),
         without wasting big stacks.
      2. Otherwise walk the biggest movable stack toward the nearest
         neutral plain or fog cell.
    """
    H, W = obs.H, obs.W
    reserve = reserve or set()
    best = None
    best_key = None
    for r in range(H):
        for c in range(W):
            if obs.owner_grid[r][c] != 1 or (r, c) in reserve:
                continue
            a = obs.army_grid[r][c] - 1
            if a < 1:
                continue
            for d, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if obs.owner_grid[nr][nc] != 0 or obs.type_grid[nr][nc] != 1:
                    continue
                if obs.army_grid[nr][nc] >= a:
                    continue
                key = (
                    fog_reveal(obs, nr, nc),
                    -(bias_dist[nr][nc]) if bias_dist else 0,
                    -obs.army_grid[r][c],
                    (nr, nc),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = (0, r, c, d, 0)
    if best is not None:
        return best

    # Nothing adjacent: walk the biggest stack toward the nearest open land.
    open_land = [
        (r, c)
        for r in range(H)
        for c in range(W)
        if (obs.owner_grid[r][c] == 0 and obs.type_grid[r][c] == 1)
        or obs.type_grid[r][c] == 0
    ]
    if not open_land:
        return None
    dist = multi_bfs(obs, open_land)
    src: Cell | None = None
    src_key = None
    for r in range(H):
        for c in range(W):
            if obs.owner_grid[r][c] != 1 or (r, c) in reserve:
                continue
            if obs.army_grid[r][c] <= 1 or dist[r][c] >= UNREACHABLE:
                continue
            key = (obs.army_grid[r][c], -dist[r][c], (-r, -c))
            if src_key is None or key > src_key:
                src_key = key
                src = (r, c)
    if src is None:
        return None
    sr, sc = src
    for d, (dr, dc) in enumerate(DIRECTIONS):
        nr, nc = sr + dr, sc + dc
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        if dist[nr][nc] < dist[sr][sc] and is_passable(obs.type_grid[nr][nc]):
            return (0, sr, sc, d, 0)
    return None


def fog_reveal(obs, r: int, c: int) -> int:
    """Fog cells in the 3x3 box around ``(r, c)`` — the vision gained there."""
    H, W = obs.H, obs.W
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and obs.type_grid[nr][nc] in (0, 5):
                count += 1
    return count


def frontier_tiles(obs) -> list[Cell]:
    """Owned cells adjacent to at least one passable non-owned cell."""
    H, W = obs.H, obs.W
    out: list[Cell] = []
    for r in range(H):
        for c in range(W):
            if obs.owner_grid[r][c] != 1:
                continue
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if obs.owner_grid[nr][nc] != 1 and is_passable(obs.type_grid[nr][nc]):
                    out.append((r, c))
                    break
    return out


def visible_enemy_tiles(obs) -> list[Cell]:
    """Currently visible cells owned by the opponent."""
    return [
        (r, c)
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.owner_grid[r][c] == 2
    ]


def mirror_tile(obs, r: int, c: int) -> Cell | None:
    """Point reflection of ``(r, c)`` through the map centre, snapped to a
    passable cell.

    Generals spawn far apart; before contact the mirror of our general is a
    reasonable anchor for where the enemy lives.
    """
    H, W = obs.H, obs.W
    mr, mc = H - 1 - r, W - 1 - c
    if is_passable(obs.type_grid[mr][mc]):
        return (mr, mc)
    for radius in range(1, H + W):
        best: Cell | None = None
        for dr in range(-radius, radius + 1):
            dc_abs = radius - abs(dr)
            for dc in {dc_abs, -dc_abs}:
                nr, nc = mr + dr, mc + dc
                if 0 <= nr < H and 0 <= nc < W and is_passable(obs.type_grid[nr][nc]):
                    if best is None or (nr, nc) < best:
                        best = (nr, nc)
        if best is not None:
            return best
    return None


def own_structures(obs) -> list[Cell]:
    """Own general and castles — the cells that raise nearby build costs."""
    return [
        (r, c)
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.owner_grid[r][c] == 1 and obs.type_grid[r][c] in (3, 4)
    ]


def build_cost(obs, r: int, c: int, structures: list[Cell] | None = None) -> int:
    """Army cost of building a castle at ``(r, c)`` (RULES.md §03).

    ``35 + sum(max(0, 14 - 2 * manhattan))`` over own structures; surcharges
    stack. Pass ``structures`` to reuse a precomputed :func:`own_structures`.
    """
    if structures is None:
        structures = own_structures(obs)
    cost = BUILD_BASE_COST
    for sr, sc in structures:
        dist = abs(sr - r) + abs(sc - c)
        cost += max(0, BUILD_SURCHARGE_CAP - BUILD_SURCHARGE_PER_STEP * dist)
    return cost


def army_on(obs, cells: Iterable[Cell]) -> int:
    """Total army on the given cells."""
    H, W = obs.H, obs.W
    return sum(
        obs.army_grid[r][c] for r, c in cells if 0 <= r < H and 0 <= c < W
    )
