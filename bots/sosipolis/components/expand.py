"""Compact neutral expansion — the economy that decides the game.

The general makes one army every *other* turn (RULES §04), so land by t=50 is
capped near 24 no matter how well we play, and Kubic reaches it. Every
produced unit therefore gets about one spare turn of transit before it costs a
capture. Two consequences drive everything here:

1. **The frontier has to stay next to the army source.** A thin tendril
   reaching for distant fog puts the only capturable cells five to seven steps
   from the general, so a unit takes five to seven turns to reach work that
   should take one. Measured on seed 0: 30 of 47 opening turns were spent
   walking army across our own land, 17 captures against Kubic's 24.
2. **A snake is free.** A stack of A army takes A−1 cells on A−1 consecutive
   turns, paying transit once. Breaking the chain to start a new run from the
   general pays it again.

So: continue the chain while it can still take a cell, otherwise take the
capturable cell nearest the general, otherwise walk the biggest stack toward
the nearest neutral. No fog-frontier pull — that is what stretched the tendril.
"""
from __future__ import annotations

from collections import deque

from components.army import (
    Action,
    is_wall,
    move_action,
    neighbors,
)
from params import Params, T_CASTLE


Cell = tuple[int, int]


def _takeable(obs, r: int, c: int) -> bool:
    """A plain neutral cell. Neutral castles cost ~35 and are not economy."""
    return (
        obs.owner_grid[r][c] == 0
        and obs.type_grid[r][c] != T_CASTLE
        and not is_wall(obs.type_grid, r, c)
    )


def _own_region_dist(obs, home: Cell | None) -> dict[Cell, int]:
    """Steps from the general through our own land, then one step beyond.

    Distance over owned cells only, so a cell's value is how far the army that
    takes it has to travel — not how far it is as the crow flies.
    """
    if home is None:
        return {}
    dist: dict[Cell, int] = {home: 0}
    q: deque[Cell] = deque([home])
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for nr, nc in neighbors(obs.H, obs.W, r, c):
            if (nr, nc) in dist or is_wall(obs.type_grid, nr, nc):
                continue
            if obs.owner_grid[nr][nc] != 1:
                # Reachable, but the walk stops here — it is not ours yet.
                dist[(nr, nc)] = d + 1
                continue
            dist[(nr, nc)] = d + 1
            q.append((nr, nc))
    return dist


def _dist_to_neutral(obs) -> dict[Cell, int]:
    """Steps from every cell to the nearest takeable neutral."""
    dist: dict[Cell, int] = {}
    q: deque[Cell] = deque()
    for r in range(obs.H):
        for c in range(obs.W):
            if _takeable(obs, r, c):
                dist[(r, c)] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for nr, nc in neighbors(obs.H, obs.W, r, c):
            if (nr, nc) in dist or is_wall(obs.type_grid, nr, nc):
                continue
            dist[(nr, nc)] = d + 1
            q.append((nr, nc))
    return dist


def capture_move(obs, state, params: Params, exclude: Cell | None = None) -> Action | None:
    """Take a neutral cell this turn, cheapest walk first.

    `exclude` keeps the assault tip out of it: while the tip is saving up for
    a march, spending it one neutral at a time is how it never gets there.
    """
    home = state.memory.own_general
    region = _own_region_dist(obs, home)
    far = obs.H + obs.W
    head = state.chain_head

    best: Action | None = None
    best_key = None
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                continue
            if exclude is not None and (r, c) == exclude:
                continue
            for nr, nc in neighbors(obs.H, obs.W, r, c):
                if not _takeable(obs, nr, nc):
                    continue
                if obs.army_grid[r][c] - 1 <= obs.army_grid[nr][nc]:
                    continue
                # Chain first: continuing a live snake costs no transit at all.
                chain = 0 if (head is not None and (r, c) == head) else 1
                key = (
                    chain,
                    region.get((nr, nc), far),
                    0 if not state.memory.ever_seen[nr][nc] else 1,
                    -_room(obs, nr, nc),
                    nr,
                    nc,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = move_action(r, c, nr, nc, 0)
    return best


def _room(obs, r: int, c: int) -> int:
    """Takeable neighbours — prefer cells the snake can carry on through."""
    return sum(
        1
        for nr, nc in neighbors(obs.H, obs.W, r, c)
        if _takeable(obs, nr, nc)
    )


def approach_move(obs, state, params: Params, exclude: Cell | None = None) -> Action | None:
    """Nothing to take: walk the biggest stack toward the nearest neutral."""
    to_neutral = _dist_to_neutral(obs)
    if not to_neutral:
        return None
    best: Action | None = None
    best_key = None
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                continue
            if exclude is not None and (r, c) == exclude:
                continue
            here = to_neutral.get((r, c))
            if here is None:
                continue
            for nr, nc in neighbors(obs.H, obs.W, r, c):
                if is_wall(obs.type_grid, nr, nc):
                    continue
                if obs.owner_grid[nr][nc] != 1:
                    continue
                there = to_neutral.get((nr, nc))
                if there is None or there >= here:
                    continue
                # Biggest stack, shortest remaining walk.
                key = (-obs.army_grid[r][c], there, nr, nc)
                if best_key is None or key < best_key:
                    best_key = key
                    best = move_action(r, c, nr, nc, 0)
    return best


def expand_move(
    obs, state, params: Params, exclude: Cell | None = None
) -> Action | None:
    """One economy move: take a cell, else carry army toward one."""
    take = capture_move(obs, state, params, exclude)
    if take is not None:
        return take
    return approach_move(obs, state, params, exclude)


def far_haul_capture(obs, state, params: Params, muster: Cell | None) -> Action | None:
    """Turn a long gather haul into a capture the same army can make now.

    Gather walks army to the muster one step a tick. For a stack far from the
    muster that is a dozen ticks of nothing, and it arrives thinned by what it
    left behind. If that stack is sitting next to a neutral it can take, the
    land is worth more than the fraction of it that would ever arrive.
    """
    if muster is None:
        return None
    head = state.chain_head
    best: Action | None = None
    best_key = None
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                continue
            if (r, c) == muster or (head is not None and (r, c) == head):
                continue
            haul = abs(r - muster[0]) + abs(c - muster[1])
            if haul < params.GATHER_LOCAL_HAUL:
                continue
            for nr, nc in neighbors(obs.H, obs.W, r, c):
                if not _takeable(obs, nr, nc):
                    continue
                if obs.army_grid[r][c] - 1 <= obs.army_grid[nr][nc]:
                    continue
                key = (-haul, -obs.army_grid[r][c], nr, nc)
                if best_key is None or key < best_key:
                    best_key = key
                    best = move_action(r, c, nr, nc, 0)
    return best
