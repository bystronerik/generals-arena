"""Mountain-enclosed small regions → skip expand."""
from __future__ import annotations

from collections import deque

from params import T_MOUNTAIN, T_STRUCT_FOG


Cell = tuple[int, int]


def _neighbors(H: int, W: int, r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            yield nr, nc


def compute_dead_pockets(
    H: int,
    W: int,
    is_passable,
    max_cells: int,
) -> set[Cell]:
    """
    Mark connected passable components of size <= max_cells that have at most
    one mouth (passable edge to a larger exterior component) as dead pockets.
    """
    seen: set[Cell] = set()
    dead: set[Cell] = set()

    for r in range(H):
        for c in range(W):
            if (r, c) in seen or not is_passable(r, c):
                continue
            # Flood the component.
            comp: list[Cell] = []
            q: deque[Cell] = deque([(r, c)])
            seen.add((r, c))
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(H, W, cr, cc):
                    if (nr, nc) in seen:
                        continue
                    if not is_passable(nr, nc):
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc))

            if len(comp) > max_cells:
                continue

            # Mouths: passable neighbours outside the component.
            mouths = 0
            comp_set = set(comp)
            for cr, cc in comp:
                for nr, nc in _neighbors(H, W, cr, cc):
                    if (nr, nc) in comp_set:
                        continue
                    if is_passable(nr, nc):
                        mouths += 1
            # Enclosed bowl or single narrow mouth.
            if mouths <= 1:
                dead.update(comp)
    return dead


def is_known_wall(known_type, r: int, c: int) -> bool:
    return known_type[r][c] in (T_MOUNTAIN, T_STRUCT_FOG)
