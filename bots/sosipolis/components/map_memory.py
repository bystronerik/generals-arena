"""Persistent fog memory: ever-seen terrain and ownership latch."""
from __future__ import annotations

from collections import deque

from params import T_FOG, T_GENERAL, T_MOUNTAIN, T_PLAIN, T_STRUCT_FOG


Cell = tuple[int, int]


class MapMemory:
    def __init__(self, H: int, W: int, min_general_distance: int):
        self.H = H
        self.W = W
        self.min_general_distance = min_general_distance
        self.ever_seen = [[False] * W for _ in range(H)]
        self.known_type = [[T_FOG] * W for _ in range(H)]
        self.known_owner = [[0] * W for _ in range(H)]
        self.own_general: Cell | None = None
        self.enemy_general: Cell | None = None
        self.candidates: set[Cell] = set()
        self.first_contact: Cell | None = None
        self._seeded = False

    def update(self, obs) -> None:
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                t = obs.type_grid[r][c]
                o = obs.owner_grid[r][c]
                # type 0 = fog (unseen). Everything else is currently visible.
                if t == T_FOG:
                    continue
                self.ever_seen[r][c] = True
                self.known_type[r][c] = t
                if o != 0:
                    self.known_owner[r][c] = o
                elif t in (T_PLAIN, T_GENERAL, 3):
                    self.known_owner[r][c] = 0

                if o == 1 and t == T_GENERAL:
                    self.own_general = (r, c)
                if o == 2 and t == T_GENERAL:
                    self.enemy_general = (r, c)
                if o == 2 and self.first_contact is None:
                    self.first_contact = (r, c)

        if self.own_general is not None and not self._seeded:
            self._seed_candidates()
            self._seeded = True

        self._prune_candidates()

    def _seed_candidates(self) -> None:
        assert self.own_general is not None
        dist = self._bfs_passable_from(self.own_general)
        self.candidates = {
            cell
            for cell, d in dist.items()
            if d >= self.min_general_distance and not self._is_known_mountain(cell)
        }

    def _prune_candidates(self) -> None:
        if self.enemy_general is not None:
            self.candidates = {self.enemy_general}
            return
        drop: list[Cell] = []
        for cell in self.candidates:
            r, c = cell
            if not self.ever_seen[r][c]:
                continue
            # Seen and not the enemy general → cannot be the general.
            if self.known_type[r][c] != T_GENERAL or self.known_owner[r][c] != 2:
                drop.append(cell)
        for cell in drop:
            self.candidates.discard(cell)
        if not self.candidates and self.own_general is not None:
            # Rebuild from never-seen passable cells at distance ≥ min.
            self._seed_candidates()
            # Keep only never-seen among rebuild.
            self.candidates = {
                cell for cell in self.candidates if not self.ever_seen[cell[0]][cell[1]]
            }

    def _is_known_mountain(self, cell: Cell) -> bool:
        r, c = cell
        return self.known_type[r][c] in (T_MOUNTAIN, T_STRUCT_FOG)

    def is_passable_belief(self, r: int, c: int) -> bool:
        t = self.known_type[r][c]
        if t in (T_MOUNTAIN, T_STRUCT_FOG):
            return False
        # Unseen fog assumed passable until proven otherwise.
        return True

    def _bfs_passable_from(self, start: Cell) -> dict[Cell, int]:
        H, W = self.H, self.W
        dist: dict[Cell, int] = {start: 0}
        q: deque[Cell] = deque([start])
        while q:
            r, c = q.popleft()
            d = dist[(r, c)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in dist:
                    continue
                if not self.is_passable_belief(nr, nc):
                    continue
                dist[(nr, nc)] = d + 1
                q.append((nr, nc))
        return dist
