"""Persistent fog memory: ever-seen terrain, ownership latch, general hunt."""
from __future__ import annotations

from collections import deque

from params import T_FOG, T_GENERAL, T_MOUNTAIN, T_PLAIN, T_STRUCT_FOG, Params


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
        self.hunt_cell: Cell | None = None
        self.hunt_turn: int = -10_000
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
        if self.enemy_general is not None:
            self.hunt_cell = self.enemy_general
            self.hunt_turn = obs.turn

    def enemy_cells(self) -> list[Cell]:
        """Remembered enemy ownership (visible now or latched in fog)."""
        cells: list[Cell] = []
        for r in range(self.H):
            for c in range(self.W):
                if self.known_owner[r][c] == 2:
                    cells.append((r, c))
        return cells

    def hunt_target(self, obs, params: Params) -> Cell | None:
        """Best remaining general candidate to walk vision onto.

        Scores each belief cell by how many candidates a visit would prune
        (``HUNT_REVEAL_RADIUS`` box), discounted by travel from owned land and
        by distance away from the enemy footprint (or the mirror of our
        general before contact). After contact, the pool prefers candidates
        within ``HUNT_CONTACT_RADIUS`` of remembered enemy land — fog behind
        their front, not the near edge of the whole ≥17 ring.
        """
        if self.enemy_general is not None:
            return self.enemy_general
        if not self.candidates:
            return None

        cached = self.hunt_cell
        if (
            cached is not None
            and cached in self.candidates
            and obs.turn - self.hunt_turn < params.HUNT_INTERVAL
        ):
            return cached

        pool = self._hunt_pool(params)
        if not pool:
            return None

        owned = [
            (r, c)
            for r in range(obs.H)
            for c in range(obs.W)
            if obs.owner_grid[r][c] == 1
        ]
        from_home = self._bfs_multi(owned) if owned else {}

        enemies = self.enemy_cells()
        from_prior: dict[Cell, int] | None = None
        if enemies:
            from_prior = self._bfs_multi(enemies)
        elif self.own_general is not None:
            mr, mc = self.own_general
            mirror = (self.H - 1 - mr, self.W - 1 - mc)
            from_prior = self._bfs_multi([mirror])

        far = self.H + self.W
        radius = params.HUNT_REVEAL_RADIUS
        best: Cell | None = None
        best_key = None
        for cell in pool:
            cost = from_home.get(cell, far)
            penalty = 1.0 + params.HUNT_TRAVEL_DECAY * cost
            if from_prior is not None:
                away = from_prior.get(cell, far)
                penalty *= 1.0 + params.HUNT_PRIOR_DECAY * min(away, far)
            prune = 0
            r, c = cell
            for rr in range(r - radius, r + radius + 1):
                for cc in range(c - radius, c + radius + 1):
                    if (rr, cc) in self.candidates:
                        prune += 1
            key = (prune / penalty, -cost, (-r, -c))
            if best_key is None or key > best_key:
                best_key, best = key, cell

        if best is None:
            return None
        self.hunt_cell = best
        self.hunt_turn = obs.turn
        return best

    def _hunt_pool(self, params: Params) -> set[Cell]:
        """Candidates behind the enemy front when a footprint exists."""
        pool = set(self.candidates)
        enemies = self.enemy_cells()
        if not enemies:
            return pool
        from_enemy = self._bfs_multi(enemies)
        near = {
            cell
            for cell in pool
            if from_enemy.get(cell, 10_000) <= params.HUNT_CONTACT_RADIUS
        }
        return near if near else pool

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
        if self.hunt_cell is not None and self.hunt_cell not in self.candidates:
            self.hunt_cell = None
        if not self.candidates and self.own_general is not None:
            # Rebuild from never-seen passable cells at distance ≥ min.
            self._seed_candidates()
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
        return self._bfs_multi([start])

    def _bfs_multi(self, starts: list[Cell]) -> dict[Cell, int]:
        H, W = self.H, self.W
        dist: dict[Cell, int] = {}
        q: deque[Cell] = deque()
        for s in starts:
            if not (0 <= s[0] < H and 0 <= s[1] < W):
                continue
            if self._is_known_mountain(s) and self.known_owner[s[0]][s[1]] == 0:
                continue
            dist[s] = 0
            q.append(s)
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
