"""Persistent fog memory: ever-seen terrain, ownership latch, general hunt."""
from __future__ import annotations

from collections import deque

from params import T_FOG, T_GENERAL, T_MOUNTAIN, T_PLAIN, T_STRUCT_FOG, PARAMS, Params


Cell = tuple[int, int]


class MapMemory:
    def __init__(self, H: int, W: int, min_general_distance: int):
        self.H = H
        self.W = W
        self.min_general_distance = min_general_distance
        self.ever_seen = [[False] * W for _ in range(H)]
        self.known_type = [[T_FOG] * W for _ in range(H)]
        self.known_owner = [[0] * W for _ in range(H)]
        self.last_seen_turn = [[-1] * W for _ in range(H)]
        self.last_enemy_army = [[-1] * W for _ in range(H)]
        self.enemy_army_delta = [[0] * W for _ in range(H)]
        self.own_general: Cell | None = None
        self.enemy_general: Cell | None = None
        self.candidates: set[Cell] = set()
        self.first_contact: Cell | None = None
        self.first_contact_turn: int = -1
        # Sticky: every cell ever seen as enemy-owned.
        self.enemy_seen: set[Cell] = set()
        # Original approach path only (first contact + early/on-axis cells).
        # Late off-axis flanks stay in enemy_seen but not here — they must not
        # yank the hunt away from the first signal.
        self.primary_path: set[Cell] = set()
        self.hunt_cell: Cell | None = None
        self.hunt_turn: int = -10_000
        self._seeded = False
        # Epoch counters for ContactMCTS invalidate-on-change caches.
        self.cand_epoch: int = 0
        self.terrain_epoch: int = 0
        self.enemy_obs_epoch: int = 0

    def update(self, obs) -> None:
        H, W = obs.H, obs.W
        prev_cand = len(self.candidates)
        terrain_changed = False
        enemy_obs_changed = False
        for r in range(H):
            for c in range(W):
                t = obs.type_grid[r][c]
                o = obs.owner_grid[r][c]
                # type 0 = fog (unseen). Everything else is currently visible.
                if t == T_FOG:
                    continue
                self.ever_seen[r][c] = True
                self.last_seen_turn[r][c] = obs.turn
                if self.known_type[r][c] != t:
                    if t in (T_MOUNTAIN, T_STRUCT_FOG) or self.known_type[r][c] in (
                        T_MOUNTAIN,
                        T_STRUCT_FOG,
                    ):
                        terrain_changed = True
                    self.known_type[r][c] = t
                if o != 0:
                    self.known_owner[r][c] = o
                elif t in (T_PLAIN, T_GENERAL, 3):
                    self.known_owner[r][c] = 0

                if o == 2:
                    cell = (r, c)
                    if cell not in self.enemy_seen:
                        self.enemy_seen.add(cell)
                        enemy_obs_changed = True
                    if self.first_contact is None:
                        self.first_contact = cell
                        self.first_contact_turn = obs.turn
                        self.primary_path.add(cell)
                        enemy_obs_changed = True
                    elif cell not in self.primary_path:
                        # Grow primary path only early, or along the first-contact axis.
                        window = PARAMS.CONTACT_PRIMARY_PATH_WINDOW
                        early = (
                            self.first_contact_turn >= 0
                            and obs.turn - self.first_contact_turn <= window
                        )
                        on_axis = False
                        if self.own_general is not None and self.first_contact is not None:
                            hx, hy = self.own_general
                            ax, ay = self.first_contact
                            vx, vy = ax - hx, ay - hy
                            denom = vx * vx + vy * vy
                            if denom > 0:
                                # Do not reuse name `t` — that shadows type_grid
                                # and blocks the enemy_general latch below.
                                axis_t = ((r - hx) * vx + (c - hy) * vy) / denom
                                cross = abs((r - hx) * vy - (c - hy) * vx) / (
                                    denom ** 0.5
                                )
                                on_axis = (
                                    axis_t >= PARAMS.CONTACT_AXIS_MIN_T * 0.8
                                    and cross <= 4.0
                                )
                        if early or on_axis:
                            self.primary_path.add(cell)
                            enemy_obs_changed = True
                    prior = self.last_enemy_army[r][c]
                    current = int(obs.army_grid[r][c])
                    delta = 0 if prior < 0 else current - prior
                    if prior != current:
                        enemy_obs_changed = True
                    self.enemy_army_delta[r][c] = delta
                    self.last_enemy_army[r][c] = current
                if o == 1 and t == T_GENERAL:
                    self.own_general = (r, c)
                if o == 2 and t == T_GENERAL:
                    self.enemy_general = (r, c)
                # first_contact set above when o==2

        if self.own_general is not None and not self._seeded:
            self._seed_candidates()
            self._seeded = True

        self._prune_candidates()
        if len(self.candidates) != prev_cand:
            self.cand_epoch += 1
        if terrain_changed:
            self.terrain_epoch += 1
        if enemy_obs_changed:
            self.enemy_obs_epoch += 1
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

    def enemy_seen_cells(self) -> list[Cell]:
        """Every cell ever observed as enemy-owned (sticky army path)."""
        return list(self.enemy_seen)

    def primary_path_cells(self) -> list[Cell]:
        """Original approach path (first contact + early/on-axis only)."""
        if self.primary_path:
            return list(self.primary_path)
        if self.first_contact is not None:
            return [self.first_contact]
        return self.enemy_seen_cells()

    def recent_enemy_cells(self, turn: int, max_age: int) -> list[Cell]:
        """Enemy-owned cells last seen within max_age turns."""
        cells: list[Cell] = []
        for r in range(self.H):
            for c in range(self.W):
                if self.known_owner[r][c] != 2:
                    continue
                seen = self.last_seen_turn[r][c]
                if seen < 0:
                    continue
                if turn - seen <= max_age:
                    cells.append((r, c))
        return cells

    def hunt_target(self, obs, params: Params) -> Cell | None:
        """Best remaining general candidate to walk vision onto.

        Pre-contact SearchMCTS target only. ContactMCTS owns post-contact
        probe waypoints and must not call this after first contact.
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
        self.cand_epoch += 1

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
        if drop:
            self.cand_epoch += 1
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
