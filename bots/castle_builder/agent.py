"""
castle_builder — convey-funded castles at minimum spacing.

Expands with smallest-sufficient-source scoring and general reserve.
When idle, conveys land income into a spaced bank cell (price 35) and builds
once the bank holds 35 + garrison army. Win checks run every tick.

See docs/bots/castle-builder.md and docs/research/strategies/optimize-existing.md.
"""
from collections import deque

from strategy_common import (
    PASS,
    DIRECTIONS,
    BeliefState,
    best_capture_move,
    bfs_distances,
    can_use_as_source,
    chase_defence,
    enemy_adjacent_to_general,
    is_passable,
    locate_own_general,
    win_check_w1,
    win_check_w2,
)

BASE_BUILD_COST = 35
BANK_MIN_SPACING = 7
BUILD_GARRISON = 10
MIN_TURN_TO_BUILD = 20
MAX_TURN_TO_BUILD = 600
MIN_LAND_TO_BUILD = 8
MAX_OWN_CASTLES = 3
BUILD_COOLDOWN_TURNS = 40


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Agent:
    """Greedy expansion plus convey-funded castles away from structures."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.general_pos = None
        self.bank_cell = None
        self.last_build_turn = -BUILD_COOLDOWN_TURNS
        self.belief = BeliefState()

    def act(self, obs):
        if self.general_pos is None:
            self.general_pos = locate_own_general(obs)
        self.belief.update(obs, self.general_pos)

        chase = chase_defence(obs, self.general_pos)
        if chase is not None:
            return chase

        w1 = win_check_w1(obs, self.belief.enemy_general, self.general_pos)
        if w1 is not None:
            return w1

        if not enemy_adjacent_to_general(obs, self.general_pos):
            w2 = win_check_w2(obs, self.belief.enemy_general, self.general_pos)
            if w2 is not None:
                return w2

        structures, own_castle_count = self._own_structures(obs)
        if self._can_build(obs, own_castle_count):
            build = self._maybe_build(obs, structures, own_castle_count)
            if build is not None:
                self.last_build_turn = obs.turn
                return build

        capture = best_capture_move(obs, self.general_pos)
        if capture is not None:
            return capture

        convey = self._convey_toward_bank(obs, structures)
        if convey is not None:
            return convey

        return PASS

    def telemetry_extras(self):
        if self.belief.first_sighting_turn is None:
            return {"enemy_general_sighted": 0}
        return {
            "enemy_general_sighted": 1,
            "first_sighting_turn": self.belief.first_sighting_turn,
        }

    def _own_structures(self, obs):
        structures = []
        castles = 0
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                t = obs.type_grid[r][c]
                if t in (3, 4):
                    structures.append((r, c))
                    if t == 3:
                        castles += 1
        return structures, castles

    def _can_build(self, obs, own_castle_count):
        if own_castle_count >= MAX_OWN_CASTLES:
            return False
        if not (MIN_TURN_TO_BUILD <= obs.turn <= MAX_TURN_TO_BUILD):
            return False
        if obs.my_land < MIN_LAND_TO_BUILD:
            return False
        if obs.turn - self.last_build_turn < BUILD_COOLDOWN_TURNS:
            return False
        return True

    def _min_structure_distance(self, cell, structures):
        if not structures:
            return float("inf")
        return min(_manhattan(cell, s) for s in structures)

    def _frontier_distances(self, obs):
        H, W = obs.H, obs.W
        dist = [[-1] * W for _ in range(H)]
        q = deque()
        for r in range(H):
            for c in range(W):
                owner = obs.owner_grid[r][c]
                ctype = obs.type_grid[r][c]
                if owner == 2 or (owner == 0 and ctype not in (0, 5)):
                    dist[r][c] = 0
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if dist[nr][nc] != -1:
                    continue
                if not is_passable(obs.type_grid[nr][nc]):
                    continue
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
        return dist

    def _choose_bank_cell(self, obs, structures):
        frontier = self._frontier_distances(obs)
        best = None
        best_frontier = float("inf")
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.type_grid[r][c] != 1:
                    continue
                if self._min_structure_distance((r, c), structures) < BANK_MIN_SPACING:
                    continue
                fd = frontier[r][c]
                if fd < 0:
                    fd = 9999
                if fd < best_frontier:
                    best_frontier = fd
                    best = (r, c)
        return best

    def _maybe_build(self, obs, structures, own_castle_count):
        if self.bank_cell is None or obs.owner_grid[self.bank_cell[0]][self.bank_cell[1]] != 1:
            self.bank_cell = self._choose_bank_cell(obs, structures)
        if self.bank_cell is None:
            return None
        br, bc = self.bank_cell
        if obs.type_grid[br][bc] != 1:
            self.bank_cell = self._choose_bank_cell(obs, structures)
            if self.bank_cell is None:
                return None
            br, bc = self.bank_cell
        if obs.army_grid[br][bc] >= BASE_BUILD_COST + BUILD_GARRISON:
            return (2, br, bc, 0, 0)
        return None

    def _convey_toward_bank(self, obs, structures):
        if self.bank_cell is None or obs.owner_grid[self.bank_cell[0]][self.bank_cell[1]] != 1:
            self.bank_cell = self._choose_bank_cell(obs, structures)
        if self.bank_cell is None:
            return None
        br, bc = self.bank_cell
        if obs.army_grid[br][bc] >= BASE_BUILD_COST + BUILD_GARRISON:
            return None

        dist = bfs_distances(obs, self.bank_cell, owned_only=True)
        H, W = obs.H, obs.W
        best_army = -1
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue
                if (r, c) == self.general_pos:
                    if not can_use_as_source(obs, r, c, 0, self.general_pos, obs.turn):
                        continue
                here = dist[r][c]
                if here <= 0:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if obs.owner_grid[nr][nc] != 1:
                        continue
                    if dist[nr][nc] < 0 or dist[nr][nc] >= here:
                        continue
                    if src_army > best_army:
                        best_army = src_army
                        best_move = (0, r, c, d, 0)
        return best_move
