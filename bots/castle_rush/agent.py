"""
castle_rush — aggressive early castle building.

Expands like `expand_plus` during setup (turn < 10) and after the rush
cap (4 castles). Between those, rests the general and builds castles with
tighter margins, shorter cooldown, and a higher cap than `castle_builder`.

See docs/bots/castle-rush.md,
docs/research/strategies/castle_rush.md, and
docs/research/experiments/011-castle-rush-aggressive-builds.md.
"""
from collections import deque

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

RUSH_START = 10
RUSH_CAP = 4
RUSH_COOLDOWN = 25
RUSH_MIN_LAND = 6
RUSH_SURPLUS_MARGIN = 8

BASE_BUILD_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2


def _is_passable(t):
    return t != 2 and t != 5


def _is_capturable(owner, cell_type):
    if owner == 2:
        return True
    return owner == 0 and cell_type not in (0, 5)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Agent:
    """expand_plus expansion with an aggressive castle rush window."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.last_build_turn = -RUSH_COOLDOWN
        self.general_pos = None
        self.enemy_general = None

    def act(self, obs):
        self._locate_general(obs)
        self._update_enemy_general_sighting(obs)

        _, own_castle_count = self._own_structures(obs)
        if own_castle_count < RUSH_CAP and obs.turn >= RUSH_START:
            return self._rush_loop(obs, own_castle_count)

        return self._expand(obs)

    def _rush_loop(self, obs, own_castle_count):
        structures, _ = self._own_structures(obs)
        resting = self._should_rest_general(obs, own_castle_count)

        build_move = self._maybe_build(obs, structures, own_castle_count)
        if build_move is not None:
            self.last_build_turn = obs.turn
            return build_move

        if resting:
            relocate_move = self._maybe_relocate_general(obs, structures)
            if relocate_move is not None:
                return relocate_move

        exclude = self.general_pos if resting else None
        return self._expand(obs, exclude_cell=exclude)

    def _locate_general(self, obs):
        if self.general_pos is not None:
            return
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 1 and obs.type_grid[r][c] == 4:
                    self.general_pos = (r, c)
                    return

    def _update_enemy_general_sighting(self, obs):
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 2 and obs.type_grid[r][c] == 4:
                    self.enemy_general = (r, c)
                    return

    def _should_rest_general(self, obs, own_castle_count):
        if self.general_pos is None:
            return False
        if own_castle_count >= RUSH_CAP:
            return False
        return obs.my_land >= RUSH_MIN_LAND

    def _own_structures(self, obs):
        structures = []
        castles = 0
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                t = obs.type_grid[r][c]
                if t == 4:
                    structures.append((r, c))
                elif t == 3:
                    structures.append((r, c))
                    castles += 1
        return structures, castles

    def _build_cost(self, cell, structures):
        cost = BASE_BUILD_COST
        for s in structures:
            d = _manhattan(cell, s)
            cost += max(0, PROXIMITY_PENALTY - PROXIMITY_DECAY * d)
        return cost

    def _maybe_build(self, obs, structures, own_castle_count):
        if own_castle_count >= RUSH_CAP:
            return None
        if obs.turn - self.last_build_turn < RUSH_COOLDOWN:
            return None

        best_cell = None
        best_surplus = float("-inf")
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.type_grid[r][c] != 1:
                    continue
                army = obs.army_grid[r][c]
                cost = self._build_cost((r, c), structures)
                surplus = army - cost
                if surplus > best_surplus:
                    best_surplus = surplus
                    best_cell = (r, c)

        if best_cell is not None and best_surplus >= RUSH_SURPLUS_MARGIN:
            r, c = best_cell
            return (2, r, c, 0, 0)
        return None

    def _maybe_relocate_general(self, obs, structures):
        gr, gc = self.general_pos
        army = obs.army_grid[gr][gc]
        if army <= 1:
            return None

        best_dir = None
        best_cost = None
        for d, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = gr + dr, gc + dc
            if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                continue
            if not _is_passable(obs.type_grid[nr][nc]):
                continue
            if obs.owner_grid[nr][nc] == 2:
                continue
            if obs.type_grid[nr][nc] in (3, 4):
                continue
            cost = self._build_cost((nr, nc), structures)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_dir = d

        if best_dir is None:
            return None

        transferred = army - 1
        if transferred < best_cost + RUSH_SURPLUS_MARGIN:
            return None
        return (0, gr, gc, best_dir, 0)

    def _expand(self, obs, exclude_cell=None):
        H, W = obs.H, obs.W
        best_score = -1.0
        best_move = None

        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if exclude_cell is not None and (r, c) == exclude_cell:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue

                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue

                    dest_owner = obs.owner_grid[nr][nc]
                    dest_army = obs.army_grid[nr][nc]
                    if src_army <= dest_army + 1:
                        continue

                    is_opp = dest_owner == 2
                    dest_type = obs.type_grid[nr][nc]
                    is_visible_neutral = dest_owner == 0 and dest_type not in (0, 5)
                    is_expansion = is_opp or is_visible_neutral
                    if not is_expansion:
                        continue

                    score = float(src_army)
                    score *= 10.0
                    if is_opp:
                        score *= 2.0

                    if score > best_score:
                        best_score = score
                        best_move = (0, r, c, d, 0)

        if best_move is not None:
            return best_move

        march_move = self._march_toward_frontier(obs, exclude_cell)
        if march_move is not None:
            return march_move

        first_valid = self._any_valid_move(obs, exclude_cell)
        if first_valid is not None:
            return first_valid
        return PASS

    def _march_toward_frontier(self, obs, exclude_cell=None):
        H, W = obs.H, obs.W
        dist = [[-1] * W for _ in range(H)]
        q = deque()
        for r in range(H):
            for c in range(W):
                if _is_capturable(obs.owner_grid[r][c], obs.type_grid[r][c]):
                    dist[r][c] = 0
                    q.append((r, c))
        if not q:
            return None

        while q:
            r, c = q.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if dist[nr][nc] != -1:
                    continue
                if not _is_passable(obs.type_grid[nr][nc]):
                    continue
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))

        best_army = -1
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if exclude_cell is not None and (r, c) == exclude_cell:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue
                here = dist[r][c]
                if here <= 0:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue
                    if obs.owner_grid[nr][nc] != 1:
                        continue
                    if dist[nr][nc] < 0 or dist[nr][nc] >= here:
                        continue
                    if src_army > best_army:
                        best_army = src_army
                        best_move = (0, r, c, d, 0)
        return best_move

    def _any_valid_move(self, obs, exclude_cell=None):
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if exclude_cell is not None and (r, c) == exclude_cell:
                    continue
                if obs.army_grid[r][c] <= 1:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and _is_passable(obs.type_grid[nr][nc]):
                        return (0, r, c, d, 0)
        return None
