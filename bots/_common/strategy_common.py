"""Shared competition strategy helpers (see docs/research/strategies/optimize-existing.md)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PROBE_CANDIDATE_SAMPLE = 25


def is_passable(t: int) -> bool:
    return t != 2 and t != 5


def is_capturable(owner: int, cell_type: int) -> bool:
    if owner == 2:
        return True
    return owner == 0 and cell_type not in (0, 5)


def direction_from_to(sr: int, sc: int, tr: int, tc: int) -> int | None:
    dr, dc = tr - sr, tc - sc
    for d, (ddr, ddc) in enumerate(DIRECTIONS):
        if ddr == dr and ddc == dc:
            return d
    return None


def locate_own_general(obs) -> tuple[int, int] | None:
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] == 1 and obs.type_grid[r][c] == 4:
                return (r, c)
    return None


def enemy_general_army(obs, pos: tuple[int, int]) -> int:
    r, c = pos
    if obs.type_grid[r][c] == 0:
        return 1 + obs.turn // 2
    return obs.army_grid[r][c]


def enemy_adjacent_to_general(obs, general_pos: tuple[int, int] | None) -> bool:
    if general_pos is None:
        return False
    gr, gc = general_pos
    H, W = obs.H, obs.W
    for dr, dc in DIRECTIONS:
        nr, nc = gr + dr, gc + dc
        if 0 <= nr < H and 0 <= nc < W and obs.owner_grid[nr][nc] == 2:
            return True
    return False


def bfs_distances(obs, start: tuple[int, int] | None, owned_only: bool = False):
    H, W = obs.H, obs.W
    dist = [[-1] * W for _ in range(H)]
    if start is None:
        return dist
    sr, sc = start
    dist[sr][sc] = 0
    q = deque([(sr, sc)])
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
            if owned_only and obs.owner_grid[nr][nc] != 1:
                continue
            dist[nr][nc] = dist[r][c] + 1
            q.append((nr, nc))
    return dist


def reveal_gain_at(obs, candidates: set[tuple[int, int]], r: int, c: int) -> int:
    H, W = obs.H, obs.W
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) in candidates:
                count += 1
    return count


@dataclass(frozen=True)
class StrategyConfig:
    reserve_opening_end: int = 60
    reserve_max: int | None = 30
    defend_from: int = 780
    sentry_from: int = 700
    deathtouch_turn: int = 800
    min_general_distance: int = 17
    enemy_dest_value: int = 12
    neutral_dest_value: int = 6
    probe_candidate_sample: int = PROBE_CANDIDATE_SAMPLE


class StrategyContext:
    """Strategy helpers bound to one bot's named constants."""

    def __init__(self, config: StrategyConfig | None = None, **kwargs):
        if config is None:
            config = StrategyConfig(**kwargs)
        self.config = config

    def general_reserve(self, turn: int) -> int:
        cfg = self.config
        if turn < cfg.reserve_opening_end:
            return 0
        reserve = 3 + (turn - cfg.reserve_opening_end) // 25
        if cfg.reserve_max is None:
            return reserve
        return min(cfg.reserve_max, reserve)

    def can_use_general_as_source(self, army: int, turn: int, split: int) -> bool:
        reserve = self.general_reserve(turn)
        if split == 0:
            return reserve <= 1
        left_behind = army - army // 2
        moving = army // 2
        return left_behind >= reserve and moving >= 1

    def can_use_as_source(self, obs, r: int, c: int, split: int, general_pos, turn: int) -> bool:
        if general_pos is None or (r, c) != general_pos:
            return True
        return self.can_use_general_as_source(obs.army_grid[r][c], turn, split)

    def dest_value(self, owner: int, cell_type: int) -> int:
        cfg = self.config
        if owner == 2:
            return cfg.enemy_dest_value
        if owner == 0 and cell_type in (1, 3):
            return cfg.neutral_dest_value
        return 0

    def frontier_gain(self, obs, r: int, c: int) -> int:
        H, W = obs.H, obs.W
        count = 0
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if is_passable(obs.type_grid[nr][nc]) and obs.owner_grid[nr][nc] != 1:
                    count += 1
        return count

    def BeliefState(self):
        return _BeliefState(self)

    def best_capture_move(self, obs, general_pos, min_score: int = 0):
        H, W = obs.H, obs.W
        best_score = min_score
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue
                if not self.can_use_as_source(obs, r, c, 0, general_pos, obs.turn):
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not is_passable(obs.type_grid[nr][nc]):
                        continue
                    dest_owner = obs.owner_grid[nr][nc]
                    dest_type = obs.type_grid[nr][nc]
                    dest_army = obs.army_grid[nr][nc]
                    if src_army <= dest_army + 1:
                        continue
                    val = self.dest_value(dest_owner, dest_type)
                    if val == 0:
                        continue
                    score = 100 * val + 15 * self.frontier_gain(obs, nr, nc) - src_army
                    if score > best_score:
                        best_score = score
                        best_move = (0, r, c, d, 0)
        return best_move

    def win_check_w1(self, obs, enemy_general, general_pos):
        if enemy_general is None:
            return None
        tr, tc = enemy_general
        H, W = obs.H, obs.W
        target_army = enemy_general_army(obs, enemy_general)
        for d, (dr, dc) in enumerate(DIRECTIONS):
            sr, sc = tr - dr, tc - dc
            if not (0 <= sr < H and 0 <= sc < W):
                continue
            if obs.owner_grid[sr][sc] != 1:
                continue
            src_army = obs.army_grid[sr][sc]
            if src_army <= target_army + 1:
                continue
            if not self.can_use_as_source(obs, sr, sc, 0, general_pos, obs.turn):
                continue
            return (0, sr, sc, d, 0)
        return None

    def win_check_w2(self, obs, enemy_general, general_pos):
        if obs.turn < self.config.deathtouch_turn or enemy_general is None:
            return None
        tr, tc = enemy_general
        H, W = obs.H, obs.W
        for d, (dr, dc) in enumerate(DIRECTIONS):
            sr, sc = tr - dr, tc - dc
            if not (0 <= sr < H and 0 <= sc < W):
                continue
            if obs.owner_grid[sr][sc] != 1:
                continue
            if obs.army_grid[sr][sc] < 2:
                continue
            if not self.can_use_as_source(obs, sr, sc, 0, general_pos, obs.turn):
                continue
            return (0, sr, sc, d, 0)
        return None

    def chase_defence(self, obs, general_pos):
        if obs.turn < self.config.defend_from or general_pos is None:
            return None
        gr, gc = general_pos
        H, W = obs.H, obs.W
        for dr, dc in DIRECTIONS:
            er, ec = gr + dr, gc + dc
            if not (0 <= er < H and 0 <= ec < W):
                continue
            if obs.owner_grid[er][ec] != 2:
                continue
            e_army = obs.army_grid[er][ec]
            for _d2, (dr2, dc2) in enumerate(DIRECTIONS):
                sr, sc = er - dr2, ec - dc2
                if not (0 <= sr < H and 0 <= sc < W):
                    continue
                if (sr, sc) == general_pos:
                    continue
                if obs.owner_grid[sr][sc] != 1:
                    continue
                if obs.army_grid[sr][sc] <= e_army + 1:
                    continue
                atk_dir = direction_from_to(sr, sc, er, ec)
                if atk_dir is None:
                    continue
                return (0, sr, sc, atk_dir, 0)
        return None

    def march_toward_frontier(self, obs, general_pos):
        H, W = obs.H, obs.W
        dist = [[-1] * W for _ in range(H)]
        q = deque()
        for r in range(H):
            for c in range(W):
                if is_capturable(obs.owner_grid[r][c], obs.type_grid[r][c]):
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
                if not is_passable(obs.type_grid[nr][nc]):
                    continue
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))

        best_army = -1
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue
                if not self.can_use_as_source(obs, r, c, 0, general_pos, obs.turn):
                    continue
                here = dist[r][c]
                if here <= 0:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not is_passable(obs.type_grid[nr][nc]):
                        continue
                    if obs.owner_grid[nr][nc] != 1:
                        continue
                    if dist[nr][nc] < 0 or dist[nr][nc] >= here:
                        continue
                    if src_army > best_army:
                        best_army = src_army
                        best_move = (0, r, c, d, 0)
        return best_move

    def probe_move(self, obs, general_pos, candidates, active_probes=None):
        if not candidates:
            return None
        if active_probes is not None and len(active_probes) >= 2:
            return None
        H, W = obs.H, obs.W
        sample = self.config.probe_candidate_sample
        cand_list = list(candidates)[:sample]
        best_ratio = -1.0
        best_target = None
        for cell in cand_list:
            tr, tc = cell
            steps = bfs_distances(obs, cell, owned_only=False)
            min_steps = None
            for r in range(H):
                for c in range(W):
                    if obs.owner_grid[r][c] != 1:
                        continue
                    d = steps[r][c]
                    if d >= 0 and (min_steps is None or d < min_steps):
                        min_steps = d
            if min_steps is None:
                continue
            gain = reveal_gain_at(obs, candidates, tr, tc)
            ratio = gain / (1 + min_steps)
            if ratio > best_ratio:
                best_ratio = ratio
                best_target = cell
        if best_target is None:
            return None

        dist_from_target = bfs_distances(obs, best_target, owned_only=False)
        best_army = -1
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army < 4:
                    continue
                if not self.can_use_as_source(obs, r, c, 1, general_pos, obs.turn):
                    continue
                here = dist_from_target[r][c]
                if here <= 0:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not is_passable(obs.type_grid[nr][nc]):
                        continue
                    nd = dist_from_target[nr][nc]
                    if nd < 0 or nd >= here:
                        continue
                    if src_army > best_army:
                        best_army = src_army
                        best_move = (0, r, c, d, 1)
        return best_move

    def sentry_convey(self, obs, general_pos, target_army: int = 12):
        if obs.turn < self.config.sentry_from or general_pos is None:
            return None
        gr, gc = general_pos
        H, W = obs.H, obs.W
        best_neighbor = None
        best_army = -1
        for dr, dc in DIRECTIONS:
            nr, nc = gr + dr, gc + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if obs.owner_grid[nr][nc] != 1:
                continue
            a = obs.army_grid[nr][nc]
            if a > best_army:
                best_army = a
                best_neighbor = (nr, nc)
        if best_neighbor is None or best_army >= target_army:
            return None

        dist = bfs_distances(obs, best_neighbor, owned_only=True)
        best_src_army = -1
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue
                if (r, c) == general_pos:
                    continue
                if not self.can_use_as_source(obs, r, c, 0, general_pos, obs.turn):
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
                    if src_army > best_src_army:
                        best_src_army = src_army
                        best_move = (0, r, c, d, 0)
        return best_move

    def final_approach_move(self, obs, enemy_general, general_pos):
        if enemy_general is None:
            return None
        H, W = obs.H, obs.W
        dist = bfs_distances(obs, enemy_general, owned_only=False)
        best_dist = None
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.army_grid[r][c] < 2:
                    continue
                if not self.can_use_as_source(obs, r, c, 0, general_pos, obs.turn):
                    continue
                here = dist[r][c]
                if here <= 0:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not is_passable(obs.type_grid[nr][nc]):
                        continue
                    if dist[nr][nc] < 0 or dist[nr][nc] >= here:
                        continue
                    if best_dist is None or here < best_dist:
                        best_dist = here
                        best_move = (0, r, c, d, 0)
        return best_move

    def eta_to_general(self, obs, enemy_general):
        if enemy_general is None:
            return None
        H, W = obs.H, obs.W
        dist = bfs_distances(obs, enemy_general, owned_only=False)
        best = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.army_grid[r][c] < 2:
                    continue
                d = dist[r][c]
                if d >= 0 and (best is None or d < best):
                    best = d
        return best


class _BeliefState:
    def __init__(self, ctx: StrategyContext):
        self._ctx = ctx
        self.enemy_general = None
        self.first_sighting_turn = None
        self.ever_seen: set[tuple[int, int]] = set()
        self.candidates: set[tuple[int, int]] | None = None
        self._initialized = False

    def update(self, obs, own_general):
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if obs.type_grid[r][c] != 0:
                    self.ever_seen.add((r, c))
                if obs.owner_grid[r][c] == 2 and obs.type_grid[r][c] == 4:
                    if self.enemy_general is None:
                        self.first_sighting_turn = obs.turn
                    self.enemy_general = (r, c)

        if not self._initialized and own_general is not None:
            self._init_candidates(obs, own_general)
            self._initialized = True

        if self.candidates is not None:
            remove = []
            for cell in self.candidates:
                if cell in self.ever_seen and cell != self.enemy_general:
                    remove.append(cell)
            for cell in remove:
                self.candidates.discard(cell)

    def _init_candidates(self, obs, own_general):
        H, W = obs.H, obs.W
        min_dist = self._ctx.config.min_general_distance
        dist = bfs_distances(obs, own_general, owned_only=False)
        self.candidates = set()
        for r in range(H):
            for c in range(W):
                if not is_passable(obs.type_grid[r][c]):
                    continue
                d = dist[r][c]
                if d >= 0 and d >= min_dist:
                    self.candidates.add((r, c))
        if not self.candidates:
            for r in range(H):
                for c in range(W):
                    if is_passable(obs.type_grid[r][c]) and (r, c) not in self.ever_seen:
                        self.candidates.add((r, c))
