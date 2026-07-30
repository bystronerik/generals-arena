"""
splitter — half-army multi-front expansion.

Same greedy capture + BFS frontier march as expand_plus. One change: score
both split=0 and split=1 candidates and pick the best, so large stacks
can capture with half army while garrisoning the source or opening a
second front.

See docs/bots/splitter.md and
docs/research/experiments/008-splitter-half-army-fronts.md.
"""
from collections import deque

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SPLIT_MIN_ARMY = 8
SPLIT_MARGIN = 2
OVERKILL_MIN = 4
W_SECOND_FRONT = 0.5
W_GARRISON = 0.5
W_OVERKILL = 0.25
DEATHTOUCH_TURN = 800


def _is_passable(t):
    return t != 2 and t != 5


def _is_capturable(owner, cell_type):
    if owner == 2:
        return True
    return owner == 0 and cell_type not in (0, 5)


def _base_score(moved, is_opp):
    score = float(moved) * 10.0
    if is_opp:
        score *= 2.0
    return score


class Agent:
    """Greedy capture with split-flag arbitration; expand_plus fallback."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.enemy_general = None

    def act(self, obs):
        self._update_enemy_general_sighting(obs)

        best_score = -1.0
        best_move = None

        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                army = obs.army_grid[r][c]
                if army < 2:
                    continue

                extra_fronts = self._extra_fronts(obs, r, c)
                contested = self._contested(obs, r, c)
                is_general = obs.type_grid[r][c] == 4

                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue

                    dest_owner = obs.owner_grid[nr][nc]
                    dest_type = obs.type_grid[nr][nc]
                    if not _is_capturable(dest_owner, dest_type):
                        continue

                    dest_army = obs.army_grid[nr][nc]
                    need = dest_army + 1
                    is_opp = dest_owner == 2
                    is_neutral = dest_owner == 0
                    overkill = army / need if need > 0 else 0.0

                    if army - 1 > dest_army:
                        score = _base_score(army - 1, is_opp)
                        best_score, best_move = self._maybe_better(
                            best_score, best_move, score, (0, r, c, d, 0)
                        )

                    if self._can_split(army, need, is_general) and self._should_split(
                        obs, army, need, extra_fronts, contested, overkill,
                        is_neutral, is_opp,
                    ):
                        bonus = self._split_bonus(
                            extra_fronts, contested, overkill, is_neutral
                        )
                        score = _base_score(army // 2, is_opp) * bonus
                        if score > best_score:
                            best_score = score
                            best_move = (0, r, c, d, 1)

        if best_move is not None:
            return best_move

        march_move = self._march_toward_frontier(obs)
        if march_move is not None:
            return march_move

        first_valid = self._any_valid_move(obs)
        if first_valid is not None:
            return first_valid
        return PASS

    def _update_enemy_general_sighting(self, obs):
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 2 and obs.type_grid[r][c] == 4:
                    self.enemy_general = (r, c)
                    return

    def _maybe_better(self, best_score, best_move, score, move):
        if score > best_score:
            return score, move
        if score == best_score and move[4] == 0:
            return score, move
        return best_score, best_move

    def _can_split(self, army, need, is_general):
        if is_general:
            return False
        if army < SPLIT_MIN_ARMY:
            return False
        return army // 2 >= need + SPLIT_MARGIN

    def _should_split(
        self, obs, army, need, extra_fronts, contested, overkill, is_neutral, is_opp
    ):
        if is_opp:
            return False
        if extra_fronts >= 1:
            return True
        if contested:
            return True
        if is_neutral and overkill >= OVERKILL_MIN:
            return True
        if (
            obs.turn >= DEATHTOUCH_TURN
            and self.enemy_general is not None
            and extra_fronts >= 1
        ):
            return True
        return False

    def _split_bonus(self, extra_fronts, contested, overkill, is_neutral):
        bonus = 1.0
        bonus += W_SECOND_FRONT * extra_fronts
        bonus += W_GARRISON * contested
        if is_neutral and overkill >= OVERKILL_MIN:
            bonus += W_OVERKILL
        return bonus

    def _extra_fronts(self, obs, r, c):
        count = 0
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                continue
            if not _is_passable(obs.type_grid[nr][nc]):
                continue
            if _is_capturable(obs.owner_grid[nr][nc], obs.type_grid[nr][nc]):
                count += 1
        return count

    def _contested(self, obs, r, c):
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                return 1
            owner = obs.owner_grid[nr][nc]
            cell_type = obs.type_grid[nr][nc]
            if owner == 2:
                return 1
            if cell_type == 0:
                return 1
        return 0

    def _march_toward_frontier(self, obs):
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

    def _any_valid_move(self, obs):
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.army_grid[r][c] <= 1:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and _is_passable(obs.type_grid[nr][nc]):
                        return (0, r, c, d, 0)
        return None
