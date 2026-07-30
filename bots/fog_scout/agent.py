"""
fog_scout — systematic fog-of-war probing.

See docs/research/strategies/fog_scout.md and docs/bots/fog-scout.md.
"""
from collections import deque

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_passable(t):
    return t != 2 and t != 5


def _is_visible(cell_type):
    return cell_type != 0


class Agent:
    """Probe fog systematically; march toward unrevealed terrain."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.ever_seen = [[False] * W for _ in range(H)]

    def act(self, obs):
        H, W = obs.H, obs.W
        self._update_ever_seen(obs)

        general_attack = self._attack_enemy_general(obs)
        if general_attack is not None:
            return general_attack

        best_score = -1.0
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
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
                    dest_army = obs.army_grid[nr][nc]
                    if src_army <= dest_army + 1:
                        continue

                    dest_owner = obs.owner_grid[nr][nc]
                    dest_type = obs.type_grid[nr][nc]
                    is_opp = dest_owner == 2
                    is_fog = dest_type == 0 or not self.ever_seen[nr][nc]
                    is_neutral = dest_owner == 0 and dest_type not in (0, 5)
                    if not (is_opp or is_fog or is_neutral):
                        continue

                    score = float(src_army) * 10.0
                    if is_fog:
                        score *= 3.0
                    if is_opp:
                        score *= 2.0
                    if score > best_score:
                        best_score = score
                        best_move = (0, r, c, d, 0)

        if best_move is not None:
            return best_move

        march_move = self._march_toward_fog(obs)
        if march_move is not None:
            return march_move

        first_valid = self._any_valid_move(obs)
        if first_valid is not None:
            return first_valid
        return PASS

    def _update_ever_seen(self, obs):
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if _is_visible(obs.type_grid[r][c]):
                    self.ever_seen[r][c] = True

    def _attack_enemy_general(self, obs):
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if obs.type_grid[nr][nc] != 4 or obs.owner_grid[nr][nc] != 2:
                        continue
                    if src_army <= obs.army_grid[nr][nc] + 1:
                        continue
                    return (0, r, c, d, 0)
        return None

    def _march_toward_fog(self, obs):
        H, W = obs.H, obs.W
        dist = [[-1] * W for _ in range(H)]
        q = deque()
        for r in range(H):
            for c in range(W):
                if self.ever_seen[r][c]:
                    continue
                if not _is_passable(obs.type_grid[r][c]):
                    continue
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
