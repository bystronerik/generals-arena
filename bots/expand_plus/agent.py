"""
expand_plus — frontier-directed expansion.

Baseline greedy capture (score = army, boosted for expansion/opponent
targets), same as `expander_python`. The difference is the fallback used
when no adjacent capture exists: instead of taking an arbitrary legal move,
march the largest owned stack one step toward the *nearest* capturable
tile (a BFS gradient over the whole board), so idle turns still consolidate
army toward the frontier instead of wandering.

See docs/bots/expand-plus.md and
docs/research/experiments/001-expand-plus-frontier-march.md.
"""
from collections import deque

# A no-op action — used when no valid move exists.
PASS = (1, 0, 0, 0, 0)

# (dr, dc) offsets for direction codes 0..3
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _is_passable(t):
    # Mountains (2) and fogged-structures (5) are impassable.
    return t != 2 and t != 5


def _is_capturable(owner, cell_type):
    # A tile we could eventually take: visible neutral or opponent-owned.
    if owner == 2:
        return True
    return owner == 0 and cell_type not in (0, 5)


class Agent:
    """Greedy capture, falling back to a BFS march toward the frontier."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W

    def act(self, obs):
        H, W = obs.H, obs.W
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

        march_move = self._march_toward_frontier(obs)
        if march_move is not None:
            return march_move

        first_valid = self._any_valid_move(obs)
        if first_valid is not None:
            return first_valid
        return PASS

    def _march_toward_frontier(self, obs):
        """Move the largest owned stack one step closer to the nearest
        capturable tile, via a multi-source BFS distance field seeded from
        every visible capturable tile."""
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
                    continue  # already on (or is) a capturable tile
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue
                    if obs.owner_grid[nr][nc] != 1:
                        continue  # only reinforce our own path while marching
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
