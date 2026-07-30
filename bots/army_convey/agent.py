"""
army_convey — funnel interior armies toward the expansion frontier.

See docs/research/strategies/army_convey.md and docs/bots/army-convey.md.
"""
from collections import deque

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

CONVEY_MIN_ARMY = 3
FRONTIER_NEIGHBOR_WEIGHT = 1.5
OPPONENT_CAPTURE_MULT = 3.0
STALEMATE_BREAK_TURN = 600
STALEMATE_OPPONENT_MULT = 2.0
STALEMATE_OPPONENT_MARGIN = 0


def _is_passable(t):
    return t != 2 and t != 5


def _is_unowned(owner, cell_type):
    if owner == 2:
        return True
    return owner == 0 and cell_type not in (0, 5)


class Agent:
    """Concentrate interior stacks at frontier tips for breakthrough captures."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W

    def act(self, obs):
        frontier = self._frontier_cells(obs)

        best_capture_score = -1.0
        best_capture = None
        for r, c in frontier:
            src_army = obs.army_grid[r][c]
            if src_army <= 1:
                continue
            for d, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                    continue
                if not _is_passable(obs.type_grid[nr][nc]):
                    continue
                if obs.owner_grid[nr][nc] == 1:
                    continue
                dest_army = obs.army_grid[nr][nc]
                margin = 1
                if obs.turn >= STALEMATE_BREAK_TURN and obs.owner_grid[nr][nc] == 2:
                    margin = STALEMATE_OPPONENT_MARGIN
                if src_army <= dest_army + margin:
                    continue
                score = float(src_army) * 10.0
                score += self._frontier_neighbor_armies(obs, r, c) * FRONTIER_NEIGHBOR_WEIGHT
                if obs.owner_grid[nr][nc] == 2:
                    opp_mult = OPPONENT_CAPTURE_MULT
                    if obs.turn >= STALEMATE_BREAK_TURN:
                        opp_mult *= STALEMATE_OPPONENT_MULT
                    score *= opp_mult
                if score > best_capture_score:
                    best_capture_score = score
                    best_capture = (0, r, c, d, 0)

        if best_capture is not None:
            return best_capture

        dist = self._convey_distance_field(obs, frontier)
        best_convey_score = -1.0
        best_convey = None
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if (r, c) in frontier:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army < CONVEY_MIN_ARMY:
                    continue
                here_dist = dist[r][c]
                if here_dist <= 0:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if obs.owner_grid[nr][nc] != 1:
                        continue
                    if dist[nr][nc] < 0 or dist[nr][nc] >= here_dist:
                        continue
                    score = float(src_army) / float(here_dist + 1)
                    if score > best_convey_score:
                        best_convey_score = score
                        best_convey = (0, r, c, d, 0)

        if best_convey is not None:
            return best_convey

        gather_move = self._frontier_gathering(obs, frontier)
        if gather_move is not None:
            return gather_move

        first_valid = self._any_valid_move(obs)
        if first_valid is not None:
            return first_valid
        return PASS

    def _frontier_neighbor_armies(self, obs, r, c):
        total = 0.0
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                continue
            if obs.owner_grid[nr][nc] == 1:
                total += float(obs.army_grid[nr][nc])
        return total

    def _frontier_cells(self, obs):
        H, W = obs.H, obs.W
        frontier = set()
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue
                    if _is_unowned(obs.owner_grid[nr][nc], obs.type_grid[nr][nc]):
                        frontier.add((r, c))
                        break
        return frontier

    def _convey_distance_field(self, obs, frontier):
        H, W = obs.H, obs.W
        dist = [[-1] * W for _ in range(H)]
        q = deque()
        for cell in frontier:
            r, c = cell
            dist[r][c] = 0
            q.append(cell)
        while q:
            r, c = q.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if dist[nr][nc] != -1:
                    continue
                if obs.owner_grid[nr][nc] != 1:
                    continue
                if not _is_passable(obs.type_grid[nr][nc]):
                    continue
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
        return dist

    def _frontier_gathering(self, obs, frontier):
        H, W = obs.H, obs.W
        best_army = -1
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if (r, c) in frontier:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) not in frontier:
                        continue
                    if obs.owner_grid[nr][nc] != 1:
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
