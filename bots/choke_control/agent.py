"""
choke_control — corridor claim and hold.

Same greedy capture + BFS frontier march as expand_plus. One change:
choke-aware capture scoring plus hold/deny rules for owned gate cells.

See docs/bots/choke-control.md and
docs/research/experiments/009-choke-control-corridor-hold.md.
"""
from collections import deque

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

W_CHOKE = 0.5
W_OPPRESS = 1.0
K = 6
HOLD_MIN_ARMY = 10
OVERWHELM_MULT = 3
REINFORCE_BONUS = 15.0
FOG_DECAY = 0.5
SPLIT_MIN_ARMY = 8


def _is_passable(t):
    return t != 2 and t != 5


def _is_visible(t):
    return t != 0


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
    """Choke-weighted capture with hold/deny; expand_plus fallback."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W

    def act(self, obs):
        H, W = obs.H, obs.W
        dist = self._build_frontier_dist(obs)
        narrow = self._narrowness(obs)
        gate = self._gate_cells(obs)
        chokes = self._choke_scores(obs, narrow, gate, dist)
        held = self._held_chokes(obs, chokes, dist)
        threat = self._threat_map(obs, held, K)

        best_score = -1.0
        best_move = None

        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                army = obs.army_grid[r][c]
                if army < 2:
                    continue

                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue

                    dest_owner = obs.owner_grid[nr][nc]
                    dest_type = obs.type_grid[nr][nc]
                    dest_army = obs.army_grid[nr][nc]

                    if (r, c) in held:
                        if not self._allowed_from_hold(
                            obs, r, c, nr, nc, army, held, gate, chokes, threat
                        ):
                            continue

                    if _is_capturable(dest_owner, dest_type) and army - 1 > dest_army:
                        is_opp = dest_owner == 2
                        choke_val = chokes.get((nr, nc), 0.0)
                        score = _base_score(army - 1, is_opp) * (1.0 + W_CHOKE * choke_val)
                        split = 1 if self._two_deep_hold(
                            obs, r, c, nr, nc, army, dest_army, narrow, gate
                        ) else 0
                        if score > best_score:
                            best_score = score
                            best_move = (0, r, c, d, split)
                        continue

                    if dest_owner == 1 and (nr, nc) in held:
                        score = REINFORCE_BONUS * chokes.get((nr, nc), 0.0)
                        if score > best_score:
                            best_score = score
                            best_move = (0, r, c, d, 0)

        if best_move is not None:
            return best_move

        march_move = self._march_toward_frontier(obs, dist, held, threat)
        if march_move is not None:
            return march_move

        first_valid = self._any_valid_move(obs)
        if first_valid is not None:
            return first_valid
        return PASS

    def _narrowness(self, obs):
        H, W = obs.H, obs.W
        narrow = [[0] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                if not _is_passable(obs.type_grid[r][c]):
                    continue
                count = 0
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        count += 1
                        continue
                    t = obs.type_grid[nr][nc]
                    if t in (0, 2, 5):
                        count += 1
                narrow[r][c] = count
        return narrow

    def _passable_visible_neighbors(self, obs, r, c):
        H, W = obs.H, obs.W
        neighbors = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            t = obs.type_grid[nr][nc]
            if _is_passable(t) and _is_visible(t):
                neighbors.append((nr, nc))
        return neighbors

    def _gate_cells(self, obs):
        H, W = obs.H, obs.W
        gate = [[False] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                if not _is_passable(obs.type_grid[r][c]):
                    continue
                neighbors = self._passable_visible_neighbors(obs, r, c)
                if len(neighbors) < 2:
                    continue
                if self._neighbor_components(neighbors) >= 2:
                    gate[r][c] = True
        return gate

    def _neighbor_components(self, neighbors):
        if not neighbors:
            return 0
        remaining = set(neighbors)
        components = 0
        while remaining:
            start = remaining.pop()
            stack = [start]
            while stack:
                r, c = stack.pop()
                for dr, dc in DIRECTIONS:
                    nbr = (r + dr, c + dc)
                    if nbr in remaining:
                        remaining.remove(nbr)
                        stack.append(nbr)
            components += 1
        return components

    def _narrow_weight(self, obs, r, c, narrow_val):
        if narrow_val < 2:
            return 0.0
        if narrow_val >= 3:
            return 1.0
        passable_dirs = []
        H, W = obs.H, obs.W
        for i, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and _is_passable(obs.type_grid[nr][nc]):
                if _is_visible(obs.type_grid[nr][nc]):
                    passable_dirs.append(i)
        if len(passable_dirs) != 2:
            return 1.0
        a, b = passable_dirs
        if (a + b) % 2 == 0:
            return 1.0
        return 0.5

    def _fog_nearby(self, obs, r, c):
        H, W = obs.H, obs.W
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and obs.type_grid[nr][nc] == 0:
                    return True
        return False

    def _frontier_relevant(self, obs, r, c, dist):
        owned_side = None
        beyond = False
        for nr, nc in self._passable_visible_neighbors(obs, r, c):
            d = dist[nr][nc]
            if d < 0:
                continue
            owner = obs.owner_grid[nr][nc]
            if owner == 1:
                if owned_side is None or d > owned_side:
                    owned_side = d
            elif _is_capturable(owner, obs.type_grid[nr][nc]):
                beyond = True
                if owned_side is None or d < owned_side:
                    owned_side = d
        return beyond

    def _opponent_near(self, obs, r, c, radius):
        H, W = obs.H, obs.W
        q = deque([(r, c, 0)])
        seen = {(r, c)}
        while q:
            cr, cc, steps = q.popleft()
            if steps > 0 and obs.owner_grid[cr][cc] == 2:
                return True
            if steps >= radius:
                continue
            for dr, dc in DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in seen:
                    continue
                if not _is_passable(obs.type_grid[nr][nc]):
                    continue
                seen.add((nr, nc))
                q.append((nr, nc, steps + 1))
        return False

    def _choke_scores(self, obs, narrow, gate, dist):
        H, W = obs.H, obs.W
        chokes = {}
        for r in range(H):
            for c in range(W):
                if not gate[r][c]:
                    continue
                nw = self._narrow_weight(obs, r, c, narrow[r][c])
                if nw <= 0:
                    continue
                if not self._frontier_relevant(obs, r, c, dist):
                    continue
                score = nw
                if self._opponent_near(obs, r, c, K):
                    score *= 1.0 + W_OPPRESS
                if self._fog_nearby(obs, r, c):
                    score *= FOG_DECAY
                chokes[(r, c)] = score
        return chokes

    def _held_chokes(self, obs, chokes, dist):
        held = set()
        for (r, c), score in chokes.items():
            if score <= 0:
                continue
            if obs.owner_grid[r][c] != 1:
                continue
            if self._frontier_relevant(obs, r, c, dist):
                held.add((r, c))
        return held

    def _threat_map(self, obs, held, radius):
        H, W = obs.H, obs.W
        threat = {cell: 0 for cell in held}
        for hr, hc in held:
            q = deque([(hr, hc, 0)])
            seen = {(hr, hc)}
            max_opp = 0
            while q:
                cr, cc, steps = q.popleft()
                if steps > 0 and obs.owner_grid[cr][cc] == 2:
                    max_opp = max(max_opp, obs.army_grid[cr][cc])
                if steps >= radius:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if (nr, nc) in seen:
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc, steps + 1))
            threat[(hr, hc)] = max_opp
        return threat

    def _allowed_from_hold(self, obs, sr, sc, dr, dc, army, held, gate, chokes, threat):
        if army < HOLD_MIN_ARMY:
            return True
        if (dr, dc) in held or gate[dr][dc]:
            return True
        if chokes.get((dr, dc), 0.0) > 0:
            return True
        t = threat.get((sr, sc), 0)
        if t > 0 and army - 1 >= OVERWHELM_MULT * t:
            return True
        return False

    def _two_deep_hold(self, obs, sr, sc, dr, dc, army, dest_army, narrow, gate):
        if army < SPLIT_MIN_ARMY:
            return False
        if narrow[sr][sc] < 2 and not gate[sr][sc]:
            return False
        if army // 2 <= dest_army:
            return False
        if not _is_capturable(obs.owner_grid[dr][dc], obs.type_grid[dr][dc]):
            return False
        return True

    def _build_frontier_dist(self, obs):
        H, W = obs.H, obs.W
        dist = [[-1] * W for _ in range(H)]
        q = deque()
        for r in range(H):
            for c in range(W):
                if _is_capturable(obs.owner_grid[r][c], obs.type_grid[r][c]):
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
                if not _is_passable(obs.type_grid[nr][nc]):
                    continue
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
        return dist

    def _march_toward_frontier(self, obs, dist, held, threat):
        H, W = obs.H, obs.W
        if not any(dist[r][c] >= 0 for r in range(H) for c in range(W)):
            return None

        best_army = -1
        best_move = None
        best_reinforce = -1
        reinforce_move = None

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

                    move = (0, r, c, d, 0)
                    if src_army > best_army:
                        best_army = src_army
                        best_move = move

                    if (nr, nc) in held and threat.get((nr, nc), 0) > 0:
                        if src_army > best_reinforce:
                            best_reinforce = src_army
                            reinforce_move = move

        if reinforce_move is not None:
            return reinforce_move
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
