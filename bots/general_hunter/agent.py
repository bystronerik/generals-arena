"""
general_hunter — deathtouch beeline bot.

Expands greedily like `expander_python` for the whole early/mid game.
Whenever the enemy general is ever glimpsed, its cell is remembered
forever — generals never move, so a single sighting is exact for the rest
of the game even after it fades back into fog.

From turn 800 (deathtouch, RULES.md section 07 / docs/competition/
deathtouch.md), any move that *executes* onto the enemy general's tile
wins instantly, however large the defending army is. So once turn >= 800
and the general's cell is known, this bot switches to beelining a runner
stack toward it: attack immediately if already adjacent, otherwise
advance the owned cell closest (by BFS distance) to the target one step
along the shortest path.

See docs/bots/general-hunter.md and
docs/research/experiments/003-general-hunter-deathtouch-beeline.md.
"""
from collections import deque

# A no-op action — used when no valid move exists.
PASS = (1, 0, 0, 0, 0)

# (dr, dc) offsets for direction codes 0..3
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

DEATHTOUCH_TURN = 800


def _is_passable(t):
    return t != 2 and t != 5


class Agent:
    """Greedy expansion; deathtouch-era beeline once the enemy general is known."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.enemy_general = None  # (r, c) once ever sighted; never moves.

    def act(self, obs):
        self._update_enemy_general_sighting(obs)

        if obs.turn >= DEATHTOUCH_TURN and self.enemy_general is not None:
            hunt_move = self._hunt(obs)
            if hunt_move is not None:
                return hunt_move

        return self._expand(obs)

    def _update_enemy_general_sighting(self, obs):
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 2 and obs.type_grid[r][c] == 4:
                    self.enemy_general = (r, c)
                    return

    def _hunt(self, obs):
        H, W = obs.H, obs.W
        tr, tc = self.enemy_general

        # Immediate execute: an owned neighbor with army >= 2 wins outright,
        # no matter the defender's army, once deathtouch is active.
        for d, (dr, dc) in enumerate(DIRECTIONS):
            r, c = tr - dr, tc - dc
            if not (0 <= r < H and 0 <= c < W):
                continue
            if obs.owner_grid[r][c] == 1 and obs.army_grid[r][c] >= 2:
                return (0, r, c, d, 0)

        # Otherwise: BFS distance field from the target, then advance the
        # owned runner cell closest to the target one step further in.
        dist = [[-1] * W for _ in range(H)]
        dist[tr][tc] = 0
        q = deque([(tr, tc)])
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

        best_dist = None
        best_move = None
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.army_grid[r][c] < 2:
                    continue
                here = dist[r][c]
                if here <= 0:
                    continue  # unreachable, or this cell is the target already
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue
                    if dist[nr][nc] < 0 or dist[nr][nc] >= here:
                        continue
                    if best_dist is None or here < best_dist:
                        best_dist = here
                        best_move = (0, r, c, d, 0)
        return best_move

    def _expand(self, obs):
        best_score = -1.0
        best_move = None
        first_valid = None

        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army <= 1:
                    continue

                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue

                    move = (0, r, c, d, 0)
                    if first_valid is None:
                        first_valid = move

                    dest_owner = obs.owner_grid[nr][nc]
                    dest_type = obs.type_grid[nr][nc]
                    dest_army = obs.army_grid[nr][nc]
                    if src_army <= dest_army + 1:
                        continue

                    is_opp = dest_owner == 2
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
                        best_move = move

        if best_move is not None:
            return best_move
        if first_valid is not None:
            return first_valid
        return PASS
