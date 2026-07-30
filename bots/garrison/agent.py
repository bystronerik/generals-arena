"""
garrison — phase-based general reserve and route-aware defense.

Protects the own general with a reserve floor, reinforcement toward the
general, and deathtouch intercept logic after turn 800. Expands only when
defense is satisfied.

See docs/research/strategies/garrison.md and
docs/research/experiments/006-garrison-phase-reserve.md.
"""
import math
from collections import deque

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

DEATHTOUCH_TURN = 800
STALE_ENEMY_TURNS = 20
PRESSURE_CAP = 60

PHASE1_END = 199
PHASE2_END = 599
PHASE3_END = 799

PHASE_FLOORS = {1: 8, 2: 16, 3: 24, 4: 12}
PHASE_SAFE_RADIUS = {1: 2, 2: 3, 3: 4, 4: 4}
PHASE_SCAN_RADIUS = {1: 6, 2: 8, 3: 10, 4: 10}

PROXIMITY_WEIGHTS = {1: 8, 2: 5, 3: 3}


def _is_passable(t):
    return t != 2 and t != 5


def _is_visible_neutral(owner, cell_type):
    return owner == 0 and cell_type not in (0, 5)


def _phase(turn):
    if turn <= PHASE1_END:
        return 1
    if turn <= PHASE2_END:
        return 2
    if turn <= PHASE3_END:
        return 3
    return 4


class Agent:
    """Reserve on the general; reinforce, expand, consolidate in that order."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.general_pos = None
        self.enemy_general_pos = None
        self.last_seen_enemy = {}
        self.last_defense_move_turn = -1
        self.last_expansion_target = None

    def act(self, obs):
        self._locate_general(obs)
        if self.general_pos is None:
            return PASS

        self._update_enemy_general(obs)
        self._update_last_seen_enemy(obs)

        phase = _phase(obs.turn)
        own_dist = self._owned_bfs(obs, self.general_pos)
        pass_dist = self._passable_bfs(obs, self.general_pos)
        pressure = self._compute_pressure(obs, pass_dist, phase)
        required = self._required_general_army(obs, phase, pressure)
        army_on_gen = obs.army_grid[self.general_pos[0]][self.general_pos[1]]
        deficit = max(0, required - army_on_gen)
        safe_radius = PHASE_SAFE_RADIUS[phase]

        emergency = self._score_emergency(obs, own_dist, pass_dist, phase, deficit, required)
        if emergency is not None:
            return emergency

        threat_in_safe = self._threat_in_safe_radius(obs, pass_dist, safe_radius)
        if deficit > 0 or threat_in_safe:
            reinforcement = self._score_reinforcement(
                obs, own_dist, pass_dist, safe_radius, deficit
            )
            if reinforcement is not None:
                self.last_defense_move_turn = obs.turn
                return reinforcement

        expansion = self._score_expansion(
            obs, own_dist, pass_dist, phase, required, deficit, safe_radius
        )
        if expansion is not None:
            return expansion

        consolidation = self._score_consolidation(obs, own_dist, safe_radius)
        if consolidation is not None:
            return consolidation

        return PASS

    def _locate_general(self, obs):
        if self.general_pos is not None:
            return
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 1 and obs.type_grid[r][c] == 4:
                    self.general_pos = (r, c)
                    return

    def _update_enemy_general(self, obs):
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 2 and obs.type_grid[r][c] == 4:
                    self.enemy_general_pos = (r, c)
                    return

    def _update_last_seen_enemy(self, obs):
        seen = set()
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 2:
                    seen.add((r, c))
                    self.last_seen_enemy[(r, c)] = (obs.turn, obs.army_grid[r][c])
        stale = []
        for cell, (turn, _) in self.last_seen_enemy.items():
            if obs.turn - turn > STALE_ENEMY_TURNS:
                stale.append(cell)
            elif cell not in seen:
                stale.append(cell)
        for cell in stale:
            del self.last_seen_enemy[cell]

    def _owned_bfs(self, obs, source):
        H, W = obs.H, obs.W
        dist = [[-1] * W for _ in range(H)]
        sr, sc = source
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
                if obs.owner_grid[nr][nc] != 1:
                    continue
                if not _is_passable(obs.type_grid[nr][nc]):
                    continue
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
        return dist

    def _passable_bfs(self, obs, source):
        H, W = obs.H, obs.W
        dist = [[-1] * W for _ in range(H)]
        sr, sc = source
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
                if not _is_passable(obs.type_grid[nr][nc]):
                    continue
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
        return dist

    def _proximity_weight(self, distance):
        if distance in PROXIMITY_WEIGHTS:
            return PROXIMITY_WEIGHTS[distance]
        return 1

    def _threat_contribution(self, enemy_army, dist_to_general):
        distance_penalty = 2 * max(0, dist_to_general - 1)
        return max(0, enemy_army - distance_penalty) * self._proximity_weight(dist_to_general)

    def _compute_pressure(self, obs, pass_dist, phase):
        scan = PHASE_SCAN_RADIUS[phase]
        total = 0.0
        gr, gc = self.general_pos

        for r in range(obs.H):
            for c in range(obs.W):
                d = pass_dist[r][c]
                if d < 0 or d > scan:
                    continue
                if obs.owner_grid[r][c] == 2:
                    total += self._threat_contribution(obs.army_grid[r][c], d)

        for (r, c), (turn, army) in self.last_seen_enemy.items():
            d = pass_dist[r][c]
            if d < 0 or d > scan:
                continue
            if (r, c) not in {(rr, cc) for rr in range(obs.H) for cc in range(obs.W)
                              if obs.owner_grid[rr][cc] == 2}:
                total += 0.25 * self._threat_contribution(army, d)

        return min(PRESSURE_CAP, total)

    def _required_general_army(self, obs, phase, pressure):
        floor = PHASE_FLOORS[phase]
        if obs.turn < DEATHTOUCH_TURN:
            return floor + math.ceil(pressure / 4)
        return floor

    def _threat_in_safe_radius(self, obs, pass_dist, safe_radius):
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 2:
                    continue
                d = pass_dist[r][c]
                if 0 <= d <= safe_radius:
                    return True
        return False

    def _iter_moves(self, obs):
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                src_army = obs.army_grid[r][c]
                if src_army < 2:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if not _is_passable(obs.type_grid[nr][nc]):
                        continue
                    yield (r, c, d, nr, nc, src_army)

    def _pick_best(self, candidates):
        if not candidates:
            return None
        candidates.sort(
            key=lambda x: (
                -x[0],
                -x[1],
                x[2],
                x[3][0],
                x[3][1],
                x[3][2],
            )
        )
        return candidates[0][3]

    def _score_emergency(self, obs, own_dist, pass_dist, phase, deficit, required):
        gr, gc = self.general_pos
        candidates = []
        deathtouch = obs.turn >= DEATHTOUCH_TURN

        for r, c, d, nr, nc, src_army in self._iter_moves(obs):
            dest_owner = obs.owner_grid[nr][nc]
            dest_army = obs.army_grid[nr][nc]
            dest_type = obs.type_grid[nr][nc]
            score = 0.0

            gen_dist = pass_dist[nr][nc] if pass_dist[nr][nc] >= 0 else 999
            enemy_adjacent_gen = gen_dist == 1 and dest_owner == 2

            if deathtouch and enemy_adjacent_gen and src_army > dest_army + 1:
                score += 90000

            if not deathtouch and enemy_adjacent_gen and src_army > dest_army + 1:
                score += 50000

            if dest_owner == 2 and dest_type == 4 and obs.turn < DEATHTOUCH_TURN:
                if src_army > dest_army + 1:
                    score += 100000

            if dest_owner == 1:
                here = own_dist[r][c]
                there = own_dist[nr][nc]
                if here > 0 and there >= 0 and there < here:
                    reduction = here - there
                    transferred = src_army - 1
                    score += 5000 * reduction + 100 * transferred

            if (r, c) == self.general_pos and (src_army - 1) < required:
                score -= 20000

            enemy_near = pass_dist[r][c] <= 2 or deficit > 0
            if score > 0 and enemy_near:
                candidates.append((score, src_army, own_dist[nr][nc] if own_dist[nr][nc] >= 0 else 999,
                                   (0, r, c, d, 0)))

        return self._pick_best(candidates)

    def _score_reinforcement(self, obs, own_dist, pass_dist, safe_radius, deficit):
        candidates = []
        for r, c, d, nr, nc, src_army in self._iter_moves(obs):
            if obs.owner_grid[nr][nc] != 1:
                continue
            here = own_dist[r][c]
            there = own_dist[nr][nc]
            if here <= 0 or there < 0 or there >= here:
                continue

            reduction = here - there
            transferred = src_army - 1
            score = 3000 * reduction + 20 * transferred - 30 * here

            if there <= safe_radius:
                score += 2000

            frontier_exposed = False
            for dr, dc in DIRECTIONS:
                er, ec = r + dr, c + dc
                if 0 <= er < obs.H and 0 <= ec < obs.W:
                    if obs.owner_grid[er][ec] == 2 and obs.army_grid[er][ec] >= src_army:
                        frontier_exposed = True
            if frontier_exposed:
                score -= 5000
            if src_army - 1 == 1:
                for dr, dc in DIRECTIONS:
                    er, ec = nr + dr, nc + dc
                    if 0 <= er < obs.H and 0 <= ec < obs.W and obs.owner_grid[er][ec] == 2:
                        score -= 4000

            candidates.append((score, src_army, there, (0, r, c, d, 0)))

        return self._pick_best(candidates)

    def _score_expansion(self, obs, own_dist, pass_dist, phase, required, deficit, safe_radius):
        candidates = []
        gr, gc = self.general_pos

        for r, c, d, nr, nc, src_army in self._iter_moves(obs):
            dest_owner = obs.owner_grid[nr][nc]
            dest_type = obs.type_grid[nr][nc]
            dest_army = obs.army_grid[nr][nc]

            is_neutral = _is_visible_neutral(dest_owner, dest_type)
            is_opp = dest_owner == 2
            if not is_neutral and not is_opp:
                continue
            if src_army <= dest_army + 1:
                continue

            if (r, c) == self.general_pos and (src_army - 1) < required:
                continue

            src_in_safe = own_dist[r][c] >= 0 and own_dist[r][c] <= safe_radius
            if deficit > 0 and src_in_safe:
                continue

            land_val = 1 if is_neutral else 3
            score = 30 * land_val + 5 * src_army - 4 * dest_army

            if is_opp and dest_type == 4 and obs.turn < DEATHTOUCH_TURN:
                score += 1000
            if is_opp and dest_type == 3:
                score += 200
            if own_dist[nr][nc] >= 0 and own_dist[nr][nc] <= 6:
                score += 40
            if src_in_safe:
                score -= 500
            if (r, c) == self.general_pos:
                score -= 1000

            remaining_on_gen = src_army - 1 if (r, c) == self.general_pos else required
            post_deficit = max(0, required - (obs.army_grid[gr][gc] if (r, c) != self.general_pos
                                               else src_army - 1))
            if post_deficit > 0 and src_in_safe:
                continue

            candidates.append((score, src_army, own_dist[nr][nc] if own_dist[nr][nc] >= 0 else 999,
                               (0, r, c, d, 0)))

        return self._pick_best(candidates)

    def _score_consolidation(self, obs, own_dist, safe_radius):
        candidates = []
        gr, gc = self.general_pos

        for r, c, d, nr, nc, src_army in self._iter_moves(obs):
            if (r, c) == self.general_pos:
                continue
            if obs.owner_grid[nr][nc] != 1:
                continue
            here = own_dist[r][c]
            there = own_dist[nr][nc]
            if here < 0 or there < 0 or here <= safe_radius:
                continue
            if there >= here:
                continue

            reduction = here - there
            score = 1000 * reduction + src_army
            candidates.append((score, src_army, there, (0, r, c, d, 0)))

        return self._pick_best(candidates)
