"""
classic_duel — remote-only champion for classic generals.io.

Expansion logistics, neutral city capture, general reserve, and strict general
kill conversion. Never emits build (pass=2) or deathtouch logic.

See docs/research/strategies/classic_duel.md.
"""
from collections import deque

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

RESERVE_BASE = 8
RESERVE_PER_LAND = 0.04
CITY_MIN_OWNED_LAND = 12
CITY_MAX_GARRISON = 35
CITY_CAPTURE_SCORE = 800
CITY_LAND_BONUS = 4
CONVEY_MIN_ARMY = 3
FRONTIER_NEIGHBOR_WEIGHT = 1.5
OPPONENT_CAPTURE_MULT = 3.0
ENEMY_GENERAL_SCORE = 10000
GENERAL_ATTACK_MIN_MARGIN = 1
SCOUT_UNSIGHTED_LAND = 80
SCOUT_FOG_CAPTURE_MULT = 2.5
SCOUT_MARCH_MIN_ARMY = 3
SCOUT_RESERVE_FACTOR = 0.75


def _is_passable(t):
    return t != 2 and t != 5


def _is_unowned(owner, cell_type):
    if owner == 2:
        return True
    return owner == 0 and cell_type not in (0, 5)


def _required_reserve(my_land):
    return RESERVE_BASE + int(my_land * RESERVE_PER_LAND)


def _can_use_as_source(obs, r, c, general_pos, reserve):
    if general_pos is None or (r, c) != general_pos:
        return True
    army = obs.army_grid[r][c]
    return army - 1 >= reserve


class Agent:
    """Classic generals.io champion: convey, cities, reserve, kill conversion."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.general_pos = None
        self.enemy_general_pos = None
        self.first_city_capture_turn = None
        self.first_sighting_turn = None
        self.ever_seen = [[False] * W for _ in range(H)]

    def act(self, obs):
        self._locate_own_general(obs)
        self._update_enemy_general(obs)
        self._update_ever_seen(obs)
        scouting = self._in_scout_mode(obs)
        reserve = _required_reserve(obs.my_land)
        if scouting:
            reserve = max(RESERVE_BASE, int(reserve * SCOUT_RESERVE_FACTOR))

        kill = self._enemy_general_capture(obs, reserve)
        if kill is not None:
            return kill

        if not scouting:
            city = self._neutral_city_capture(obs, reserve)
            if city is not None:
                if self.first_city_capture_turn is None:
                    self.first_city_capture_turn = obs.turn
                return city

        frontier = self._frontier_cells(obs)
        fog_mult = SCOUT_FOG_CAPTURE_MULT if scouting else 1.0
        capture = self._frontier_capture(obs, frontier, reserve, fog_mult=fog_mult)
        if capture is not None:
            return capture

        if scouting:
            march = self._fog_march(obs)
            if march is not None:
                return march

        convey = self._interior_convey(obs, frontier)
        if convey is not None:
            return convey

        gather = self._frontier_gathering(obs, frontier)
        if gather is not None:
            return gather

        fallback = self._any_valid_move(obs, reserve)
        if fallback is not None:
            return fallback
        return PASS

    def telemetry_extras(self):
        extras = {}
        if self.first_sighting_turn is not None:
            extras["enemy_general_sighted"] = 1
            extras["first_sighting_turn"] = self.first_sighting_turn
        else:
            extras["enemy_general_sighted"] = 0
        if self.first_city_capture_turn is not None:
            extras["first_city_capture_turn"] = self.first_city_capture_turn
        return extras

    def _locate_own_general(self, obs):
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
                if obs.type_grid[r][c] == 4 and obs.owner_grid[r][c] == 2:
                    if self.enemy_general_pos != (r, c):
                        self.enemy_general_pos = (r, c)
                    if self.first_sighting_turn is None:
                        self.first_sighting_turn = obs.turn
                    return

    def _update_ever_seen(self, obs):
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.type_grid[r][c] != 0:
                    self.ever_seen[r][c] = True

    def _in_scout_mode(self, obs):
        return (
            self.enemy_general_pos is None
            and obs.my_land >= SCOUT_UNSIGHTED_LAND
        )

    def _enemy_general_capture(self, obs, reserve):
        if self.enemy_general_pos is None:
            return None
        er, ec = self.enemy_general_pos
        if obs.owner_grid[er][ec] != 2 or obs.type_grid[er][ec] != 4:
            return None

        best = None
        best_score = -1.0
        dest_army = obs.army_grid[er][ec]
        margin = GENERAL_ATTACK_MIN_MARGIN

        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if not _can_use_as_source(obs, r, c, self.general_pos, reserve):
                    continue
                src_army = obs.army_grid[r][c]
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if (nr, nc) != (er, ec):
                        continue
                    if src_army <= dest_army + margin:
                        continue
                    score = float(ENEMY_GENERAL_SCORE) + float(src_army)
                    if score > best_score:
                        best_score = score
                        best = (0, r, c, d, 0)
        return best

    def _neutral_city_capture(self, obs, reserve):
        if obs.my_land < CITY_MIN_OWNED_LAND:
            return None

        best = None
        best_score = -1.0
        margin = GENERAL_ATTACK_MIN_MARGIN

        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if not _can_use_as_source(obs, r, c, self.general_pos, reserve):
                    continue
                src_army = obs.army_grid[r][c]
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                        continue
                    if obs.owner_grid[nr][nc] != 0 or obs.type_grid[nr][nc] != 3:
                        continue
                    garrison = obs.army_grid[nr][nc]
                    if garrison > CITY_MAX_GARRISON:
                        continue
                    if src_army <= garrison + margin:
                        continue
                    score = float(CITY_CAPTURE_SCORE + obs.my_land * CITY_LAND_BONUS - garrison)
                    if score > best_score:
                        best_score = score
                        best = (0, r, c, d, 0)
        return best

    def _frontier_capture(self, obs, frontier, reserve, fog_mult=1.0):
        best_score = -1.0
        best_capture = None
        margin = GENERAL_ATTACK_MIN_MARGIN

        for r, c in frontier:
            if not _can_use_as_source(obs, r, c, self.general_pos, reserve):
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
                if obs.owner_grid[nr][nc] == 1:
                    continue
                if obs.type_grid[nr][nc] == 3 and obs.owner_grid[nr][nc] == 0:
                    continue
                dest_army = obs.army_grid[nr][nc]
                if src_army <= dest_army + margin:
                    continue
                score = float(src_army) * 10.0
                score += self._frontier_neighbor_armies(obs, r, c) * FRONTIER_NEIGHBOR_WEIGHT
                if obs.owner_grid[nr][nc] == 2:
                    score *= OPPONENT_CAPTURE_MULT
                is_fog = obs.type_grid[nr][nc] == 0 or not self.ever_seen[nr][nc]
                if is_fog and fog_mult > 1.0:
                    score *= fog_mult
                if score > best_score:
                    best_score = score
                    best_capture = (0, r, c, d, 0)
        return best_capture

    def _fog_march(self, obs):
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
                if src_army < SCOUT_MARCH_MIN_ARMY:
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

    def _interior_convey(self, obs, frontier):
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
        return best_convey

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

    def _any_valid_move(self, obs, reserve):
        H, W = obs.H, obs.W
        for r in range(H):
            for c in range(W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if not _can_use_as_source(obs, r, c, self.general_pos, reserve):
                    continue
                if obs.army_grid[r][c] <= 1:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and _is_passable(obs.type_grid[nr][nc]):
                        return (0, r, c, d, 0)
        return None

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
