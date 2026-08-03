"""
late_rush — early expansion, rally accumulation, turn-700 committed rush.

Expands early, selects a rally cell and main stack, then commits toward
the enemy between turns 650–800. After turn 800, deathtouch contact on
the known enemy general is highest priority.

See docs/research/strategies/late_rush.md.
"""
from collections import deque

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

DEATHTOUCH_TURN = 800
RALLY_START = 400
ACCUM_START = 550
COMMIT_EARLY = 625
COMMIT_MID = 650
COMMIT_LATE = 675
RALLY_REFRESH = 20
STALL_LIMIT = 8

RESERVE_EARLY = 10
RESERVE_RALLY = 12
RESERVE_ACCUM = 12
MAIN_STACK_MIN = 10
MAIN_STACK_DELAY = 25
EMERGENCY_THREAT_MOVES = 3


def _is_passable(t):
    return t != 2 and t != 5


def _is_visible_neutral(owner, cell_type):
    return owner == 0 and cell_type not in (0, 5)


class Agent:
    """Expand, rally, accumulate, then rush the enemy general."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.own_general_pos = None
        self.enemy_general_pos = None
        self.enemy_memory = {}
        self.rally_pos = None
        self.main_stack_pos = None
        self.commit_started = False
        self.commit_turn = None
        self.target_pos = None
        self.target_kind = "unknown"
        self.stalled_turns = 0
        self.last_rally_refresh = -RALLY_REFRESH
        self.commit_delay_turns = 0

    def act(self, obs):
        self._locate_general(obs)
        if self.own_general_pos is None:
            return PASS

        self._update_enemy_memory(obs)
        self._validate_state(obs)

        if obs.turn >= DEATHTOUCH_TURN and self.enemy_general_pos is not None:
            contact = self._contact_moves(obs)
            if contact is not None:
                return contact

        emergency = self._emergency_defense(obs)
        if emergency is not None:
            return emergency

        if not self.commit_started:
            if obs.turn >= RALLY_START:
                self._maybe_refresh_rally(obs)
            if obs.turn >= ACCUM_START:
                self._select_main_stack(obs)
            if self._should_commit(obs):
                self.commit_started = True
                self.commit_turn = obs.turn
                self.stalled_turns = 0

        if self.commit_started:
            move = self._rush_turn(obs)
            if move is not None:
                return move

        if RALLY_START <= obs.turn < COMMIT_LATE:
            move = self._consolidation_or_expansion(obs)
            if move is not None:
                return move

        move = self._early_expansion(obs)
        if move is not None:
            return move

        move = self._idle_toward_rally_or_frontier(obs)
        if move is not None:
            return move

        return PASS

    def _locate_general(self, obs):
        if self.own_general_pos is not None:
            return
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 1 and obs.type_grid[r][c] == 4:
                    self.own_general_pos = (r, c)
                    return

    def _update_enemy_memory(self, obs):
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 2:
                    self.enemy_memory[(r, c)] = (
                        obs.turn, obs.army_grid[r][c], obs.type_grid[r][c]
                    )
                    if obs.type_grid[r][c] == 4:
                        self.enemy_general_pos = (r, c)

    def _validate_state(self, obs):
        if self.main_stack_pos is not None:
            r, c = self.main_stack_pos
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] < 2:
                self.main_stack_pos = None
        if self.rally_pos is not None:
            rr, rc = self.rally_pos
            if obs.owner_grid[rr][rc] != 1:
                self.rally_pos = None

    def _owned_bfs(self, obs, source):
        H, W = obs.H, obs.W
        dist = [[-1] * W for _ in range(H)]
        if source is None:
            return dist
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
        if source is None:
            return dist
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

    def _frontier_cells(self, obs):
        cells = []
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                for dr, dc in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                        continue
                    no = obs.owner_grid[nr][nc]
                    nt = obs.type_grid[nr][nc]
                    if no == 2 or no == 0:
                        cells.append((r, c))
                        break
        return cells

    def _reserve_for_turn(self, turn):
        if turn < RALLY_START:
            return RESERVE_EARLY
        if turn < ACCUM_START:
            return RESERVE_RALLY
        return RESERVE_ACCUM

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

    def _pick_best(self, candidates, prefer_main=False):
        if not candidates:
            return None
        candidates.sort(
            key=lambda x: (
                -x[0],
                -int(prefer_main and x[4]),
                -x[1],
                -x[2],
                x[3][0],
                x[3][1],
                x[3][2],
            )
        )
        return candidates[0][3]

    def _contact_moves(self, obs):
        if self.enemy_general_pos is None:
            return None
        tr, tc = self.enemy_general_pos
        candidates = []
        for d, (dr, dc) in enumerate(DIRECTIONS):
            r, c = tr - dr, tc - dc
            if not (0 <= r < obs.H and 0 <= c < obs.W):
                continue
            if obs.owner_grid[r][c] == 1 and obs.army_grid[r][c] >= 2:
                candidates.append((1000000, obs.army_grid[r][c], 0, (0, r, c, d, 0), True))
        return self._pick_best(candidates)

    def _emergency_defense(self, obs):
        if self.own_general_pos is None:
            return None
        pass_dist = self._passable_bfs(obs, self.own_general_pos)
        candidates = []
        for r, c, d, nr, nc, src_army in self._iter_moves(obs):
            if obs.owner_grid[nr][nc] != 2:
                continue
            if src_army <= obs.army_grid[nr][nc] + 1:
                continue
            threat_dist = pass_dist[nr][nc]
            if threat_dist < 0 or threat_dist > EMERGENCY_THREAT_MOVES:
                continue
            score = 80000 - 1000 * threat_dist + src_army
            candidates.append((score, src_army, 0, (0, r, c, d, 0), False))
        return self._pick_best(candidates)

    def _should_commit(self, obs):
        if obs.turn >= COMMIT_LATE:
            return True
        if obs.turn >= COMMIT_EARLY and self.enemy_general_pos is not None:
            return True
        if obs.turn >= COMMIT_MID and self._has_stable_target(obs):
            return True
        return False

    def _has_stable_target(self, obs):
        for cell in self.enemy_memory:
            return True
        return len(self._frontier_cells(obs)) > 0

    def _maybe_refresh_rally(self, obs):
        need = (
            self.rally_pos is None
            or obs.turn - self.last_rally_refresh >= RALLY_REFRESH
        )
        if not need:
            return
        self.last_rally_refresh = obs.turn
        gen_dist = self._owned_bfs(obs, self.own_general_pos)
        frontier = self._frontier_cells(obs)
        if not frontier:
            return

        best = None
        best_score = float("-inf")
        for r, c in frontier:
            if (r, c) == self.own_general_pos:
                continue
            gd = gen_dist[r][c]
            if gd < 0 or gd < 4:
                continue

            enemy_facing = self._enemy_facing_score(obs, r, c)
            local_support = self._local_support(obs, gen_dist, r, c)
            general_risk = max(0, 4 - gd) * 20
            score = 20 * enemy_facing + 4 * local_support - 3 * general_risk

            if self.enemy_general_pos is not None:
                route = self._passable_bfs(obs, self.enemy_general_pos)
                if route[r][c] >= 0:
                    score += 200

            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < obs.H and 0 <= nc < obs.W and obs.owner_grid[nr][nc] == 2:
                    score += 100

            exits = sum(
                1 for dr, dc in DIRECTIONS
                if 0 <= r + dr < obs.H and 0 <= c + dc < obs.W
                and _is_passable(obs.type_grid[r + dr][c + dc])
            )
            if exits <= 1:
                score -= 150

            if score > best_score or (score == best_score and best is not None
                                      and local_support > self._local_support(obs, gen_dist, best[0], best[1])):
                best_score = score
                best = (r, c)

        if best is not None:
            self.rally_pos = best

    def _enemy_facing_score(self, obs, r, c):
        best_dist = 20
        for (er, ec), (turn, army, _) in self.enemy_memory.items():
            d = abs(er - r) + abs(ec - c)
            best_dist = min(best_dist, d)
        if self.enemy_general_pos is not None:
            gr, gc = self.enemy_general_pos
            best_dist = min(best_dist, abs(gr - r) + abs(gc - c))
        if not self.enemy_memory and self.enemy_general_pos is None:
            gr, gc = self.own_general_pos
            best_dist = abs(r - gr) + abs(c - gc)
        return 20 - min(20, best_dist)

    def _local_support(self, obs, gen_dist, r, c):
        total = 0
        for rr in range(obs.H):
            for cc in range(obs.W):
                if obs.owner_grid[rr][cc] != 1:
                    continue
                d = gen_dist[rr][cc]
                if d < 0 or d > 4:
                    continue
                army = obs.army_grid[rr][cc]
                if army >= 2:
                    total += army
        return min(50, total)

    def _select_main_stack(self, obs):
        gen_dist = self._owned_bfs(obs, self.own_general_pos)
        rally_dist = self._owned_bfs(obs, self.rally_pos)
        target = self._select_target(obs)
        target_dist = self._passable_bfs(obs, target)

        candidates = []
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if (r, c) == self.own_general_pos:
                    continue
                army = obs.army_grid[r][c]
                if army < 2:
                    continue
                gd = gen_dist[r][c]
                if gd >= 0 and gd < 2:
                    continue

                min_army = MAIN_STACK_MIN if obs.turn < COMMIT_LATE else 2
                if army < min_army:
                    continue

                td = target_dist[r][c] if target_dist[r][c] >= 0 else 999
                rd = rally_dist[r][c] if rally_dist[r][c] >= 0 else 999
                score = 10 * army - 5 * rd - 2 * td
                if self.main_stack_pos == (r, c):
                    score += 300
                candidates.append((score, army, td, (r, c)))

        if not candidates:
            self.commit_delay_turns += 1
            if self.commit_delay_turns >= MAIN_STACK_DELAY:
                best_army = -1
                best_cell = None
                for r in range(obs.H):
                    for c in range(obs.W):
                        if obs.owner_grid[r][c] != 1:
                            continue
                        if (r, c) == self.own_general_pos:
                            continue
                        if obs.army_grid[r][c] > best_army:
                            best_army = obs.army_grid[r][c]
                            best_cell = (r, c)
                if best_cell and best_army >= 2:
                    self.main_stack_pos = best_cell
            return

        candidates.sort(key=lambda x: (-x[0], -x[1], x[3][0], x[3][1]))
        self.main_stack_pos = candidates[0][3]

    def _select_target(self, obs):
        if self.enemy_general_pos is not None:
            self.target_pos = self.enemy_general_pos
            self.target_kind = "enemy_general"
            return self.target_pos

        best = None
        best_score = float("-inf")
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 2:
                    continue
                t = obs.type_grid[r][c]
                type_val = 1000 if t == 4 else (20 if t == 3 else 5)
                route = self._passable_bfs(obs, (r, c))
                from_main = 999
                if self.main_stack_pos:
                    md = route[self.main_stack_pos[0]][self.main_stack_pos[1]]
                    from_main = md if md >= 0 else 999
                score = 8 * type_val + 4 * obs.army_grid[r][c] + 20 - 5 * from_main
                if score > best_score:
                    best_score = score
                    best = (r, c)
                    self.target_kind = "visible_enemy"

        if best is not None:
            self.target_pos = best
            return best

        freshest = None
        freshest_turn = -1
        for cell, (turn, army, _) in self.enemy_memory.items():
            if turn > freshest_turn:
                freshest_turn = turn
                freshest = cell
        if freshest is not None:
            self.target_pos = freshest
            self.target_kind = "remembered_enemy"
            return freshest

        probe = self._frontier_probe(obs)
        self.target_pos = probe
        self.target_kind = "frontier_probe"
        return probe

    def _frontier_probe(self, obs):
        gen_dist = self._passable_bfs(obs, self.own_general_pos)
        frontier = self._frontier_cells(obs)
        best = None
        best_score = float("-inf")
        for r, c in frontier:
            gd = gen_dist[r][c]
            if gd >= 0 and gd < 3:
                continue
            exits = sum(
                1 for dr, dc in DIRECTIONS
                if 0 <= r + dr < obs.H and 0 <= c + dc < obs.W
                and _is_passable(obs.type_grid[r + dr][c + dc])
            )
            score = (gd if gd >= 0 else 0) + 2 * exits
            if score > best_score:
                best_score = score
                best = (r, c)
        if best is None and frontier:
            best = frontier[0]
        if best is None:
            best = self.own_general_pos
        return best

    def _rush_turn(self, obs):
        target = self._select_target(obs)
        target_dist = self._passable_bfs(obs, target)
        prev_dist = None
        if self.main_stack_pos:
            prev_dist = target_dist[self.main_stack_pos[0]][self.main_stack_pos[1]]

        candidates = []
        deathtouch = obs.turn >= DEATHTOUCH_TURN

        for r, c, d, nr, nc, src_army in self._iter_moves(obs):
            dest_owner = obs.owner_grid[nr][nc]
            dest_type = obs.type_grid[nr][nc]
            dest_army = obs.army_grid[nr][nc]
            is_main = self.main_stack_pos == (r, c)

            if dest_owner == 2:
                if not deathtouch and src_army <= dest_army + 1:
                    continue
                if deathtouch and dest_type == 4:
                    pass
                elif src_army <= dest_army + 1:
                    continue

            here = target_dist[r][c] if target_dist[r][c] >= 0 else 999
            there = target_dist[nr][nc] if target_dist[nr][nc] >= 0 else 999
            reduction = here - there if there < here else 0

            score = 10000 * reduction + 20 * src_army
            if deathtouch and dest_owner == 2 and dest_type == 4:
                score += 1000000
            elif dest_owner == 2 and dest_type == 4 and obs.turn < DEATHTOUCH_TURN:
                score += 500000
            if is_main:
                score += 50000
            if dest_owner == 2 and reduction > 0:
                score += 10000
            if dest_owner == 2 and dest_type == 3:
                score += 5000
            if (r, c) == self.own_general_pos:
                score -= 30000
            if there > here:
                score -= 20000

            candidates.append((score, reduction, src_army, (0, r, c, d, 0), is_main))

        move = self._pick_best(candidates, prefer_main=True)
        if move is not None:
            if self.main_stack_pos:
                md = target_dist[self.main_stack_pos[0]][self.main_stack_pos[1]]
                if md >= 0 and prev_dist is not None and prev_dist >= 0 and md >= prev_dist:
                    self.stalled_turns += 1
                else:
                    self.stalled_turns = 0
            return move

        self.stalled_turns += 1
        if self.stalled_turns >= STALL_LIMIT:
            self.target_pos = self._frontier_probe(obs)
            self.target_kind = "frontier_probe"
            self.stalled_turns = 0
        return None

    def _consolidation_or_expansion(self, obs):
        reserve = self._reserve_for_turn(obs.turn)
        rally_dist = self._owned_bfs(obs, self.rally_pos)
        main_dist = self._owned_bfs(obs, self.main_stack_pos)

        candidates = []
        for r, c, d, nr, nc, src_army in self._iter_moves(obs):
            dest_owner = obs.owner_grid[nr][nc]
            dest_type = obs.type_grid[nr][nc]
            dest_army = obs.army_grid[nr][nc]

            if dest_owner == 1:
                ref = rally_dist if self.rally_pos else main_dist
                here = ref[r][c]
                there = ref[nr][nc]
                if here >= 0 and there >= 0 and there < here:
                    reduction = here - there
                    score = 2000 * reduction + 20 * (src_army - 1)
                    if self.rally_pos == (nr, nc):
                        score += 1000
                    if self.main_stack_pos == (nr, nc):
                        score += 500
                    if (r, c) == self.own_general_pos and (src_army - 1) < reserve:
                        score -= 5000
                    candidates.append((score, reduction, src_army, (0, r, c, d, 0), False))
                continue

            is_neutral = _is_visible_neutral(dest_owner, dest_type)
            is_opp = dest_owner == 2
            if not is_neutral and not is_opp:
                continue
            if src_army <= dest_army + 1:
                continue
            if (r, c) == self.main_stack_pos and obs.turn >= ACCUM_START:
                continue
            if (r, c) == self.own_general_pos and (src_army - 1) < reserve:
                continue

            cap_val = 1 if is_neutral else 4
            if is_opp and dest_type == 3:
                cap_val = 20
            if is_opp and dest_type == 4:
                cap_val = 10000
            score = 50 * cap_val + 8 * src_army - 6 * dest_army
            candidates.append((score, 0, src_army, (0, r, c, d, 0), False))

        prefer_consol = obs.turn >= ACCUM_START
        if prefer_consol:
            consol = [c for c in candidates if c[1] > 0]
            if consol:
                return self._pick_best(consol)
        return self._pick_best(candidates)

    def _early_expansion(self, obs):
        reserve = self._reserve_for_turn(obs.turn)
        candidates = []
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
            if (r, c) == self.own_general_pos:
                if (src_army - 1) < reserve:
                    continue

            cap_val = 1 if is_neutral else 4
            if is_opp and dest_type == 3:
                cap_val = 20
            if is_opp and dest_type == 4 and src_army > dest_army + 1:
                cap_val = 10000

            score = 50 * cap_val + 8 * src_army - 6 * dest_army
            if obs.turn < RALLY_START and self.own_general_pos:
                gr, gc = self.own_general_pos
                score += 40 * (abs(r - gr) + abs(c - gc))

            if (r, c) == self.own_general_pos:
                score -= 1000

            candidates.append((score, 0, src_army, (0, r, c, d, 0), False))

        return self._pick_best(candidates)

    def _idle_toward_rally_or_frontier(self, obs):
        dest = self.rally_pos or self._frontier_probe(obs)
        dist = self._owned_bfs(obs, dest)
        candidates = []
        for r, c, d, nr, nc, src_army in self._iter_moves(obs):
            if obs.owner_grid[nr][nc] != 1:
                continue
            if (r, c) == self.own_general_pos:
                continue
            here = dist[r][c]
            there = dist[nr][nc]
            if here < 0 or there < 0 or there >= here:
                continue
            reduction = here - there
            candidates.append((2000 * reduction + src_army, reduction, src_army,
                               (0, r, c, d, 0), False))
        return self._pick_best(candidates)
