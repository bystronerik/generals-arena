"""
castle_builder — castle-aware economy bot.

Expands greedily like `expander_python`. Once land is established, it stops
spending the *general's own* army on captures and lets it rest so army can
accumulate there (every other turn, plus the 50-turn land bonus) — every
other owned cell keeps expanding as normal. Once the general is holding
enough army to afford a nearby castle, it relocates that whole stack one
hop and builds (`pass=2`) on the next turn.

This reserve step exists because pure opportunistic building (build
whenever some already-owned cell happens to have spare army) almost never
fires under greedy expansion: any cell large enough to afford a castle is
also the cell the expander immediately spends on the next capture, so army
never idles long enough to reach the ~50 a castle costs. Deliberately
resting the general is the cheapest way to grow one stack.

Build cost (see RULES.md section 03 / docs/competition/build-castles.md):
    cost = 35 + sum_over_own_structures(max(0, 14 - 2 * manhattan_dist))

Only the general and this bot's own castles count toward its own price.

See docs/bots/castle-builder.md and
docs/research/experiments/002-castle-builder-early-investment.md.
"""

# A no-op action — used when no valid move exists.
PASS = (1, 0, 0, 0, 0)

# (dr, dc) offsets for direction codes 0..3
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BASE_BUILD_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2

# Only consider building once we have some land, only early/mid game (a
# castle built in the last stretch of a 1200-turn game barely produces),
# and stop after a couple of castles so we don't keep resting the general.
MIN_TURN_TO_BUILD = 20
MAX_TURN_TO_BUILD = 900
MIN_LAND_TO_BUILD = 8
MAX_OWN_CASTLES = 2
BUILD_SURPLUS_MARGIN = 15
BUILD_COOLDOWN_TURNS = 40


def _is_passable(t):
    return t != 2 and t != 5


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Agent:
    """Greedy expansion, plus a rested general funding one or two castles."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W
        self.last_build_turn = -BUILD_COOLDOWN_TURNS
        self.general_pos = None

    def act(self, obs):
        self._locate_general(obs)
        structures, own_castle_count = self._own_structures(obs)
        resting = self._should_rest_general(obs, own_castle_count)

        build_move = self._maybe_build(obs, structures, own_castle_count)
        if build_move is not None:
            self.last_build_turn = obs.turn
            return build_move

        if resting:
            relocate_move = self._maybe_relocate_general(obs, structures)
            if relocate_move is not None:
                return relocate_move

        exclude = self.general_pos if resting else None
        return self._expand(obs, exclude_cell=exclude)

    def _locate_general(self, obs):
        if self.general_pos is not None:
            return
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] == 1 and obs.type_grid[r][c] == 4:
                    self.general_pos = (r, c)
                    return

    def _should_rest_general(self, obs, own_castle_count):
        if self.general_pos is None:
            return False
        if own_castle_count >= MAX_OWN_CASTLES:
            return False
        if not (MIN_TURN_TO_BUILD <= obs.turn <= MAX_TURN_TO_BUILD):
            return False
        if obs.my_land < MIN_LAND_TO_BUILD:
            return False
        return True

    def _own_structures(self, obs):
        """Return (list of (r, c) for general + own castles, own castle count)."""
        structures = []
        castles = 0
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                t = obs.type_grid[r][c]
                if t == 4:
                    structures.append((r, c))
                elif t == 3:
                    structures.append((r, c))
                    castles += 1
        return structures, castles

    def _build_cost(self, cell, structures):
        cost = BASE_BUILD_COST
        for s in structures:
            d = _manhattan(cell, s)
            cost += max(0, PROXIMITY_PENALTY - PROXIMITY_DECAY * d)
        return cost

    def _maybe_build(self, obs, structures, own_castle_count):
        if own_castle_count >= MAX_OWN_CASTLES:
            return None
        if not (MIN_TURN_TO_BUILD <= obs.turn <= MAX_TURN_TO_BUILD):
            return None
        if obs.turn - self.last_build_turn < BUILD_COOLDOWN_TURNS:
            return None

        best_cell = None
        best_surplus = float("-inf")
        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if obs.type_grid[r][c] != 1:
                    continue  # only plain cells are buildable
                army = obs.army_grid[r][c]
                cost = self._build_cost((r, c), structures)
                surplus = army - cost
                if surplus > best_surplus:
                    best_surplus = surplus
                    best_cell = (r, c)

        if best_cell is not None and best_surplus >= BUILD_SURPLUS_MARGIN:
            r, c = best_cell
            return (2, r, c, 0, 0)
        return None

    def _maybe_relocate_general(self, obs, structures):
        """Once the rested general can afford a nearby castle, move its whole
        stack one hop to the cheapest neighbor so `_maybe_build` can act on
        it next turn."""
        gr, gc = self.general_pos
        army = obs.army_grid[gr][gc]
        if army <= 1:
            return None

        best_dir = None
        best_cost = None
        for d, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = gr + dr, gc + dc
            if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                continue
            if not _is_passable(obs.type_grid[nr][nc]):
                continue
            if obs.owner_grid[nr][nc] == 2:
                continue  # do not pick a fight to stage a build
            if obs.type_grid[nr][nc] in (3, 4):
                continue  # not buildable
            cost = self._build_cost((nr, nc), structures)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_dir = d

        if best_dir is None:
            return None

        transferred = army - 1  # one always stays behind on the source
        if transferred < best_cost + BUILD_SURPLUS_MARGIN:
            return None  # still resting; not enough to afford it yet
        return (0, gr, gc, best_dir, 0)

    def _expand(self, obs, exclude_cell=None):
        best_score = -1.0
        best_move = None
        first_valid = None

        for r in range(obs.H):
            for c in range(obs.W):
                if obs.owner_grid[r][c] != 1:
                    continue
                if exclude_cell is not None and (r, c) == exclude_cell:
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
