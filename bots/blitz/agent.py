"""Blitz — general rush (spec: docs/research/strategies/blitz.md).

Grid-native rewrite of the generals-bot Blitz strategy. Priority ladder each
turn:

1. **Finish** — take the enemy general if any cell of ours can, right now
   (including the turn-800 deathtouch, where any two-army stack suffices).
2. **Defend** — only when our general is genuinely losable.
3. **Open** — chain expansion until ``OPENING_END``.
4. **Assault** — advance or feed the strike stack (rebuild → rally → push).
5. **Expand** — anything left over; a whiffed rush must not also be idle.

All cross-turn state lives in :class:`BlitzMemory` so a composite bot
(proteus) can own several strategies side by side and share one
:class:`~_common.oppmodel.OpponentModel` between them.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from _common.oppmodel import OpponentModel
from _common.strategy_common import (
    DIRECTIONS,
    PASS,
    StrategyContext,
    direction_from_to,
    enemy_general_army,
    is_passable,
    locate_own_general,
)
from _common.tactics import (
    UNREACHABLE,
    expansion_step,
    fog_reveal,
    gather_step,
    mirror_tile,
    multi_bfs,
    visible_enemy_tiles,
)

Cell = tuple[int, int]

INF = float("inf")

# Blitz empties its home by design: the general is never held back by a
# reserve schedule (defence is handled by the explicit threat model instead).
_NEVER = 1 << 30
_STRATEGY = StrategyContext(reserve_opening_end=_NEVER)

DEATHTOUCH_TURN = 800
"""From this turn any adjacent stack of 2+ takes a general (RULES.md §07)."""

CHASE_DEFEND_FROM = 780
"""Shortly before deathtouch, kill any enemy cell that reaches our general."""

# Phase names, exposed for traces/tests.
PHASE_OPENING = "opening"
PHASE_REBUILD = "rebuild"
PHASE_RALLY = "rally"
PHASE_ASSAULT = "assault"
PHASE_DEFEND = "defend"
PHASE_FINISH = "finish"
PHASE_IDLE = "idle"


@dataclass(frozen=True)
class BlitzConfig:
    """All Blitz knobs. Values carried over from the generals-bot tuning;
    the turn cadence (production every other turn, land bonus every 50)
    matches, so they transfer as hypotheses to re-measure in the arena."""

    # ---------------------------------------------------------- opening
    opening_end: int = 50
    """First turn of the assault phase — the first land-bonus turn."""

    chain_detour: int = 3
    """How far a stalled chain head may walk over owned land to find fresh
    neutral cells before the chain is abandoned."""

    # ------------------------------------------------------- wave cycle
    rally_ticks: int = 14
    """How long a wave concentrates army before it launches regardless."""

    rebuild_ticks: int = 50
    """Cap on the expansion stretch between a dead wave and the next rally."""

    collection_walk: bool = True
    """Rally by walking the stack forward through our own surplus instead of
    ferrying stacks back to it — travelling and gathering become one move."""

    spent_army: int = 5
    """A strike stack this small has done its damage; the wave is over."""

    # ---------------------------------------------------------- assault
    min_strike_floor: int = 18
    """Absolute floor on a worthwhile strike stack."""

    strike_ratio: float = 0.7
    """Stack target as a fraction of the opponent's *mobile* army
    (``army - land``: the part not pinned as one-per-cell)."""

    travel_margin: float = 1.0
    """Army budgeted per hop of the approach."""

    feed_budget: int = 20
    """Max turns a blocked strike spends reinforcing before giving up."""

    restack_ratio: float = 1.5
    """Switch the strike stack to another cell once that cell holds this
    multiple of the current stack (i.e. the old strike died)."""

    # ---------------------------------------------------------- targeting
    retarget_interval: int = 20
    """Re-estimate a fogged enemy general at most this often."""

    contact_radius: int = 6
    """How far behind their visible front line to look for their general."""

    # ---------------------------------------------------------- defence
    defense_dist: int = 12
    """Only enemy cells within this BFS distance of our general count as a
    home threat — Blitz would rather race than turtle."""

    defense_margin: int = 2
    """Extra army we want at home on top of the incoming threat."""

    # ---------------------------------------------------------- pathing
    own_tile_bonus: float = 0.05
    """Per-army discount for routing the strike through our own cells."""

    own_tile_bonus_cap: int = 8
    """Army count past which extra army stops making a cell cheaper."""

    neutral_cost: float = 1.15
    """Cost of crossing neutral/fog land (slightly worse than own land)."""

    enemy_cost_per_army: float = 0.03
    """Extra path cost per defending army on an enemy cell."""


@dataclass
class BlitzMemory:
    """Everything Blitz remembers between turns."""

    opp: OpponentModel = field(default_factory=OpponentModel)
    belief: object = field(default_factory=_STRATEGY.BeliefState)

    my_general: Cell | None = None
    chain_head: Cell | None = None
    chain_visited: set[Cell] = field(default_factory=set)
    stack: Cell | None = None
    target: Cell | None = None
    target_turn: int = -1
    committed: bool = False
    rebuild_until: int = -1
    rally_until: int = -1
    feed_ticks: int = 0
    strikes: int = 0
    phase: str = PHASE_IDLE
    last_turn: int = -1

    def observe(self, obs) -> None:
        """Fold this turn's observation in. Idempotent within a turn."""
        if obs.turn == self.last_turn:
            return
        self.last_turn = obs.turn
        self.opp.update(obs)
        if self.my_general is None:
            self.my_general = locate_own_general(obs)
        self.belief.update(obs, self.my_general)


def _owned_cells(obs) -> list[Cell]:
    return [
        (r, c)
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.owner_grid[r][c] == 1
    ]


def _move(src: Cell, dst: Cell):
    d = direction_from_to(src[0], src[1], dst[0], dst[1])
    if d is None:
        return None
    return (0, src[0], src[1], d, 0)


# ---------------------------------------------------------------- targeting
def enemy_anchor(obs, mem: BlitzMemory) -> Cell | None:
    """Best cheap guess of "the enemy's side of the map".

    Known general > the enemy cell farthest from us > mirror of our general.
    """
    if mem.belief.enemy_general is not None:
        return mem.belief.enemy_general
    enemies = visible_enemy_tiles(obs)
    if enemies:
        if mem.my_general is None:
            return min(enemies)
        home = multi_bfs(obs, [mem.my_general])
        return max(
            enemies,
            key=lambda cell: (
                home[cell[0]][cell[1]] if home[cell[0]][cell[1]] < UNREACHABLE else -1,
                cell,
            ),
        )
    if mem.my_general is None:
        return None
    return mirror_tile(obs, *mem.my_general)


def bias_distances(obs, mem: BlitzMemory) -> list[list[int]] | None:
    """Distance grid from the enemy anchor — lower means "more enemy-ward"."""
    anchor = enemy_anchor(obs, mem)
    if anchor is None:
        return None
    return multi_bfs(obs, [anchor])


def unscouted_target(obs, mem: BlitzMemory) -> Cell | None:
    """Pre-contact probe: fog nearest the mirror-side anchor (not the fog
    farthest from us, which on a real map is a corner)."""
    anchor = enemy_anchor(obs, mem)
    if anchor is None:
        return None
    if obs.type_grid[anchor[0]][anchor[1]] == 0:
        return anchor
    fog = [
        (r, c)
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.type_grid[r][c] == 0
    ]
    if not fog:
        return anchor
    from_anchor = multi_bfs(obs, [anchor])
    return min(fog, key=lambda cell: (from_anchor[cell[0]][cell[1]], cell))


def contact_target(obs, mem: BlitzMemory, radius: int = 6) -> Cell | None:
    """Post-contact guess: the fog behind their front line, farthest from
    our territory (their visible cells are their newest land)."""
    enemies = visible_enemy_tiles(obs)
    if not enemies:
        return None
    from_enemy = multi_bfs(obs, enemies)
    fog = [
        (r, c)
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.type_grid[r][c] == 0 and from_enemy[r][c] <= radius
    ]
    if not fog:
        return None
    mine = _owned_cells(obs)
    if not mine:
        return min(fog)
    from_home = multi_bfs(obs, mine)
    return max(
        fog,
        key=lambda cell: (
            from_home[cell[0]][cell[1]]
            if from_home[cell[0]][cell[1]] < UNREACHABLE
            else -1,
            cell,
        ),
    )


def strike_target(obs, mem: BlitzMemory, config: BlitzConfig) -> Cell | None:
    """The cell the strike stack should walk at, with hysteresis.

    A seen general always wins. The guess is kept until it stops being fog
    (we scouted it), we are standing on it, or ``retarget_interval`` turns
    have passed — an oscillating target stalls the stack.
    """
    known = mem.belief.enemy_general
    if known is not None:
        if mem.target != known:
            mem.target_turn = obs.turn
        mem.target = known
        return known

    current = mem.target
    stale = (
        current is None
        or obs.type_grid[current[0]][current[1]] != 0
        or current == mem.stack
        or obs.turn - mem.target_turn >= config.retarget_interval
    )
    if not stale:
        return current

    estimate = contact_target(obs, mem, config.contact_radius)
    if estimate is None:
        estimate = unscouted_target(obs, mem)
    if estimate is None:
        estimate = enemy_anchor(obs, mem)
    mem.target = estimate
    mem.target_turn = obs.turn
    return estimate


# ------------------------------------------------------------------ defence
def home_threat(obs, mem: BlitzMemory, config: BlitzConfig) -> tuple[int, Cell | None]:
    """``(deficit, threatening_cell)``.

    Deliberately pessimistic about our side: **one** reinforcing stack, not
    the sum of every cell in range — we get one move per turn, so scattered
    army cannot converge the way a naive sum implies.
    """
    general = mem.my_general
    if general is None:
        return 0, None
    enemies = visible_enemy_tiles(obs)
    if not enemies:
        return 0, None
    dist = multi_bfs(obs, [general])
    near = [
        cell
        for cell in enemies
        if dist[cell[0]][cell[1]] <= config.defense_dist
        and obs.army_grid[cell[0]][cell[1]] > 1
    ]
    if not near:
        return 0, None
    threat_cell = max(near, key=lambda cell: (obs.army_grid[cell[0]][cell[1]], cell))
    threat = obs.army_grid[threat_cell[0]][threat_cell[1]] - 1
    arrival = dist[threat_cell[0]][threat_cell[1]]

    reinforcement = 0
    for cell in _owned_cells(obs):
        d = dist[cell[0]][cell[1]]
        if cell == general or d >= UNREACHABLE or d > arrival:
            continue
        reinforcement = max(reinforcement, obs.army_grid[cell[0]][cell[1]] - 1)
    # The general also produces one army every two turns while they walk.
    defense = obs.army_grid[general[0]][general[1]] + reinforcement + arrival // 2
    return threat - defense + config.defense_margin, threat_cell


def defense_move(obs, mem: BlitzMemory, config: BlitzConfig, threat_cell: Cell | None):
    """Kill the threat if we can reach it, else pull army home — preferring
    army that can arrive *before they do*."""
    general = mem.my_general
    if general is None:
        return None

    if threat_cell is not None:
        tr, tc = threat_cell
        defenders = obs.army_grid[tr][tc]
        best, best_army = None, None
        for dr, dc in DIRECTIONS:
            nr, nc = tr + dr, tc + dc
            if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                continue
            if obs.owner_grid[nr][nc] != 1:
                continue
            if obs.army_grid[nr][nc] - 1 <= defenders:
                continue
            if best_army is None or obs.army_grid[nr][nc] > best_army:
                best_army, best = obs.army_grid[nr][nc], (nr, nc)
        if best is not None:
            return _move(best, threat_cell)

        arrival = multi_bfs(obs, [general])[tr][tc]
        if arrival < UNREACHABLE:
            in_time = gather_step(
                obs, general, min_army=2, max_dist=arrival, exclude={general}
            )
            if in_time is not None:
                return in_time

    return gather_step(obs, general, min_army=2, exclude={general})


# ------------------------------------------------------------------ opening
def should_launch(obs, cell: Cell, config: BlitzConfig, deadline: int | None = None) -> bool:
    """Spend the general's stack on a new chain once the chain we can afford
    is long enough to use the production left before the deadline:
    ``2 * (army - 1) >= turns_left``."""
    army = obs.army_grid[cell[0]][cell[1]]
    if army < 2:
        return False
    if deadline is None:
        deadline = config.opening_end
    turns_left = deadline - obs.turn
    if turns_left <= 0:
        return True
    return 2 * (army - 1) >= turns_left


def chain_step(obs, cell: Cell, bias, visited: set[Cell], config: BlitzConfig) -> Cell | None:
    """Next cell for a chain standing on ``cell``: an adjacent neutral plain
    (enemy-ward first, then vision gain), or a short detour over own land."""
    r, c = cell
    army = obs.army_grid[r][c]
    if army < 2:
        return None
    reach = army - 1

    best, best_key = None, None
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W) or (nr, nc) in visited:
            continue
        if obs.owner_grid[nr][nc] != 0 or obs.type_grid[nr][nc] != 1:
            continue
        if obs.army_grid[nr][nc] >= reach:
            continue
        bias_key = -bias[nr][nc] if bias else 0
        key = (bias_key, fog_reveal(obs, nr, nc), (-nr, -nc))
        if best_key is None or key > best_key:
            best_key, best = key, (nr, nc)
    if best is not None:
        return best

    # Boxed in: hop over our own land toward fresh neutral ground.
    if army < 3:
        return None
    empties = [
        (rr, cc)
        for rr in range(obs.H)
        for cc in range(obs.W)
        if obs.owner_grid[rr][cc] == 0
        and obs.type_grid[rr][cc] == 1
        and (rr, cc) not in visited
    ]
    if not empties:
        return None
    dist = multi_bfs(obs, empties)
    if dist[r][c] >= UNREACHABLE or dist[r][c] > config.chain_detour:
        return None
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W) or (nr, nc) in visited:
            continue
        if is_passable(obs.type_grid[nr][nc]) and dist[nr][nc] < dist[r][c]:
            return (nr, nc)
    return None


def opening_move(obs, mem: BlitzMemory, config: BlitzConfig, bias=None, deadline: int | None = None):
    """One opening move: extend the live chain, or launch a new one.

    A move leaves one army behind, so walking a stack of ``A`` outward
    captures ``A - 1`` cells in ``A - 1`` turns with zero waste. Also serves
    as the between-waves rebuild (with the rebuild's own deadline).
    """
    general = mem.my_general
    if general is None:
        return None

    head = mem.chain_head
    if (
        head is not None
        and obs.owner_grid[head[0]][head[1]] == 1
        and obs.army_grid[head[0]][head[1]] >= 2
    ):
        step = chain_step(obs, head, bias, mem.chain_visited, config)
        if step is not None:
            mem.chain_head = step
            mem.chain_visited.add(step)
            return _move(head, step)

    mem.chain_head = None
    if should_launch(obs, general, config, deadline):
        mem.chain_visited = {general}
        step = chain_step(obs, general, bias, mem.chain_visited, config)
        if step is not None:
            mem.chain_head = step
            mem.chain_visited.add(step)
            return _move(general, step)

    # Spare army on old chains keeps nibbling; the general's stack is
    # reserved for the next chain launch.
    return expansion_step(obs, reserve={general}, bias_dist=bias)


# ------------------------------------------------------------------ assault
def tile_cost(obs, r: int, c: int, config: BlitzConfig) -> float | None:
    """Cost of routing the strike through ``(r, c)``; None = impassable.

    Own cells are discounted (they add army to the stack), defended enemy
    cells are surcharged. Enemy castles are just enemy cells — capturing one
    en route is profit, not a wall.
    """
    if not is_passable(obs.type_grid[r][c]):
        return None
    owner = obs.owner_grid[r][c]
    if owner == 1:
        discount = config.own_tile_bonus * min(
            obs.army_grid[r][c], config.own_tile_bonus_cap
        )
        return max(0.1, 1.0 - discount)
    if owner == 2:
        return 1.0 + config.enemy_cost_per_army * obs.army_grid[r][c]
    return config.neutral_cost


def cost_field(obs, target: Cell, config: BlitzConfig) -> list[list[float]]:
    """Dijkstra potential: cost of reaching ``target`` from every cell."""
    H, W = obs.H, obs.W
    dist = [[INF] * W for _ in range(H)]
    tr, tc = target
    if not (0 <= tr < H and 0 <= tc < W):
        return dist
    dist[tr][tc] = 0.0
    queue: list[tuple[float, Cell]] = [(0.0, target)]
    while queue:
        d, (r, c) = heapq.heappop(queue)
        if d > dist[r][c]:
            continue
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            cost = tile_cost(obs, nr, nc, config)
            if cost is None:
                continue
            nd = d + cost
            if nd < dist[nr][nc] - 1e-9:
                dist[nr][nc] = nd
                heapq.heappush(queue, (nd, (nr, nc)))
    return dist


def ensure_stack(obs, mem: BlitzMemory, config: BlitzConfig) -> tuple[Cell | None, bool]:
    """Latch onto the cell carrying the strike stack (sticky; jumps only when
    the current one is gone/spent or another cell grew ``restack_ratio``
    bigger). Returns ``(cell, restacked)``."""
    candidates = [
        cell for cell in _owned_cells(obs) if obs.army_grid[cell[0]][cell[1]] >= 2
    ]
    if not candidates:
        mem.stack = None
        return None, False
    general = mem.my_general
    best = max(
        candidates,
        key=lambda cell: (
            obs.army_grid[cell[0]][cell[1]],
            1 if cell == general else 0,
            (-cell[0], -cell[1]),
        ),
    )
    current = mem.stack
    alive = (
        current is not None
        and obs.owner_grid[current[0]][current[1]] == 1
        and obs.army_grid[current[0]][current[1]] >= 2
    )
    if not alive:
        mem.stack = best
        return best, current is not None
    if (
        best != current
        and obs.army_grid[best[0]][best[1]]
        > obs.army_grid[current[0]][current[1]] * config.restack_ratio
    ):
        mem.stack = best
        return best, True
    return current, False


def required_army(obs, mem: BlitzMemory, config: BlitzConfig, travel: int = 0) -> int:
    """Enough to beat the defence the scoreboard implies, plus one army per
    hop of the approach."""
    defence = int(config.strike_ratio * mem.opp.opponent_mobile())
    return max(config.min_strike_floor, defence) + int(
        config.travel_margin * max(0, travel)
    )


def rally_tile(obs, dist: list[list[float]]) -> Cell | None:
    """Our own cell closest (by strike cost) to the target — the wave is
    already facing the right way when it launches."""
    best, best_key = None, None
    for cell in _owned_cells(obs):
        d = dist[cell[0]][cell[1]]
        if d == INF:
            continue
        key = (d, cell)
        if best_key is None or key < best_key:
            best_key, best = key, cell
    return best


def start_wave(obs, mem: BlitzMemory, config: BlitzConfig) -> None:
    """Begin a rebuild → rally → push cycle, synced to the land bonus (right
    after a bonus every cell has two army and the rally sweeps them up)."""
    mem.committed = False
    mem.feed_ticks = 0
    period = config.opening_end or 50
    next_bonus = ((obs.turn // period) + 1) * period
    mem.rebuild_until = min(next_bonus, obs.turn + config.rebuild_ticks)
    mem.rally_until = mem.rebuild_until + config.rally_ticks


def strike_step(obs, stack: Cell, dist, config: BlitzConfig, target: Cell | None = None):
    """One step of the stack down the potential field. Refuses moves that
    would throw the stack away on a defender it cannot beat."""
    r, c = stack
    if dist[r][c] == INF:
        return None
    attacking = obs.army_grid[r][c] - 1
    if attacking < 1:
        return None
    best, best_key = None, None
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            continue
        if not is_passable(obs.type_grid[nr][nc]):
            continue
        if dist[nr][nc] >= dist[r][c]:
            continue
        own = obs.owner_grid[nr][nc] == 1
        if not own and obs.army_grid[nr][nc] >= attacking:
            continue
        pickup = obs.army_grid[nr][nc] if own else 0
        key = (dist[nr][nc], -pickup, (nr, nc))
        if best_key is None or key < best_key:
            best_key, best = key, (nr, nc)
    if best is None:
        return None
    return _move(stack, best)


def finishing_move(obs, mem: BlitzMemory):
    """Capture the enemy general right now if any cell of ours can. From the
    deathtouch turn any two-army neighbour suffices."""
    target = mem.belief.enemy_general
    if target is None:
        return None
    tr, tc = target
    if obs.owner_grid[tr][tc] == 1:
        return None
    deathtouch = obs.turn >= DEATHTOUCH_TURN
    defenders = 1 if deathtouch else enemy_general_army(obs, target)
    best, best_army = None, None
    for dr, dc in DIRECTIONS:
        nr, nc = tr + dr, tc + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            continue
        if obs.owner_grid[nr][nc] != 1:
            continue
        army = obs.army_grid[nr][nc]
        if deathtouch:
            if army < 2:
                continue
        elif army - 1 <= defenders:
            continue
        if best_army is None or army > best_army:
            best_army, best = army, (nr, nc)
    if best is None:
        return None
    return _move(best, target)


def assault_move(obs, mem: BlitzMemory, config: BlitzConfig, target: Cell | None, bias=None):
    """One move of the wave cycle: rebuild → rally → push.

    Reaching ``required_army`` early launches early; the rally deadline
    launches late waves whether or not they are up to size — the turn 60–160
    window does not wait. A committed wave never stops to gather.
    """
    if target is None:
        return None
    stack, restacked = ensure_stack(obs, mem, config)
    if stack is None:
        return expansion_step(obs, bias_dist=bias)
    if mem.rally_until < 0:
        # First wave: no rebuild, the opening just finished.
        mem.rebuild_until = obs.turn
        mem.rally_until = obs.turn + config.rally_ticks

    dist = cost_field(obs, target, config)
    hops = multi_bfs(obs, [target])
    travel = hops[stack[0]][stack[1]]
    travel = travel if travel < UNREACHABLE else 0
    stack_army = obs.army_grid[stack[0]][stack[1]]

    if mem.committed and (restacked or stack_army < config.spent_army):
        start_wave(obs, mem, config)

    if not mem.committed:
        need = required_army(obs, mem, config, travel)
        # The deadline must not scale with the opponent, or the wave never
        # launches at all.
        deadline = (
            obs.turn >= mem.rally_until and stack_army >= config.min_strike_floor
        )
        if stack_army >= need or deadline:
            mem.committed = True
            mem.strikes += 1

    if mem.committed:
        mem.phase = PHASE_ASSAULT
        push = strike_step(obs, stack, dist, config, target=target)
        if push is not None:
            mem.stack = (
                push[1] + DIRECTIONS[push[3]][0],
                push[2] + DIRECTIONS[push[3]][1],
            )
            mem.feed_ticks = 0
            return push
        # Blocked — every forward step is a fight we would lose. Reinforce
        # for a while, then let the wave lapse and rebuild.
        mem.feed_ticks += 1
        if mem.feed_ticks > config.feed_budget:
            start_wave(obs, mem, config)
        else:
            feed = gather_step(obs, stack, min_army=2, exclude={stack})
            if feed is not None:
                return feed

    if obs.turn < mem.rebuild_until:
        # Chain-expand exactly as in the opening: the general's regen turns
        # into land, and the next land bonus turns that land into the army
        # the following wave rallies.
        mem.phase = PHASE_REBUILD
        grow = opening_move(obs, mem, config, bias, deadline=mem.rebuild_until)
        if grow is not None:
            return grow

    mem.phase = PHASE_RALLY
    rally = rally_tile(obs, dist)
    if rally is not None:
        if config.collection_walk and rally != stack:
            # Collection walk: move the *stack* to the front through our own
            # surplus rather than ferrying stacks back to it.
            walk = strike_step(obs, stack, cost_field(obs, rally, config), config, target=rally)
            if walk is not None:
                mem.stack = (
                    walk[1] + DIRECTIONS[walk[3]][0],
                    walk[2] + DIRECTIONS[walk[3]][1],
                )
                return walk
        concentrate = gather_step(obs, rally, min_army=2)
        if concentrate is not None:
            return concentrate
    return expansion_step(obs, bias_dist=bias)


def blitz_move(obs, mem: BlitzMemory, config: BlitzConfig | None = None):
    """Blitz's move for this turn (a unified action, possibly PASS)."""
    config = config or BlitzConfig()
    mem.observe(obs)

    finish = finishing_move(obs, mem)
    if finish is not None:
        mem.phase = PHASE_FINISH
        return finish

    if mem.my_general is None:
        mem.phase = PHASE_IDLE
        return expansion_step(obs) or PASS

    if obs.turn >= CHASE_DEFEND_FROM:
        chase = _STRATEGY.chase_defence(obs, mem.my_general)
        if chase is not None:
            mem.phase = PHASE_DEFEND
            return chase

    deficit, threat_cell = home_threat(obs, mem, config)
    if deficit > 0:
        move = defense_move(obs, mem, config, threat_cell)
        if move is not None:
            mem.phase = PHASE_DEFEND
            dest = (move[1] + DIRECTIONS[move[3]][0], move[2] + DIRECTIONS[move[3]][1])
            mem.stack = dest if obs.owner_grid[dest[0]][dest[1]] == 1 else None
            return move

    bias = bias_distances(obs, mem)

    if obs.turn < config.opening_end:
        # PASS here means "the general is still accumulating" — the opening
        # is army-bound, not move-bound.
        mem.phase = PHASE_OPENING
        return opening_move(obs, mem, config, bias) or PASS

    target = strike_target(obs, mem, config)
    move = assault_move(obs, mem, config, target, bias)
    if move is not None:
        return move  # assault_move sets the sub-phase itself

    move = expansion_step(obs, bias_dist=bias)
    mem.phase = PHASE_REBUILD if move is not None else PHASE_IDLE
    return move or PASS


class BlitzCore:
    """Stateful wrapper around :func:`blitz_move`, composable by proteus."""

    def __init__(self, player_id: int, H: int, W: int,
                 config: BlitzConfig | None = None,
                 model: OpponentModel | None = None) -> None:
        self.player_id = player_id
        self.H = H
        self.W = W
        self.config = config or BlitzConfig()
        self.memory = BlitzMemory(opp=model or OpponentModel())

    @property
    def phase(self) -> str:
        return self.memory.phase

    def observe(self, obs) -> None:
        """Keep memory current without moving (proteus warms inactive cores)."""
        self.memory.observe(obs)

    def decide(self, obs):
        return blitz_move(obs, self.memory, self.config)


class Agent:
    """Arena entrypoint (contract: agent_class(player_id=..., H=..., W=...))."""

    def __init__(self, player_id: int, H: int, W: int) -> None:
        self._core = BlitzCore(player_id, H, W)

    def act(self, obs):
        return self._core.decide(obs)

    def telemetry_extras(self) -> dict:
        mem = self._core.memory
        return {
            "phase": mem.phase,
            "strikes": mem.strikes,
            "enemy_general_sighted": 1 if mem.belief.enemy_general else 0,
        }
