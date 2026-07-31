"""Boom — fast-expand economy (spec: docs/research/strategies/boom.md).

Grid-native rewrite of the generals-bot Boom strategy. Priority ladder each
turn:

1. **Finish** — take the enemy general if we can, right now.
2. **Defend** — an enemy is at the door and home cannot cover it.
3. **Evict** — kill an enemy cell that pushed into our half (two-land swing).
4. **Free captures** — never skipped, not even mid-attack: this is what
   turns every 50-turn land bonus into a burst of new cells.
5. **Attack** — once the endgame latch flips: collect one fist, walk it in.
6. **Build** — a castle, when the price is small relative to spare army
   (the arena replacement for the source's neutral-city purchases).
7. **Run** — send the bank on an expansion run once it pays for the walk.
8. **Consolidate** — walk stranded army to wherever it is spent next.
"""
from __future__ import annotations

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
    build_cost,
    fog_reveal,
    gather_step,
    mirror_tile,
    multi_bfs,
    next_step_toward,
    own_structures,
    visible_enemy_tiles,
)

Cell = tuple[int, int]

_STRATEGY = StrategyContext(defend_from=780, sentry_from=700)

DEATHTOUCH_TURN = 800


@dataclass(frozen=True)
class BoomParams:
    """Behaviour knobs. Source-tuned values carry over as hypotheses (the
    turn cadence matches); the build knobs replace the source's city knobs."""

    # ------------------------------------------------------------ expansion
    run_slack: int = 0
    """Extra army required beyond ``distance to nearest neutral`` before a
    stack is sent on a run (break-even is 0: one travel move per capture is
    free at one production per two turns)."""
    max_run_dist: int = 25
    """Ignore neutral land further than this from the moving stack."""
    avoid_enemy_adjacent_until: int = 160
    """Until this turn prefer neutral cells not adjacent to the enemy."""

    # ----------------------------------------------------------- home guard
    min_guard: int = 2
    guard_growth_per_100: float = 3.0
    guard_decay_per_tile: float = 1.0
    """Army an attacker is assumed to lose per cell of our land it crosses."""
    guard_margin: int = 2
    guard_memory_factor: float = 0.5
    """Weight applied to a threat we remember but cannot currently see."""
    threat_dist: int = 12
    intercept_dist: int = 8
    max_guard_fraction: float = 0.7
    defenders_counted: int = 2
    """Nearby stacks assumed to make it home in time (one move per turn)."""
    defend_margin: float = 1.0

    # --------------------------------------------------------------- builds
    build_earliest_turn: int = 60
    """Do not even look at castle builds before this turn."""
    build_spare_multiple: float = 2.0
    """Build only when ``cost * this <= spare army``."""
    build_keep_multiple: float = 1.5
    """Keep a started build plan alive while ``cost * this <= spare``."""
    build_keep: int = 3
    """Extra army banked beyond the price, left as the castle's garrison."""

    # -------------------------------------------------------------- endgame
    commit_turn: int = 200
    endgame_ratio: float = 1.5
    counter_drop: int = 20
    counter_ratio: float = 1.15
    no_land_attack_turn: int = 150
    stall_window: int = 40
    stall_growth: int = 2
    contest_turn: int = 120
    stall_attack_ratio: float = 1.0
    abandon_ratio: float = 0.95
    strike_floor: int = 25
    strike_fraction: float = 0.8
    max_gather_ticks: int = 60
    collect_radius: int = 12
    retarget_ticks: int = 40
    min_strike_army: int = 15
    force_commit_turn: int = 950
    """Arena addition: with a hard draw at 1200 turns, flip the attack latch
    unconditionally this late — hoarding into a draw wins nothing."""


@dataclass
class BoomMemory:
    """Everything Boom remembers between turns."""

    model: OpponentModel = field(default_factory=OpponentModel)
    belief: object = field(default_factory=_STRATEGY.BeliefState)
    my_general: Cell | None = None
    #: Current castle build plan: the cell to build on, and the funding stack.
    build_target: Cell | None = None
    build_stack: Cell | None = None
    #: Cell holding the strike stack during the endgame.
    strike_tile: Cell | None = None
    attack_target: Cell | None = None
    target_age: int = 0
    striking: bool = False
    phase: str = "open"
    guard: int = 0
    gather_ticks: int = 0
    reason: str = ""
    last_turn: int = -1

    def observe(self, obs) -> None:
        """Fold this turn's observation in. Idempotent within a turn."""
        if obs.turn == self.last_turn:
            return
        self.last_turn = obs.turn
        self.model.update(obs)
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


def _neighbors(obs, cell: Cell):
    r, c = cell
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < obs.H and 0 <= nc < obs.W:
            yield (nr, nc)


# ---------------------------------------------------------------- home guard
def guard_need(obs, model: OpponentModel, params: BoomParams) -> int:
    """Defensive power the general needs available right now. The threat is
    discounted one army per cell of our land it must cross (it bleeds out)."""
    base = params.min_guard + int(obs.turn * params.guard_growth_per_100 / 100.0)

    scaled = 0.0
    if model.closest_enemy_dist_now < UNREACHABLE:
        scaled = (
            model.biggest_enemy_stack_now
            - params.guard_decay_per_tile * model.closest_enemy_dist_now
        )
    elif model.closest_enemy_dist < UNREACHABLE:
        # Nothing in sight, but we have met them: half-weight the memory.
        scaled = (
            model.biggest_enemy_stack
            - params.guard_decay_per_tile * model.closest_enemy_dist
        ) * params.guard_memory_factor

    need = base
    if scaled > 0:
        need = max(need, int(scaled) + params.guard_margin)

    if obs.my_army > 0:
        cap = max(params.min_guard, int(obs.my_army * params.max_guard_fraction))
        need = min(need, cap)
    return max(0, need)


def enemy_arrival(model: OpponentModel) -> int:
    dist = model.closest_enemy_dist_now
    if dist >= UNREACHABLE:
        dist = model.closest_enemy_dist
    return dist


def home_power(obs, model: OpponentModel, params: BoomParams) -> int:
    """Army on the general plus the few biggest stacks close enough to walk
    home in time — deliberately not the sum of every two-army cell."""
    general = model.my_general
    if general is None:
        return 0
    horizon = enemy_arrival(model)
    if horizon >= UNREACHABLE:
        horizon = params.threat_dist
    horizon = max(1, min(horizon, params.threat_dist))
    dist = multi_bfs(obs, [general])
    reserves = sorted(
        (
            obs.army_grid[r][c] - 1
            for r, c in _owned_cells(obs)
            if (r, c) != general
            and dist[r][c] <= horizon
            and obs.army_grid[r][c] > 1
        ),
        reverse=True,
    )
    gr, gc = general
    return obs.army_grid[gr][gc] + sum(reserves[: params.defenders_counted])


def general_free(obs, model: OpponentModel, params: BoomParams, need: int) -> bool:
    """Whether the general's stack may be spent on an expansion run."""
    dist = enemy_arrival(model)
    if dist >= UNREACHABLE:
        return True  # never seen an enemy cell: nothing can reach us yet
    general = model.my_general
    if general is None:
        return True
    gr, gc = general
    without_bank = home_power(obs, model, params) - max(
        0, obs.army_grid[gr][gc] - 1
    )
    # The general regenerates one army per two turns while they walk.
    return without_bank + dist // 2 >= need


def threatened(obs, model: OpponentModel, params: BoomParams, need: int) -> bool:
    """An enemy is at the door and home cannot comfortably cover it."""
    if model.my_general is None:
        return False
    if model.closest_enemy_dist_now > params.threat_dist:
        return False
    return home_power(obs, model, params) < need * params.defend_margin


def intercept_move(obs, model: OpponentModel, params: BoomParams,
                   max_dist: int, reserve: set[Cell] | None = None):
    """Kill an enemy cell that pushed into our half, closest first — only
    when we win the exchange outright, and never with a locked bank."""
    general = model.my_general
    if general is None:
        return None
    if model.closest_enemy_dist_now > max_dist:
        return None
    reserve = reserve or set()
    dist = multi_bfs(obs, [general])
    best, best_key = None, None
    for src in _owned_cells(obs):
        if src in reserve:
            continue
        attack = obs.army_grid[src[0]][src[1]] - 1
        if attack < 1:
            continue
        for tgt in _neighbors(obs, src):
            tr, tc = tgt
            if obs.owner_grid[tr][tc] != 2:
                continue
            if dist[tr][tc] > max_dist:
                continue
            if attack <= obs.army_grid[tr][tc]:
                continue
            key = (
                -dist[tr][tc],
                obs.army_grid[tr][tc],
                -obs.army_grid[src[0]][src[1]],
                (-tr, -tc),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (src, tgt)
    if best is None:
        return None
    return _move(best[0], best[1])


def _walk_home(obs, params: BoomParams, general: Cell):
    """Send the single biggest reserve home whole — a stack walking over our
    own land keeps its size; a ferry delivers one army per move."""
    dist = multi_bfs(obs, [general])
    best, best_army = None, 1
    for cell in _owned_cells(obs):
        d = dist[cell[0]][cell[1]]
        if cell == general or d >= UNREACHABLE or d > params.threat_dist:
            continue
        if obs.army_grid[cell[0]][cell[1]] > best_army:
            best, best_army = cell, obs.army_grid[cell[0]][cell[1]]
    if best is None:
        return None
    for n in _neighbors(obs, best):
        if dist[n[0]][n[1]] < dist[best[0]][best[1]] and is_passable(
            obs.type_grid[n[0]][n[1]]
        ):
            return _move(best, n)
    return None


def defense_move(obs, model: OpponentModel, params: BoomParams,
                 reserve: set[Cell] | None = None):
    """Fight the intruder if we can beat it, otherwise reinforce home."""
    move = intercept_move(obs, model, params, params.threat_dist, reserve)
    if move is not None:
        return move
    general = model.my_general
    if general is None:
        return None
    move = _walk_home(obs, params, general)
    if move is not None:
        return move
    return gather_step(obs, general, min_army=2, exclude={general})


# ------------------------------------------------------------------- collect
def collect_step(obs, cell: Cell, params: BoomParams,
                 avoid: set[Cell] | None = None):
    """Walk the stack on ``cell`` onto the nearest surplus army we own —
    every move collects the whole surplus of the cell it steps onto."""
    r, c = cell
    if obs.army_grid[r][c] < 2:
        return None
    avoid = avoid or set()
    best, best_army = None, 1
    for n in _neighbors(obs, cell):
        if n in avoid:
            continue
        if obs.owner_grid[n[0]][n[1]] == 1 and obs.army_grid[n[0]][n[1]] > best_army:
            best_army = obs.army_grid[n[0]][n[1]]
            best = n
    if best is not None:
        return _move(cell, best)

    def mine(rr: int, cc: int) -> bool:
        return obs.owner_grid[rr][cc] == 1 and (rr, cc) not in avoid

    surplus = [
        cl
        for cl in _owned_cells(obs)
        if cl != cell and cl not in avoid and obs.army_grid[cl[0]][cl[1]] >= 2
    ]
    if not surplus:
        return None
    dist = multi_bfs(obs, surplus, passable=mine)
    if dist[r][c] >= UNREACHABLE or dist[r][c] > params.collect_radius:
        return None
    for n in _neighbors(obs, cell):
        if mine(*n) and dist[n[0]][n[1]] < dist[r][c]:
            return _move(cell, n)
    return None


def biggest_stack(obs, exclude: set[Cell] | None = None) -> Cell | None:
    exclude = exclude or set()
    movable = [
        cell
        for cell in _owned_cells(obs)
        if cell not in exclude and obs.army_grid[cell[0]][cell[1]] > 1
    ]
    if not movable:
        return None
    return max(
        movable,
        key=lambda cell: (obs.army_grid[cell[0]][cell[1]], (-cell[0], -cell[1])),
    )


# ----------------------------------------------------------------- expansion
def _neutral_targets(obs) -> list[Cell]:
    """Visible neutral plains and plain fog — everything worth expanding at."""
    return [
        (r, c)
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.type_grid[r][c] == 0
        or (obs.owner_grid[r][c] == 0 and obs.type_grid[r][c] == 1)
    ]


def _routable(obs):
    """Expansion pathing: our land and neutral land only."""

    def ok(r: int, c: int) -> bool:
        return is_passable(obs.type_grid[r][c]) and obs.owner_grid[r][c] != 2

    return ok


def neutral_distance_map(obs) -> list[list[int]]:
    targets = _neutral_targets(obs)
    if not targets:
        return [[UNREACHABLE] * obs.W for _ in range(obs.H)]
    return multi_bfs(obs, targets, passable=_routable(obs))


def _enemy_adjacent(obs, cell: Cell) -> bool:
    return any(obs.owner_grid[n[0]][n[1]] == 2 for n in _neighbors(obs, cell))


def direct_capture(obs, params: BoomParams, reserve: set[Cell] | None = None,
                   allow_enemy: bool = False, general: Cell | None = None):
    """A capture that needs no travel. Sources are ranked to protect the
    accumulator: frontier stacks are spent before the general's bank."""
    reserve = reserve or set()
    best, best_key = None, None
    for src in _owned_cells(obs):
        if src in reserve:
            continue
        attack = obs.army_grid[src[0]][src[1]] - 1
        if attack < 1:
            continue
        for tgt in _neighbors(obs, src):
            tr, tc = tgt
            enemy = obs.owner_grid[tr][tc] == 2
            if enemy:
                if not allow_enemy:
                    continue
            elif not (obs.owner_grid[tr][tc] == 0 and obs.type_grid[tr][tc] == 1):
                continue
            if attack <= obs.army_grid[tr][tc]:
                continue
            calm = not (
                obs.turn < params.avoid_enemy_adjacent_until
                and _enemy_adjacent(obs, tgt)
            )
            key = (
                0 if enemy else 1,           # neutral land is cheaper
                0 if src == general else 1,  # spend frontier stacks, not the bank
                1 if calm else 0,
                fog_reveal(obs, tr, tc),
                -obs.army_grid[src[0]][src[1]],
                (-tr, -tc),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (src, tgt)
    if best is None:
        return None
    return _move(best[0], best[1])


def run_step(obs, params: BoomParams, reserve: set[Cell] | None = None):
    """One step of an expansion run: launch when ``army >= distance`` (one
    travel move per capture is free), walk the gradient collecting en route."""
    reserve = reserve or set()
    dist = neutral_distance_map(obs)
    passable = _routable(obs)
    best_src, best_key = None, None
    for src in _owned_cells(obs):
        if src in reserve:
            continue
        army = obs.army_grid[src[0]][src[1]]
        d = dist[src[0]][src[1]]
        if army < 2 or d >= UNREACHABLE or d > params.max_run_dist:
            continue
        if army < d + params.run_slack:
            continue  # not enough army to make the walk pay for itself
        key = (army - d, army, (-src[0], -src[1]))
        if best_key is None or key > best_key:
            best_key = key
            best_src = src
    if best_src is None:
        return None

    step, step_key = None, None
    for n in _neighbors(obs, best_src):
        if dist[n[0]][n[1]] >= dist[best_src[0]][best_src[1]] or not passable(*n):
            continue
        own = 1 if obs.owner_grid[n[0]][n[1]] == 1 else 0
        key = (own, obs.army_grid[n[0]][n[1]] if own else 0, (-n[0], -n[1]))
        if step_key is None or key > step_key:
            step_key = key
            step = n
    if step is None:
        return None
    return _move(best_src, step)


# ------------------------------------------------------------------- endgame
def expansion_stalled(model: OpponentModel, window: int, growth: int) -> bool:
    lands = model.my_land
    if len(lands) <= window:
        return False
    return lands[-1] - lands[-1 - window] <= growth


def endgame_ready(obs, model: OpponentModel, params: BoomParams,
                  expansion_available: bool, attacking: bool = False) -> bool:
    """The latch that flips Boom from booming to killing. Releasable — an
    attack while behind on army throws away a won game — except after
    ``force_commit_turn``, when a draw is the thing being avoided."""
    if obs.turn >= params.force_commit_turn:
        return True
    ratio = model.army_ratio()
    if attacking:
        return ratio >= params.abandon_ratio or not expansion_available
    if obs.turn >= params.no_land_attack_turn and not expansion_available:
        return True  # no land left to buy: army is only worth what it kills
    if (
        obs.turn >= params.no_land_attack_turn
        and ratio >= params.stall_attack_ratio
        and expansion_stalled(model, params.stall_window, params.stall_growth)
    ):
        return True
    if obs.turn < params.commit_turn:
        return False
    if ratio >= params.endgame_ratio:
        return True
    return (
        model.opp_total_drop() >= params.counter_drop
        and ratio >= params.counter_ratio
    )


def strike_threshold(model: OpponentModel, params: BoomParams) -> int:
    """Their total includes one army welded to every cell they own, so the
    fraction is measured against their *mobile* army."""
    return max(
        params.strike_floor,
        int(params.strike_fraction * model.opponent_mobile()),
    )


def guess_enemy_general(obs, mem: BoomMemory) -> Cell | None:
    """Their real position if ever seen; otherwise the fog behind their
    visible territory; before contact, the mirror-point prior."""
    if mem.belief.enemy_general is not None:
        return mem.belief.enemy_general
    enemies = visible_enemy_tiles(obs)
    if enemies:
        from_enemy = multi_bfs(obs, enemies)
        fog = [
            (r, c)
            for r in range(obs.H)
            for c in range(obs.W)
            if obs.type_grid[r][c] == 0 and from_enemy[r][c] <= 6
        ]
        if fog:
            mine = _owned_cells(obs)
            if mine:
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
            return min(fog)
    if mem.my_general is not None:
        return mirror_tile(obs, *mem.my_general)
    return None


def attack_target(obs, mem: BoomMemory, params: BoomParams) -> Cell | None:
    """Where the fist is aimed, held steady across turns (a wandering target
    is worse than a wrong one)."""
    if mem.belief.enemy_general is not None:
        mem.attack_target = mem.belief.enemy_general
        return mem.attack_target
    cell = mem.attack_target
    stale = (
        cell is None
        or obs.owner_grid[cell[0]][cell[1]] == 1
        or mem.target_age >= params.retarget_ticks
    )
    if stale:
        mem.attack_target = guess_enemy_general(obs, mem)
        mem.target_age = 0
    else:
        mem.target_age += 1
    return mem.attack_target


def _strike_tile(obs, mem: BoomMemory, params: BoomParams,
                 reserve: set[Cell] | None = None) -> Cell | None:
    """The stack we push with. Identity is tracked: a spent strike must drop
    back to collecting, not be silently replaced by the next biggest cell."""
    reserve = reserve or set()
    cell = mem.strike_tile
    alive = (
        cell is not None
        and cell not in reserve
        and obs.owner_grid[cell[0]][cell[1]] == 1
        and obs.army_grid[cell[0]][cell[1]] > 1
    )
    if mem.striking and not (
        alive and obs.army_grid[cell[0]][cell[1]] >= params.min_strike_army
    ):
        mem.striking = False  # the fist is spent: go and build another
        mem.gather_ticks = 0
        alive = False
    if alive:
        return cell
    cell = biggest_stack(obs, exclude=reserve)
    mem.strike_tile = cell
    return cell


def attack_move(obs, model: OpponentModel, mem: BoomMemory, params: BoomParams,
                reserve: set[Cell] | None = None):
    """Collect the army into one stack, then walk it at their general.
    Committing the economy does not mean committing the *general*."""
    target = attack_target(obs, mem, params)
    if target is None:
        return None

    cell = _strike_tile(obs, mem, params, reserve)
    if cell is None:
        return None

    if not mem.striking:
        mem.gather_ticks += 1
        enough = obs.army_grid[cell[0]][cell[1]] >= strike_threshold(model, params)
        if not enough and mem.gather_ticks < params.max_gather_ticks:
            move = collect_step(obs, cell, params, avoid=reserve)
            if move is not None:
                mem.strike_tile = (
                    move[1] + DIRECTIONS[move[3]][0],
                    move[2] + DIRECTIONS[move[3]][1],
                )
                return move
        mem.striking = True
        mem.gather_ticks = 0

    step = next_step_toward(obs, cell, [target])
    if step is not None:
        mem.strike_tile = step
        return _move(cell, step)
    mem.striking = False
    mem.strike_tile = None
    return None


# -------------------------------------------------------------------- builds
def spare_army(obs, guard: int) -> int:
    """Army we could actually spend: movable army minus the home guard."""
    movable = sum(
        max(0, obs.army_grid[r][c] - 1) for r, c in _owned_cells(obs)
    )
    return movable - guard


def choose_build_cell(obs, spare: int, params: BoomParams,
                      reserve: set[Cell] | None = None) -> Cell | None:
    """Cheapest owned plain cell worth building a castle on, preferring cells
    already holding army (less collecting to fund the build)."""
    reserve = reserve or set()
    structures = own_structures(obs)
    best, best_key = None, None
    for r, c in _owned_cells(obs):
        if obs.type_grid[r][c] != 1 or (r, c) in reserve:
            continue
        cost = build_cost(obs, r, c, structures)
        if cost * params.build_spare_multiple > spare:
            continue
        key = (cost, -obs.army_grid[r][c], (r, c))
        if best_key is None or key < best_key:
            best_key = key
            best = (r, c)
    return best


def build_move(obs, mem: BoomMemory, spare: int, params: BoomParams,
               reserve: set[Cell] | None = None):
    """Advance the castle build plan by one move (choosing a site if needed).

    The arena replacement for the source's city sieges: instead of walking a
    stack at a neutral city, the funding stack collects and walks onto the
    chosen cell, then converts to a castle (``2 r c 0 0``). A started plan
    survives anything short of the cell being lost or the spare evaporating.
    """
    if obs.turn < params.build_earliest_turn:
        return None
    reserve = reserve or set()
    structures = own_structures(obs)

    cell = mem.build_target
    if cell is not None:
        lost = obs.owner_grid[cell[0]][cell[1]] != 1 or obs.type_grid[cell[0]][cell[1]] != 1
        too_dear = (
            build_cost(obs, cell[0], cell[1], structures)
            * params.build_keep_multiple
            > spare
        )
        if lost or too_dear:
            cell = mem.build_target = None
            mem.build_stack = None
    if cell is None:
        cell = mem.build_target = choose_build_cell(obs, spare, params, reserve)
        mem.build_stack = None
    if cell is None:
        return None

    cost = build_cost(obs, cell[0], cell[1], structures)
    # Builds resolve before moves, so the army must already be standing there.
    if obs.army_grid[cell[0]][cell[1]] >= cost + params.build_keep:
        mem.build_target = None
        mem.build_stack = None
        return (2, cell[0], cell[1], 0, 0)

    stack = mem.build_stack
    if (
        stack is None
        or obs.owner_grid[stack[0]][stack[1]] != 1
        or obs.army_grid[stack[0]][stack[1]] < 2
        or stack in reserve
    ):
        stack = biggest_stack(obs, exclude=set(reserve) | {cell})
    if stack is None:
        return None
    mem.build_stack = stack

    if obs.army_grid[stack[0]][stack[1]] < cost + params.build_keep:
        move = collect_step(obs, stack, params, avoid=set(reserve) | {cell})
        if move is not None:
            mem.build_stack = (
                move[1] + DIRECTIONS[move[3]][0],
                move[2] + DIRECTIONS[move[3]][1],
            )
            return move
        mem.build_target = None  # cannot afford it after all
        mem.build_stack = None
        return None

    # Funded: walk the stack onto the build cell through our own land.
    step = next_step_toward(
        obs, stack, [cell],
        passable=lambda r, c: obs.owner_grid[r][c] == 1,
    )
    if step is not None:
        mem.build_stack = step
        return _move(stack, step)
    return None


# --------------------------------------------------------------- consolidate
def _consolidate(obs, neutral_dist, general: Cell | None, bank_locked: bool):
    """Walk stranded army to where it can become land (or defence)."""
    if general is not None and not bank_locked:
        return gather_step(obs, general, min_army=2, exclude={general})
    owned = [
        cell
        for cell in _owned_cells(obs)
        if neutral_dist[cell[0]][cell[1]] < UNREACHABLE
    ]
    if not owned:
        return None
    rally = min(
        owned,
        key=lambda cell: (
            neutral_dist[cell[0]][cell[1]],
            -obs.army_grid[cell[0]][cell[1]],
            cell,
        ),
    )
    exclude = {rally}
    if general is not None and bank_locked:
        exclude.add(general)
    return gather_step(obs, rally, min_army=2, exclude=exclude)


def _fallback(obs, reserve: set[Cell]):
    """Last resort: push the biggest stack enemy-ward, else shuffle it."""
    movable = [
        cell
        for cell in _owned_cells(obs)
        if cell not in reserve and obs.army_grid[cell[0]][cell[1]] > 1
    ]
    if not movable:
        return None
    src = max(
        movable,
        key=lambda cell: (obs.army_grid[cell[0]][cell[1]], (-cell[0], -cell[1])),
    )
    enemies = visible_enemy_tiles(obs)
    if enemies:
        dist = multi_bfs(obs, enemies)
        if dist[src[0]][src[1]] < UNREACHABLE:
            for n in _neighbors(obs, src):
                if dist[n[0]][n[1]] < dist[src[0]][src[1]] and is_passable(
                    obs.type_grid[n[0]][n[1]]
                ):
                    return _move(src, n)
    for n in _neighbors(obs, src):
        if is_passable(obs.type_grid[n[0]][n[1]]):
            return _move(src, n)
    return None


def _finishing_move(obs, mem: BoomMemory):
    """Take the enemy general the moment any cell of ours can."""
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


def choose_move(obs, mem: BoomMemory, params: BoomParams | None = None):
    """Pick Boom's move for this turn. Pure apart from mutating ``mem``."""
    params = params or BoomParams()
    mem.observe(obs)
    model = mem.model
    general = mem.my_general
    model.my_general = model.my_general or general

    # 0. Finish: nothing outranks ending the game.
    finish = _finishing_move(obs, mem)
    if finish is not None:
        return _remember(mem, finish, "finish")

    if obs.turn >= 780 and general is not None:
        chase = _STRATEGY.chase_defence(obs, general)
        if chase is not None:
            return _remember(mem, chase, "defend")

    need = guard_need(obs, model, params)
    mem.guard = need
    bank_locked = general is not None and not general_free(obs, model, params, need)
    reserve: set[Cell] = {general} if (bank_locked and general is not None) else set()

    # 1. Defence beats everything else: a lost general ends the game.
    if threatened(obs, model, params, need):
        mem.phase = "defend"
        move = defense_move(obs, model, params, reserve)
        if move is not None:
            return _remember(mem, move, "defend")

    # 2. Evict raiders: two-land swing, and it stops a raid early.
    move = intercept_move(obs, model, params, params.intercept_dist, reserve)
    if move is not None:
        return _remember(mem, move, "intercept")

    neutral_dist = neutral_distance_map(obs)
    expansion_possible = any(
        neutral_dist[r][c] <= params.max_run_dist for r, c in _owned_cells(obs)
    )

    # 3. Endgame decision (a latch, so the walk is not restarted every turn).
    attacking = endgame_ready(
        obs, model, params, expansion_possible, mem.phase == "attack"
    )
    if attacking:
        mem.phase = "attack"
    else:
        mem.striking = False
        mem.strike_tile = None
        mem.attack_target = None
        mem.gather_ticks = 0
        mem.phase = "expand" if obs.turn >= 50 else "open"

    # 4. Free captures — never skipped: every cell compounds.
    allow_enemy = not expansion_possible or obs.turn >= params.contest_turn
    capture_reserve = set(reserve)
    if attacking and mem.strike_tile is not None:
        capture_reserve.add(mem.strike_tile)
    move = direct_capture(
        obs, params, capture_reserve, allow_enemy=allow_enemy, general=general
    )
    if move is not None:
        return _remember(mem, move, "capture")

    if attacking:
        move = attack_move(obs, model, mem, params, reserve)
        if move is not None:
            return _remember(mem, move, "attack")

    # 5. Castle builds (the arena's replacement for city purchases).
    move = build_move(obs, mem, spare_army(obs, need), params, reserve)
    if move is not None:
        return _remember(mem, move, "build")

    # 6. Expansion runs.
    move = run_step(obs, params, reserve)
    if move is not None:
        return _remember(mem, move, "run")

    # 7. Consolidate stranded army toward wherever it is spent next.
    move = _consolidate(obs, neutral_dist, general, bank_locked)
    if move is not None:
        return _remember(mem, move, "consolidate")

    return _remember(mem, _fallback(obs, reserve), "fallback")


def _remember(mem: BoomMemory, move, reason: str):
    mem.reason = reason if move is not None else "pass"
    return move if move is not None else PASS


class BoomCore:
    """Stateful wrapper around :func:`choose_move`, composable by proteus."""

    def __init__(self, player_id: int, H: int, W: int,
                 params: BoomParams | None = None,
                 model: OpponentModel | None = None) -> None:
        self.player_id = player_id
        self.H = H
        self.W = W
        self.params = params or BoomParams()
        self.memory = BoomMemory(model=model or OpponentModel())

    @property
    def phase(self) -> str:
        return self.memory.phase

    def observe(self, obs) -> None:
        self.memory.observe(obs)

    def decide(self, obs):
        return choose_move(obs, self.memory, self.params)


class Agent:
    """Arena entrypoint."""

    def __init__(self, player_id: int, H: int, W: int) -> None:
        self._core = BoomCore(player_id, H, W)

    def act(self, obs):
        return self._core.decide(obs)

    def telemetry_extras(self) -> dict:
        mem = self._core.memory
        return {
            "phase": mem.phase,
            "reason": mem.reason,
            "guard": mem.guard,
        }
