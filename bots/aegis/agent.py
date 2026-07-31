"""Aegis — turtle + counterattack (spec: docs/research/strategies/aegis.md).

Grid-native rewrite of the generals-bot Aegis strategy. Priority ladder:

1. **Kill shot** — the enemy general is adjacent and takeable.
2. **Interception** — destroy an incoming stack with local superiority;
   failing that, a sortie by the general itself against a doorstep stack.
3. **Man the door / emergency garrison** — the keep steps back onto a
   threatened general; funnel army home when the wave is imminent.
4. **Hunt** — meet a raider rather than let it eat the map.
5. **Home deficit** — rally the border army on the keep.
6. **Counterattack** — their attack is spent (or we simply outgun them):
   commit at their general for a bounded window.
7. **Economy** — safe home castle builds once the garrison is covered.
8. **Growth** — compact perimeter claims; border raids; marches.
9. **Fallback** — consolidate or shuffle.

The keep — a reserve parked one cell *off* the general — is the load-
bearing idea: on the general the army only fights the battle the opponent
picks; beside it, the same army also hunts raiders and leads the counter.
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
    expansion_step,
    gather_step,
    mirror_tile,
    multi_bfs,
    next_step_toward,
    own_structures,
    visible_enemy_tiles,
)

Cell = tuple[int, int]

_STRATEGY = StrategyContext(defend_from=780)

DEATHTOUCH_TURN = 800

FAR = 20
"""Assumed threat distance when we have never seen the opponent."""

AWAY_CAP = 30
"""Clamp for the 'distance from the enemy' preference term."""


@dataclass(frozen=True)
class AegisConfig:
    """Knobs for the turtle/counter policy. All distances are BFS steps.
    Source-tuned values carry over as hypotheses."""

    # --- defence sizing ---------------------------------------------------
    margin: int = 3
    min_defense: int = 2
    precontact_cap: int = 6
    wave_frac: float = 0.10
    """Share of their *mobile* army we assume fog is hiding in one wave."""
    wave_growth: float = 1.0
    min_wave: int = 8
    max_defense_frac: float = 0.50
    max_attrition_credit: int = 10
    ring_horizon: int = 5
    pull_radius: int = 10
    lock_radius: int = 5
    imminent_radius: int = 4
    intercept_radius: int = 8
    intercept_min_army: int = 5
    hunt_radius: int = 7
    hunt_leash: int = 6
    general_sortie_cover: int = 25

    # --- expansion --------------------------------------------------------
    chain_expand: bool = False
    open_free_turn: int = 45

    # --- counterattack ----------------------------------------------------
    counter_drop: int = 12
    counter_drop_frac: float = 0.20
    counter_window: int = 25
    counter_hold: int = 80
    counter_edge: float = 1.15
    counter_abort_loss: float = 0.15
    dead_stack_army: int = 15
    strong_ratio: float = 1.5
    strong_turn: int = 200
    stall_free_land: int = 2
    stall_min_turn: int = 120
    stall_ratio: float = 0.75
    late_turn: int = 350
    """After this turn, army parity is enough to commit (anti-stalemate;
    with the 1200-turn draw this doubles as the draw guard)."""
    min_strike: int = 8
    strike_frac: float = 0.15
    collection_walk: bool = True
    detour_army: int = 15
    gather_ticks: int = 25
    raid_open_army: int = 10
    sortie_ratio: float = 2.5

    # --- economy ----------------------------------------------------------
    econ_turn: int = 300
    econ_early_turn: int = 150
    econ_tile_ratio: float = 1.5
    econ_defense_frac: float = 0.35
    raid_radius: int = 999
    build_anytime: bool = True
    """Build castles whenever the garrison is covered, not only in econ mode
    (was ``city_anytime``)."""
    build_radius: int = 10
    """Only build this close to home ('safe' castles; was ``city_radius``)."""
    build_keep: int = 3
    """Army banked beyond the price, left as the castle's garrison."""
    build_reserve: int = 0
    """Spare army required on top of the price before staging a build."""
    max_castles: int = 3
    """Compounding stops paying once prices outrun the turtle's bank."""


@dataclass
class AegisMemory:
    """Rolling state of one Aegis game."""

    model: OpponentModel = field(default_factory=OpponentModel)
    belief: object = field(default_factory=_STRATEGY.BeliefState)

    home: Cell | None = None
    was_attacked: bool = False
    peak_opp_total: int = 0
    peak_my_total: int = 0

    counter_until: int = -1
    counter_started: int | None = None
    counter_start_tiles: int = 0
    strike_tile: Cell | None = None
    counter_goal: Cell | None = None

    econ_since: int | None = None
    phase: str = "open"
    lost_tiles: int = 0
    peak_my_tiles: int = 0
    last_contact_turn: int | None = None
    keep: Cell | None = None
    free_land: int = 1
    build_site: Cell | None = None
    last_turn: int = -1

    def observe(self, obs) -> None:
        """Fold one turn of observations in. Idempotent within a turn."""
        if obs.turn == self.last_turn:
            return
        self.last_turn = obs.turn
        self.model.update(obs)
        if self.home is None:
            self.home = locate_own_general(obs)
        self.belief.update(obs, self.home)

        self.peak_opp_total = max(self.peak_opp_total, obs.opp_army)
        self.peak_my_total = max(self.peak_my_total, obs.my_army)
        self.peak_my_tiles = max(self.peak_my_tiles, obs.my_land)
        self.lost_tiles = max(0, self.peak_my_tiles - obs.my_land)

        d = self.model.closest_enemy_dist_now
        if d < UNREACHABLE:
            self.last_contact_turn = obs.turn
            if d <= 6:
                self.was_attacked = True
        if self.lost_tiles >= 2:
            self.was_attacked = True

    def counter_active(self, turn: int) -> bool:
        return turn <= self.counter_until

    def start_counter(self, turn: int, hold: int) -> None:
        if not self.counter_active(turn):
            self.counter_started = turn
            self.counter_start_tiles = (
                self.model.my_land[-1] if self.model.my_land else 0
            )
        self.counter_until = max(self.counter_until, turn + hold)

    def abort_counter(self) -> None:
        self.counter_until = -1
        self.strike_tile = None
        self.counter_goal = None


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


def _army(obs, cell: Cell) -> int:
    return obs.army_grid[cell[0]][cell[1]]


def _mine(obs, cell: Cell) -> bool:
    return obs.owner_grid[cell[0]][cell[1]] == 1


def _enemy(obs, cell: Cell) -> bool:
    return obs.owner_grid[cell[0]][cell[1]] == 2


# ------------------------------------------------------------------ defence
def max_strike_force(obs) -> int:
    """Largest single stack the opponent could possibly have assembled:
    every cell they hold except one must keep one army behind."""
    return max(0, obs.opp_army - max(obs.opp_land - 1, 0))


def threat_distance(memory: AegisMemory) -> int:
    """Distance to the nearest accountable enemy; decays one cell per turn
    once they leave vision (a single sighting must not pin us forever)."""
    model = memory.model
    d = model.closest_enemy_dist_now
    if d < UNREACHABLE:
        return min(d, FAR)
    if model.closest_enemy_dist >= UNREACHABLE:
        return FAR
    stale = 0
    if memory.last_contact_turn is not None and model.turns:
        stale = max(0, model.turns[-1] - memory.last_contact_turn)
    return min(model.closest_enemy_dist + stale, FAR)


def wave_estimate(obs, memory: AegisMemory, cfg: AegisConfig) -> int:
    """Army to expect in a single enemy wave: the largest stack they have
    shown, or a share of their mobile army, capped by what is possible."""
    model = memory.model
    mobile = max(0, obs.opp_army - obs.opp_land)
    seen = max(model.biggest_enemy_stack, model.biggest_enemy_stack_now)
    est = max(
        int(seen * cfg.wave_growth),
        int(cfg.wave_frac * mobile),
        cfg.min_wave,
    )
    return min(est, max_strike_force(obs))


def defense_need(obs, memory: AegisMemory, cfg: AegisConfig) -> int:
    """How much army the defence should be able to bring to the general."""
    model = memory.model

    # Opening: nobody can cross the map this early. Spend it all.
    if model.first_contact_turn is None and obs.turn < cfg.open_free_turn:
        return 0

    dist = threat_distance(memory)
    attrition = max(0, min(dist, cfg.max_attrition_credit) - 1)
    need = wave_estimate(obs, memory, cfg) - attrition + cfg.margin

    if model.first_contact_turn is None:
        need = min(need, max(cfg.precontact_cap, int(cfg.wave_frac * obs.my_army)))

    cap = (
        obs.my_army
        if dist <= cfg.imminent_radius
        else int(cfg.max_defense_frac * obs.my_army)
    )
    if econ_active(obs, memory, cfg) and dist > cfg.imminent_radius:
        cap = min(cap, int(cfg.econ_defense_frac * obs.my_army))
    need = min(need, max(cfg.min_defense, cap))
    return max(cfg.min_defense, min(need, obs.my_army))


def reachable_defense(obs, home: Cell, dist_home, horizon: int = 5) -> int:
    """Army that can be funnelled onto the general within ``horizon`` steps."""
    total = _army(obs, home) if _mine(obs, home) else 0
    for cell in _owned_cells(obs):
        if cell == home:
            continue
        if dist_home[cell[0]][cell[1]] <= horizon:
            total += max(0, _army(obs, cell) - 1)
    return total


def econ_active(obs, memory: AegisMemory, cfg: AegisConfig) -> bool:
    """A pure turtle loses on the scoreboard to a peaceful opponent, so we
    time out of it — early if visibly out-expanded, at ``econ_turn`` flat."""
    if obs.turn >= cfg.econ_turn:
        return True
    if memory.was_attacked:
        return False
    if obs.turn >= cfg.econ_early_turn:
        if obs.opp_land > max(obs.my_land, 1) * cfg.econ_tile_ratio:
            return True
    return False


def threat_tiles(obs, dist_home, cfg: AegisConfig) -> list[tuple[Cell, int]]:
    """Visible enemy stacks worth reacting to, biggest first."""
    out = [
        (cell, _army(obs, cell))
        for cell in visible_enemy_tiles(obs)
        if _army(obs, cell) >= cfg.intercept_min_army
        and dist_home[cell[0]][cell[1]] <= cfg.intercept_radius
    ]
    out.sort(key=lambda t: (-t[1], dist_home[t[0][0]][t[0][1]], t[0]))
    return out


def intercept_move(obs, home: Cell, dist_home, cfg: AegisConfig):
    """Destroy an incoming stack with genuine local superiority. Never
    spends the general — it defends by sitting still (ties go to it)."""
    best, best_key = None, None
    for cell, army in threat_tiles(obs, dist_home, cfg):
        for n in _neighbors(obs, cell):
            if n == home or not _mine(obs, n):
                continue
            if _army(obs, n) - 1 <= army:
                continue
            key = (army, -dist_home[cell[0]][cell[1]], -_army(obs, n))
            if best_key is None or key > best_key:
                best_key = key
                best = (n, cell)
    return _move(best[0], best[1]) if best is not None else None


def general_intercept(obs, home: Cell, dist_home, cfg: AegisConfig):
    """Let the garrison kill a stack on the doorstep — only when exactly one
    enemy is adjacent (no second stack to exploit the tick of exposure) and
    the survivors still cover the next wave."""
    if not _mine(obs, home):
        return None
    garrison = _army(obs, home) - 1
    adjacent_enemies = [n for n in _neighbors(obs, home) if _enemy(obs, n)]
    if len(adjacent_enemies) != 1:
        return None
    target = adjacent_enemies[0]
    if _army(obs, target) < cfg.intercept_min_army:
        return None
    if garrison <= _army(obs, target):
        return None
    survivors = garrison - _army(obs, target)
    rest = max(0, obs.opp_army - _army(obs, target))
    if survivors < min(int(cfg.wave_frac * rest), cfg.general_sortie_cover):
        return None
    return _move(home, target)


def man_the_door(obs, home: Cell, keep: Cell | None, need: int):
    """Step the keep back onto the general when the door is threatened."""
    if keep is None or keep == home or not _mine(obs, keep):
        return None
    if _army(obs, home) >= need:
        return None
    dist = multi_bfs(obs, [home])
    danger = max(
        (
            _army(obs, cell)
            for cell in visible_enemy_tiles(obs)
            if dist[cell[0]][cell[1]] <= 2
        ),
        default=0,
    )
    if danger and _army(obs, home) <= danger and _army(obs, keep) > 1:
        return _move(keep, home)
    return None


def keep_tile(obs, home: Cell, dist_home, cfg: AegisConfig) -> Cell:
    """Where the field army lives: next to the general, threat-side."""
    options = [n for n in _neighbors(obs, home) if _mine(obs, n)]
    if not options:
        return home
    threats = threat_tiles(obs, dist_home, cfg)
    if threats:
        d_threat = multi_bfs(obs, [t for t, _ in threats])
        return min(
            options,
            key=lambda cell: (d_threat[cell[0]][cell[1]], -_army(obs, cell), cell),
        )
    return max(options, key=lambda cell: (_army(obs, cell), (-cell[0], -cell[1])))


def hunt_move(obs, home: Cell, dist_home, cfg: AegisConfig):
    """Walk the field army out to kill a raider before it eats the map."""
    threats = threat_tiles(obs, dist_home, cfg)
    if not threats:
        return None
    target, target_army = threats[0]
    d_target = multi_bfs(obs, [target])

    best, best_key = None, None
    for src in _owned_cells(obs):
        if src == home:
            continue  # the general never leaves; see general_intercept
        if _army(obs, src) - 1 <= target_army:
            continue
        d = d_target[src[0]][src[1]]
        if d > cfg.hunt_radius or d == 0:
            continue
        step = next_step_toward(obs, src, [target])
        if step is None:
            continue
        # Never chase so far that we uncover the door we are guarding.
        if (
            dist_home[step[0]][step[1]] > cfg.hunt_leash
            and dist_home[step[0]][step[1]] > dist_home[src[0]][src[1]]
        ):
            continue
        key = (-d, _army(obs, src), (-src[0], -src[1]))
        if best_key is None or key > best_key:
            best_key = key
            best = (src, step)
    return _move(best[0], best[1]) if best is not None else None


def consolidate_move(obs, home: Cell):
    """Pull the most useful outlying stack one step toward the general."""
    return gather_step(obs, home, min_army=2, exclude={home})


def kill_shot(obs, memory: AegisMemory):
    """Take the enemy general right now if a neighbour of ours can."""
    target = memory.belief.enemy_general
    if target is None:
        return None
    if _mine(obs, target):
        return None
    deathtouch = obs.turn >= DEATHTOUCH_TURN
    defender = 1 if deathtouch else enemy_general_army(obs, target)
    for n in _neighbors(obs, target):
        if not _mine(obs, n):
            continue
        army = _army(obs, n)
        if (deathtouch and army >= 2) or (not deathtouch and army - 1 > defender):
            return _move(n, target)
    return None


# ------------------------------------------------------------------ counter
def counter_reason(obs, memory: AegisMemory, cfg: AegisConfig) -> str | None:
    """Why we should counterattack right now, or ``None`` to keep turtling."""
    model = memory.model
    my_total = obs.my_army
    opp_total = obs.opp_army

    edge = opp_total * cfg.counter_edge
    if model.first_contact_turn is not None and my_total >= edge:
        drop = model.opp_total_drop(window=cfg.counter_window)
        floor = max(
            cfg.counter_drop, int(cfg.counter_drop_frac * memory.peak_opp_total)
        )
        if drop >= floor:
            return "attack-spent"
        if (
            model.biggest_enemy_stack >= cfg.dead_stack_army
            and model.closest_enemy_dist_now < UNREACHABLE
            and model.biggest_enemy_stack_now * 3 <= model.biggest_enemy_stack
            and model.closest_enemy_dist <= cfg.intercept_radius
            and drop >= model.biggest_enemy_stack // 2
        ):
            return "stack-died"

    if obs.turn >= cfg.strong_turn and my_total >= opp_total * cfg.strong_ratio:
        return "overwhelming"
    if (
        obs.turn >= cfg.stall_min_turn
        and memory.free_land <= cfg.stall_free_land
        and my_total >= opp_total * cfg.stall_ratio
    ):
        return "land-exhausted"
    if obs.turn >= cfg.late_turn and my_total >= opp_total:
        return "late-game"
    return None


def counter_target(obs, memory: AegisMemory, strike: Cell | None) -> Cell | None:
    """Their general if known; else the nearest enemy cell to the strike;
    else a *remembered* fog guess (recomputing every turn oscillates)."""
    if memory.belief.enemy_general is not None:
        return memory.belief.enemy_general

    enemies = visible_enemy_tiles(obs)
    if enemies and strike is not None:
        dist = multi_bfs(obs, [strike])
        reachable = [c for c in enemies if dist[c[0]][c[1]] < UNREACHABLE]
        if reachable:
            return min(reachable, key=lambda c: (dist[c[0]][c[1]], c))

    goal = memory.counter_goal
    if goal is not None and not _mine(obs, goal):
        return goal
    guess = None
    if memory.home is not None:
        guess = mirror_tile(obs, *memory.home)
    if guess is None and enemies:
        guess = min(enemies)
    memory.counter_goal = guess
    return guess


def collection_step(obs, strike: Cell, target: Cell, cfg: AegisConfig) -> Cell | None:
    """Next cell for the strike, collecting army on the way (a route through
    our fat cells arrives far bigger at no tick cost)."""
    dist = multi_bfs(
        obs, [target], passable=lambda r, c: is_passable(obs.type_grid[r][c])
    )
    here = dist[strike[0]][strike[1]]
    if here >= UNREACHABLE:
        return None

    forward = [
        n
        for n in _neighbors(obs, strike)
        if dist[n[0]][n[1]] < here
        and (is_passable(obs.type_grid[n[0]][n[1]]) or n == target)
    ]
    if not forward:
        return None

    def pickup(cell: Cell) -> int:
        return _army(obs, cell) if _mine(obs, cell) else 0

    best = max(forward, key=lambda n: (pickup(n), (-n[0], -n[1])))

    # Worth one step sideways for a pile bigger than what lies ahead.
    detour = None
    for n in _neighbors(obs, strike):
        if n in forward or not _mine(obs, n):
            continue
        if dist[n[0]][n[1]] > here + 1:
            continue
        if _army(obs, n) >= max(cfg.detour_army, pickup(best) + 1):
            if detour is None or _army(obs, n) > _army(obs, detour):
                detour = n
    return detour if detour is not None else best


def _movable_total(obs, home: Cell | None) -> int:
    return sum(
        max(0, _army(obs, cell) - 1)
        for cell in _owned_cells(obs)
        if cell != home
    )


def _pick_strike(obs, home: Cell | None, allow_home: bool) -> Cell | None:
    candidates = [
        cell
        for cell in _owned_cells(obs)
        if _army(obs, cell) > 1 and cell != home
    ]
    if candidates:
        return max(
            candidates,
            key=lambda cell: (_army(obs, cell), (-cell[0], -cell[1])),
        )
    if allow_home and home is not None and _army(obs, home) > 1:
        return home
    return None


def strike_move(obs, memory: AegisMemory, home: Cell | None, cfg: AegisConfig):
    """One move of the committed counterattack: reinforce the designated
    strike for a bounded window, then drive it at the target."""
    my_total = obs.my_army
    opp_total = obs.opp_army

    home_army = _army(obs, home) - 1 if home is not None else 0
    door_clear = home is None or not any(
        _enemy(obs, n) for n in _neighbors(obs, home)
    )
    allow_home = opp_total <= max(cfg.raid_open_army, int(0.3 * my_total)) or (
        door_clear
        and home_army >= cfg.sortie_ratio * max(wave_estimate(obs, memory, cfg), 1)
    )

    strike = memory.strike_tile
    if strike is None or not _mine(obs, strike) or _army(obs, strike) <= 1:
        strike = _pick_strike(obs, home, allow_home)
        memory.strike_tile = strike
    if strike is None:
        return None

    target = counter_target(obs, memory, strike)
    if target is None:
        return None

    threshold = max(cfg.min_strike, int(cfg.strike_frac * _movable_total(obs, home)))
    started = memory.counter_started
    if started is not None and obs.turn - started >= cfg.gather_ticks:
        threshold = cfg.min_strike

    if _army(obs, strike) >= threshold or strike == target:
        step = None
        if cfg.collection_walk:
            step = collection_step(obs, strike, target, cfg)
        if step is None:
            step = next_step_toward(obs, strike, [target])
        if step is not None:
            memory.strike_tile = step
            return _move(strike, step)
        # Target unreachable (fully walled off): fall through to gathering.

    exclude: set[Cell] = set() if allow_home else ({home} if home is not None else set())
    exclude.add(strike)
    reinforce = gather_step(obs, strike, min_army=2, exclude=exclude)
    if reinforce is not None:
        return reinforce

    step = next_step_toward(obs, strike, [target])
    if step is not None:
        memory.strike_tile = step
        return _move(strike, step)
    return None


def enemy_distance_map(obs, memory: AegisMemory):
    """BFS distances from known enemy territory (biases expansion away)."""
    sources = visible_enemy_tiles(obs)
    if memory.belief.enemy_general is not None:
        sources.append(memory.belief.enemy_general)
    if not sources:
        return None
    dist = multi_bfs(obs, sorted(set(sources)))
    return [
        [d if d < UNREACHABLE else 999 for d in row]
        for row in dist
    ]


# ---------------------------------------------------------------- perimeter
def adjacency(obs, cell: Cell) -> tuple[int, int]:
    """``(own_neighbours, wall_neighbours)``; map edges count as walls."""
    r, c = cell
    mine = walls = 0
    for dr, dc in DIRECTIONS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            walls += 1
            continue
        if obs.type_grid[nr][nc] in (2, 5):
            walls += 1
        elif obs.owner_grid[nr][nc] == 1:
            mine += 1
    return mine, walls


def perimeter_step(obs, home: Cell | None, dist_home, reserve: int = 0,
                   avoid_dist=None, chain: bool = True):
    """One land claim that keeps the shape compact: nearest ring first, then
    the cell that plugs the most holes / hides behind the most walls."""
    if home is None:
        return None
    best, best_key = None, None
    for src in _owned_cells(obs):
        avail = _army(obs, src) - 1
        if src == home:
            avail -= reserve
        if avail < 1:
            continue
        for tgt in _neighbors(obs, src):
            tr, tc = tgt
            if obs.owner_grid[tr][tc] != 0 or obs.type_grid[tr][tc] != 1:
                continue
            if obs.army_grid[tr][tc] >= avail:
                continue
            d = dist_home[tr][tc]
            if d >= UNREACHABLE:
                continue
            mine, walls = adjacency(obs, tgt)
            away = 0
            if avoid_dist is not None:
                away = min(avoid_dist[tr][tc], AWAY_CAP)
            src_pref = _army(obs, src) if chain else -_army(obs, src)
            key = (-d, mine, walls, away, src_pref, (-tr, -tc))
            if best_key is None or key > best_key:
                best_key = key
                best = (src, tgt)
    return _move(best[0], best[1]) if best is not None else None


def raid_step(obs, home: Cell | None, dist_home, reserve: int = 0,
              max_dist: int | None = None):
    """Take an adjacent enemy cell we can afford — the cheapest growth once
    neutral land runs out. An enemy castle outranks any plain cell."""
    if home is None:
        return None
    best, best_key = None, None
    for src in _owned_cells(obs):
        avail = _army(obs, src) - 1
        if src == home:
            avail -= reserve
        if avail < 1:
            continue
        for tgt in _neighbors(obs, src):
            tr, tc = tgt
            if obs.owner_grid[tr][tc] != 2:
                continue
            if obs.army_grid[tr][tc] >= avail:
                continue
            d = dist_home[tr][tc]
            if d >= UNREACHABLE:
                continue
            if max_dist is not None and d > max_dist:
                continue
            key = (
                1 if obs.type_grid[tr][tc] == 3 else 0,
                -d,
                -obs.army_grid[tr][tc],
                -_army(obs, src),
                (-tr, -tc),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (src, tgt)
    return _move(best[0], best[1]) if best is not None else None


def free_land(obs) -> int:
    """Neutral plains left, from the exact scoreboard: fog hides *enemy*
    land too, so anything vision-based reads a conquered map as open.
    Mountains still under fog are not counted, so this over-estimates early
    — harmless, it only delays the land-exhausted counter trigger."""
    blocked = sum(
        1
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.type_grid[r][c] in (2, 5)
    )
    claimed = obs.my_land + obs.opp_land
    return max(0, obs.H * obs.W - blocked - claimed)


def march_step(obs, home: Cell | None, dist_home, reserve: int = 0,
               scout: bool = True):
    """Walk a stack out to the nearest claimable land when nothing borders
    it — staying inside (or just outside) the blob we already hold."""
    if home is None:
        return None
    owned = _owned_cells(obs)
    if not owned:
        return None
    edge = max(
        (dist_home[r][c] for r, c in owned if dist_home[r][c] < UNREACHABLE),
        default=0,
    )
    goals = [
        (r, c)
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.owner_grid[r][c] == 0
        and obs.type_grid[r][c] == 1
        and dist_home[r][c] <= edge + 2
    ]
    if not goals and scout:
        goals = [
            (r, c)
            for r in range(obs.H)
            for c in range(obs.W)
            if obs.type_grid[r][c] == 0 and dist_home[r][c] <= edge + 2
        ]
    if not goals:
        return None
    dist = multi_bfs(obs, goals)

    movers = []
    for cell in owned:
        avail = _army(obs, cell) - 1 - (reserve if cell == home else 0)
        if avail >= 1 and 0 < dist[cell[0]][cell[1]] < UNREACHABLE:
            movers.append(cell)
    if not movers:
        return None
    src = max(
        movers,
        key=lambda cell: (
            _army(obs, cell),
            -dist[cell[0]][cell[1]],
            (-cell[0], -cell[1]),
        ),
    )
    for n in _neighbors(obs, src):
        if dist[n[0]][n[1]] < dist[src[0]][src[1]] and is_passable(
            obs.type_grid[n[0]][n[1]]
        ):
            return _move(src, n)
    return None


def build_step(obs, memory: AegisMemory, home: Cell | None, dist_home,
               spare: int, cfg: AegisConfig, allow_home: bool = False):
    """Build a safe home castle (the arena's replacement for city grabs).

    Only cells within ``build_radius`` of the general are considered — a
    turtle that compounds production while turtling is the thesis intact,
    but reaching out for a distant site is exactly the thin line the turtle
    refuses to draw. The general's stack may fund the build via the site
    gather when nothing is bearing down (``allow_home``).
    """
    if home is None:
        return None
    if len(own_structures(obs)) - 1 >= cfg.max_castles:
        return None

    structures = own_structures(obs)
    site = memory.build_site
    if site is not None and (
        not _mine(obs, site)
        or obs.type_grid[site[0]][site[1]] != 1
        or dist_home[site[0]][site[1]] > cfg.build_radius
    ):
        site = memory.build_site = None
    if site is None:
        best, best_key = None, None
        for r, c in _owned_cells(obs):
            if obs.type_grid[r][c] != 1 or (r, c) == home:
                continue
            if dist_home[r][c] > cfg.build_radius:
                continue
            cost = build_cost(obs, r, c, structures)
            key = (cost, -obs.army_grid[r][c], (r, c))
            if best_key is None or key < best_key:
                best_key = key
                best = (r, c)
        site = memory.build_site = best
    if site is None:
        return None

    cost = build_cost(obs, site[0], site[1], structures)
    if spare < cost + cfg.build_reserve:
        memory.build_site = None
        return None
    if obs.army_grid[site[0]][site[1]] >= cost + cfg.build_keep:
        memory.build_site = None
        return (2, site[0], site[1], 0, 0)

    exclude: set[Cell] = {site}
    if not allow_home:
        exclude.add(home)
    return gather_step(obs, site, min_army=2, exclude=exclude)


# ------------------------------------------------------------------- policy
class AegisPolicy:
    """Stateless turtle-and-counter strategy over a caller-owned memory."""

    def __init__(self, cfg: AegisConfig | None = None) -> None:
        self.cfg = cfg or AegisConfig()

    def decide(self, obs, memory: AegisMemory):
        cfg = self.cfg
        memory.observe(obs)
        owned = _owned_cells(obs)
        if not owned:
            return PASS

        home = memory.home
        if home is None or not _mine(obs, home):
            memory.phase = "headless"
            return self._fallback(obs, owned[0]) or PASS
        dist_home = multi_bfs(obs, [home])

        # 1. Kill shot ---------------------------------------------------
        shot = kill_shot(obs, memory)
        if shot is not None:
            memory.phase = "kill"
            return shot

        if obs.turn >= 780:
            chase = _STRATEGY.chase_defence(obs, home)
            if chase is not None:
                memory.phase = "chase"
                return chase

        memory.free_land = free_land(obs)
        need = defense_need(obs, memory, cfg)
        threat = threat_distance(memory)
        econ = econ_active(obs, memory, cfg)

        keep = memory.keep
        if (
            keep is None
            or keep == home
            or not _mine(obs, keep)
            or keep not in set(_neighbors(obs, home))
        ):
            keep = keep_tile(obs, home, dist_home, cfg)
        memory.keep = keep

        # 2. Interception ------------------------------------------------
        if threat <= cfg.intercept_radius:
            move = intercept_move(obs, home, dist_home, cfg)
            if move is not None:
                memory.phase = "intercept"
                memory.abort_counter()
                return move
            move = general_intercept(obs, home, dist_home, cfg)
            if move is not None:
                memory.phase = "sortie"
                memory.abort_counter()
                return move

        # 3. Man the door ------------------------------------------------
        move = man_the_door(obs, home, keep, need)
        if move is not None:
            memory.phase = "door"
            memory.abort_counter()
            return move

        # 3a. Emergency garrison -----------------------------------------
        if threat <= cfg.imminent_radius and _army(obs, home) < need:
            move = consolidate_move(obs, home)
            if move is not None:
                memory.phase = "garrison"
                memory.abort_counter()
                return move

        # 3b. Hunt the raider --------------------------------------------
        if threat <= cfg.intercept_radius:
            move = hunt_move(obs, home, dist_home, cfg)
            if move is not None:
                memory.phase = "hunt"
                return move

        # 4. Home deficit ------------------------------------------------
        horizon = max(1, min(threat, cfg.ring_horizon))
        deficit = need - reachable_defense(obs, home, dist_home, horizon)
        if threat <= cfg.pull_radius and deficit > 0:
            rally = keep_tile(obs, home, dist_home, cfg)
            memory.keep = rally
            move = gather_step(obs, rally, min_army=2, exclude={home, rally})
            if move is None:
                move = consolidate_move(obs, home)
            if move is not None:
                memory.phase = "pull-home"
                memory.abort_counter()
                return move

        # 5. Counterattack -----------------------------------------------
        if memory.counter_active(obs.turn):
            started = memory.counter_start_tiles
            now = memory.model.my_land[-1] if memory.model.my_land else 0
            if started and now < started * (1 - cfg.counter_abort_loss):
                memory.abort_counter()
            else:
                move = strike_move(obs, memory, home, cfg)
                if move is not None:
                    memory.phase = "counter"
                    return move
        reason = counter_reason(obs, memory, cfg)
        if reason is not None:
            memory.start_counter(obs.turn, cfg.counter_hold)
            move = strike_move(obs, memory, home, cfg)
            if move is not None:
                memory.phase = f"counter:{reason}"
                return move

        # 6. Castles ------------------------------------------------------
        if econ and memory.econ_since is None:
            memory.econ_since = obs.turn
        if econ or cfg.build_anytime:
            spare = max(0, obs.my_army - need)
            move = build_step(
                obs, memory, home, dist_home, spare, cfg,
                allow_home=threat > cfg.pull_radius,
            )
            if move is not None:
                memory.phase = "build"
                return move

        # 7. Perimeter growth --------------------------------------------
        lock = (
            _army(obs, home)
            if need > cfg.min_defense and threat <= cfg.lock_radius
            else need
        )

        avoid = enemy_distance_map(obs, memory)
        move = perimeter_step(
            obs, home, dist_home, lock, avoid, chain=cfg.chain_expand
        )
        if move is not None:
            memory.phase = "econ-expand" if econ else "turtle-expand"
            return move

        move = raid_step(obs, home, dist_home, lock, cfg.raid_radius)
        if move is not None:
            memory.phase = "raid"
            return move

        move = march_step(obs, home, dist_home, lock, scout=memory.free_land > 0)
        if move is not None:
            memory.phase = "march"
            return move

        if econ:
            move = expansion_step(
                obs, reserve={home} if need > cfg.min_defense else None
            )
            if move is not None:
                memory.phase = "econ-reach"
                return move

        # 8. Fallback -----------------------------------------------------
        memory.phase = "hold"
        return self._fallback(obs, home) or PASS

    def _fallback(self, obs, home: Cell):
        move = gather_step(obs, home, min_army=2, exclude={home})
        if move is not None:
            return move
        for cell in _owned_cells(obs):
            if cell == home or _army(obs, cell) < 2:
                continue
            for n in _neighbors(obs, cell):
                if _mine(obs, n):
                    return _move(cell, n)
        for cell in _owned_cells(obs):
            if cell == home or _army(obs, cell) < 2:
                continue
            for n in _neighbors(obs, cell):
                if is_passable(obs.type_grid[n[0]][n[1]]):
                    return _move(cell, n)
        # Nothing but the general left: passing beats emptying it.
        return None


class AegisCore:
    """Stateful wrapper around :class:`AegisPolicy`, composable by proteus."""

    def __init__(self, player_id: int, H: int, W: int,
                 cfg: AegisConfig | None = None,
                 model: OpponentModel | None = None) -> None:
        self.player_id = player_id
        self.H = H
        self.W = W
        self.policy = AegisPolicy(cfg)
        self.memory = AegisMemory(model=model or OpponentModel())

    @property
    def phase(self) -> str:
        return self.memory.phase

    def observe(self, obs) -> None:
        self.memory.observe(obs)

    def decide(self, obs):
        return self.policy.decide(obs, self.memory)


class Agent:
    """Arena entrypoint."""

    def __init__(self, player_id: int, H: int, W: int) -> None:
        self._core = AegisCore(player_id, H, W)

    def act(self, obs):
        return self._core.decide(obs)
