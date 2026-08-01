"""The endgame core: play RULES.md §07 as its own game, from turn 800.

Deathtouch is **not new here.** `StrategyConfig.deathtouch_turn = 800` and a
`DEATHTOUCH_TURN = 800` constant already exist in blitz, aegis, metro, garrison,
late_rush and splitter, and every one of them uses it the same way: as one extra
clause in a finishing check — "past 800, a 2-army neighbour of the enemy general
also wins". That clause is correct and this core keeps it.

What none of them do is change *how they play* because the rule exists. From
turn 800 the game the board is playing is a different game, and three of its
consequences contradict what a pre-800 core is optimising:

1. **Army on our own general buys nothing.** Below 800 the general's stack is
   its defence (§05: the attacker needs strictly more). At 800 one unit is
   lethal, so a 60-army general dies to a 2-stack exactly as fast as a 2-army
   one. Every army banked at home is now dead weight.

2. **The only defence is a chase, and it must come from a third tile.** §07:
   capture the attack's *source* cell that same turn and the touch never
   executes, because the engine re-tests `owns_source` at execution
   (generals/modifiers/deathtouch.py, `_executes_onto_general`). Two things
   follow that are easy to get wrong:

   - Reducing the source is not enough. The attacker moves all-but-one, so a
     source left holding 2 still sends 1 and still touches. The chase has to
     *take* the cell: strictly more army than sits on it.
   - The counter cannot come from the general. General-versus-source is a
     mutual chase, so §02 falls through to smaller-army-first; the general
     moves first, is too small to strip the source, and the touch lands anyway.
     The modifier's own docstring spells this out.

   A tile that can chase a threat standing next to our general is adjacent to
   that threat and is not the general — so it sits at distance 2 from the
   general. **The endgame garrison belongs at distance 2, not at home.** That
   is the whole positional consequence of the rule and it is why this is a
   core rather than a clause.

3. **Reach beats size.** A 2-stack that arrives wins; a 40-stack that is still
   walking does not. But a 2-stack moves one unit, and §05 needs strictly more
   than the defender, so it can cross our own land and empty neutral land and
   nothing else. Routing a touch is a path problem over friendly ground, not
   an army-accumulation problem — the opposite of what blitz's wave cycle and
   boom's strike-fist are built to do.

What this core cannot buy
-------------------------
Against cm_hunter it is very close to irrelevant: 1 game in 50 of the round5
sample reached turn 800 at all, and in the 300-game baseline measured for this
effort the median win came at turn ~384. Its value is entirely in the
draw-heavy matchups — fog_scout (34.0% draws), army_convey (15.8%), metro
(13.0%), aegis (11.8%) — where both sides are alive at 1200 and the game ends
on the clock. It is measured there, as a draw-conversion result, and it is not
evidence about the cm_hunter target.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from _common.oppmodel import OpponentModel
from _common.strategy_common import (
    DIRECTIONS,
    PASS,
    StrategyContext,
    direction_from_to,
    is_passable,
    locate_own_general,
)
from _common.tactics import UNREACHABLE, expansion_step, multi_bfs

Cell = tuple[int, int]

_STRATEGY = StrategyContext()

PHASE_TOUCH = "touch"
PHASE_CHASE = "chase"
PHASE_MARCH = "march"
PHASE_GARRISON = "garrison"
PHASE_HUNT = "hunt"
PHASE_IDLE = "dt_idle"


@dataclass(frozen=True)
class DeathtouchConfig:
    #: Turn the rule switches on (RULES.md §07). Also this core's start turn.
    touch_turn: int = 800

    #: Distance from our general within which an enemy stack is a live touch
    #: threat. 1 is "can touch this turn"; 2 is "can touch next turn", and
    #: pre-positioning against it is worth a move because a chase tile cannot
    #: be conjured on the turn it is needed.
    threat_radius: int = 2

    #: Army a distance-2 garrison tile wants, over and above the largest enemy
    #: stack seen near home. The chase must *capture* the source, so it needs
    #: strictly more army than the source holds, plus the one it leaves behind.
    garrison_margin: int = 2

    #: Never spend so much on the march that no chase is possible. At least
    #: this many owned distance-2 tiles must stay above the garrison target
    #: before a march move is allowed to consume the best one.
    garrison_tiles: int = 1

    #: Marchers to keep walking at once. Reach beats size, so several thin
    #: threats on different approaches beat one fat one — the defender can
    #: only chase one source a turn.
    march_streams: int = 3

    #: Below this many turns left, stop banking anything at all: an unconverted
    #: won board scores the same as a lost one (§07's 1200-turn hard draw).
    all_in_from: int = 1100


@dataclass
class DeathtouchMemory:
    model: OpponentModel = field(default_factory=OpponentModel)
    belief: object = field(default_factory=_STRATEGY.BeliefState)
    my_general: Cell | None = None
    phase: str = PHASE_IDLE
    last_turn: int = -1

    def observe(self, obs) -> None:
        if obs.turn == self.last_turn:
            return
        self.last_turn = obs.turn
        self.model.update(obs)
        if self.my_general is None:
            self.my_general = locate_own_general(obs)
        self.belief.update(obs, self.my_general)


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


# --------------------------------------------------------------------- 1. win
def touch_move(obs, mem: DeathtouchMemory):
    """Any owned neighbour of the enemy general with 2+ army wins now (§07).

    Prefer the *smallest* qualifying stack. The move is lethal at any size, so
    spending the big one is pure waste if the touch is chased off — and the
    engine's chase test looks at the source cell, not at what left it.
    """
    target = mem.belief.enemy_general
    if target is None or obs.owner_grid[target[0]][target[1]] == 1:
        return None
    best, best_army = None, None
    for cell in _neighbors(obs, target):
        if obs.owner_grid[cell[0]][cell[1]] != 1:
            continue
        army = obs.army_grid[cell[0]][cell[1]]
        if army < 2:
            continue
        if best_army is None or army < best_army:
            best_army, best = army, cell
    return _move(best, target) if best else None


# ------------------------------------------------------------------ 2. defend
def door_threats(obs, general: Cell) -> list[tuple[Cell, int]]:
    """Enemy cells *adjacent* to our general — the ones that touch this turn.

    Four cells, checked directly: a BFS to answer "distance <= 1" would scan
    the board to learn what four neighbour lookups already say, and this runs
    every turn from 800 to 1200.
    """
    out = [
        (cell, obs.army_grid[cell[0]][cell[1]])
        for cell in _neighbors(obs, general)
        if obs.owner_grid[cell[0]][cell[1]] == 2 and obs.army_grid[cell[0]][cell[1]] >= 2
    ]
    out.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return out


def near_threats(obs, general: Cell, radius: int) -> list[tuple[Cell, int]]:
    """Enemy stacks within `radius` BFS steps — the ones that touch *soon*."""
    dist = multi_bfs(obs, [general])
    out = [
        ((r, c), obs.army_grid[r][c])
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.owner_grid[r][c] == 2
        and obs.army_grid[r][c] >= 2
        and dist[r][c] <= radius
    ]
    out.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return out


def chase_move(obs, mem: DeathtouchMemory):
    """Capture the source of a touch that could execute this turn (§07).

    Three constraints, all of them the rule's and none of them heuristics:

    - the chasing tile must not be the general (mutual chase, §02 hands it to
      the smaller army, and the general loses that);
    - it must *capture*, so `army - 1 > source army`; leaving the source with
      1 army is not enough, it still moves nothing — leaving it with 2 is
      fatal, it still moves one;
    - among candidates, spend the smallest sufficient stack, so the rest of
      the garrison survives for the next turn's threat.
    """
    general = mem.my_general
    if general is None:
        return None
    for threat, army in door_threats(obs, general):
        best, best_army = None, None
        for cell in _neighbors(obs, threat):
            if cell == general or obs.owner_grid[cell[0]][cell[1]] != 1:
                continue
            mine = obs.army_grid[cell[0]][cell[1]]
            if mine - 1 <= army:
                continue
            if best_army is None or mine < best_army:
                best_army, best = mine, cell
        if best is not None:
            return _move(best, threat)
    return None


# ---------------------------------------------------------------- 3. garrison
def chase_ring(obs, general: Cell) -> list[Cell]:
    """Owned cells from which a threat *at* our general's door can be chased.

    Every orthogonal neighbour of a neighbour of the general, minus the general
    itself. Geometrically that is the distance-2 shell plus nothing — which is
    the positional content of §07's chase clause.
    """
    ring: list[Cell] = []
    seen = {general}
    for door in _neighbors(obs, general):
        if not is_passable(obs.type_grid[door[0]][door[1]]):
            continue
        for cell in _neighbors(obs, door):
            if cell in seen or cell == general:
                continue
            seen.add(cell)
            if obs.owner_grid[cell[0]][cell[1]] == 1:
                ring.append(cell)
    return ring


def garrison_target(obs, mem: DeathtouchMemory, cfg: DeathtouchConfig,
                    threats: list[tuple[Cell, int]]) -> int:
    """How much a ring tile needs to capture the source that is actually
    coming — the largest stack currently inside `threat_radius + 2`, not the
    largest ever seen anywhere. Sizing off a latched maximum would put the
    garrison in a permanent deficit and starve the march forever."""
    biggest = max((army for _, army in threats), default=0)
    return max(3, biggest + cfg.garrison_margin)


def garrison_move(obs, mem: DeathtouchMemory, cfg: DeathtouchConfig,
                  threats: list[tuple[Cell, int]]):
    """Top up the chase ring — the only defensive spending §07 rewards.

    Runs **only when a threat is actually inbound**. With nothing near home
    there is no source to chase, so army spent here is army not walking at
    their general, and an unconverted board draws.
    """
    general = mem.my_general
    if general is None or not threats:
        return None
    ring = chase_ring(obs, general)
    if not ring:
        return None
    need = garrison_target(obs, mem, cfg, threats)
    if max(obs.army_grid[r][c] for r, c in ring) >= need:
        return None

    tdist = multi_bfs(obs, [threats[0][0]])
    target = min(ring, key=lambda cell: (tdist[cell[0]][cell[1]], cell))

    # The general is a legal donor here and nowhere else: past 800 its own
    # stack defends nothing, so draining it into the ring is strictly better
    # than holding it (see this module's header, point 1).
    return _gather_to(obs, target, exclude={target})


def _gather_to(obs, target: Cell, exclude: set[Cell]):
    """One step of the best-value reachable owned stack toward `target`."""
    dist = multi_bfs(
        obs,
        [target],
        passable=lambda r, c: obs.owner_grid[r][c] == 1 or (r, c) == target,
    )
    best, best_key = None, None
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1 or (r, c) in exclude:
                continue
            army = obs.army_grid[r][c]
            if army < 2 or dist[r][c] >= UNREACHABLE or dist[r][c] == 0:
                continue
            key = (army - 1) / (dist[r][c] + 1.0)
            if best_key is None or key > best_key:
                best_key, best = key, (r, c)
    if best is None:
        return None
    br, bc = best
    for dr, dc in DIRECTIONS:
        nr, nc = br + dr, bc + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            continue
        if dist[nr][nc] < dist[br][bc]:
            return _move(best, (nr, nc))
    return None


# ------------------------------------------------------------------- 4. march
def touch_field(obs, target: Cell) -> list[list[int]]:
    """Steps to `target` over ground a *2-stack* can actually walk.

    A 2-stack moves one unit, and §05 needs strictly more than the defender,
    so it can cross our own cells and empty neutral cells and nothing else —
    every enemy cell holds at least one army. The target itself is always
    enterable, because §07 makes the general's garrison irrelevant.
    """
    def walkable(r: int, c: int) -> bool:
        if (r, c) == target:
            return True
        if not is_passable(obs.type_grid[r][c]):
            return False
        owner = obs.owner_grid[r][c]
        if owner == 1:
            return True
        # Neutral plain or unexplored fog: passable if nothing is standing on
        # it. Fog reports army 0 (competition/protocol.py), so this treats
        # fog as empty — optimistic, and the reason a march can stall.
        return owner == 0 and obs.army_grid[r][c] == 0

    return multi_bfs(obs, [target], passable=walkable)


def march_move(obs, mem: DeathtouchMemory, cfg: DeathtouchConfig, reserved: set[Cell]):
    """Walk the *closest* 2+ stack at the enemy general.

    Closest, not biggest: the touch is lethal at 2, so the only thing that
    separates a winning stack from a losing one is arrival time.
    """
    target = mem.belief.enemy_general
    if target is None:
        return None
    field_ = touch_field(obs, target)
    best, best_key = None, None
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1 or (r, c) in reserved:
                continue
            if obs.army_grid[r][c] < 2:
                continue
            d = field_[r][c]
            if d >= UNREACHABLE or d == 0:
                continue
            key = (d, -obs.army_grid[r][c], (r, c))
            if best_key is None or key < best_key:
                best_key, best = key, (r, c)
    if best is None:
        # No clean lane: fall back to a plain distance field and punch through
        # with the biggest stack we have.
        return _punch(obs, target, reserved)
    br, bc = best
    for dr, dc in DIRECTIONS:
        nr, nc = br + dr, bc + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            continue
        if field_[nr][nc] < field_[br][bc]:
            return _move(best, (nr, nc))
    return None


def _punch(obs, target: Cell, reserved: set[Cell]):
    """No 2-stack lane exists: send the biggest stack down the raw distance."""
    dist = multi_bfs(obs, [target])
    best, best_key = None, None
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1 or (r, c) in reserved:
                continue
            army = obs.army_grid[r][c]
            if army < 2 or dist[r][c] >= UNREACHABLE or dist[r][c] == 0:
                continue
            key = (-army, dist[r][c], (r, c))
            if best_key is None or key < best_key:
                best_key, best = key, (r, c)
    if best is None:
        return None
    br, bc = best
    attacking = obs.army_grid[br][bc] - 1
    for dr, dc in DIRECTIONS:
        nr, nc = br + dr, bc + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            continue
        if dist[nr][nc] >= dist[br][bc] or not is_passable(obs.type_grid[nr][nc]):
            continue
        if (nr, nc) != target and obs.owner_grid[nr][nc] == 2 \
                and obs.army_grid[nr][nc] >= attacking:
            continue
        return _move(best, (nr, nc))
    return None


# --------------------------------------------------------------------- 5. hunt
def hunt_move(obs, mem: DeathtouchMemory):
    """The general is fogged: walk at the nearest cell that could hold it.

    `BeliefState.candidates` is the set of passable cells at least
    `min_general_distance` from our own general that we have never seen, minus
    everything we have. Reducing it is the only way a touch ever gets a target.
    """
    candidates = getattr(mem.belief, "candidates", None)
    if not candidates:
        return None
    dist = multi_bfs(obs, sorted(candidates))
    best, best_key = None, None
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] < 2:
                continue
            d = dist[r][c]
            if d >= UNREACHABLE or d == 0:
                continue
            key = (d, -obs.army_grid[r][c], (r, c))
            if best_key is None or key < best_key:
                best_key, best = key, (r, c)
    if best is None:
        return None
    br, bc = best
    for dr, dc in DIRECTIONS:
        nr, nc = br + dr, bc + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            continue
        if dist[nr][nc] < dist[br][bc] and is_passable(obs.type_grid[nr][nc]):
            return _move(best, (nr, nc))
    return None


# --------------------------------------------------------------------- ladder
def deathtouch_move(obs, mem: DeathtouchMemory, cfg: DeathtouchConfig):
    mem.observe(obs)
    general = mem.my_general

    win = touch_move(obs, mem)
    if win is not None:
        mem.phase = PHASE_TOUCH
        return win

    chase = chase_move(obs, mem)
    if chase is not None:
        mem.phase = PHASE_CHASE
        return chase

    reserved: set[Cell] = set()
    if general is not None and obs.turn < cfg.all_in_from:
        threats = near_threats(obs, general, cfg.threat_radius + 2)
        if threats:
            # Hold the fullest ring tiles back from the march, so a chase is
            # still possible next turn. Past `all_in_from` nothing is held: an
            # unconverted board and a lost one score the same (§07).
            ring = chase_ring(obs, general)
            need = garrison_target(obs, mem, cfg, threats)
            keep = sorted(
                (c for c in ring if obs.army_grid[c[0]][c[1]] >= need),
                key=lambda c: -obs.army_grid[c[0]][c[1]],
            )
            reserved.update(keep[: cfg.garrison_tiles])

            top_up = garrison_move(obs, mem, cfg, threats)
            if top_up is not None:
                mem.phase = PHASE_GARRISON
                return top_up

    march = march_move(obs, mem, cfg, reserved)
    if march is not None:
        mem.phase = PHASE_MARCH
        return march

    hunt = hunt_move(obs, mem)
    if hunt is not None:
        mem.phase = PHASE_HUNT
        return hunt

    mem.phase = PHASE_IDLE
    return expansion_step(obs) or PASS


class DeathtouchCore:
    """Stateful wrapper, shaped like BlitzCore/BoomCore so the switcher can
    hold all three the same way."""

    def __init__(self, player_id: int, H: int, W: int,
                 config: DeathtouchConfig | None = None,
                 model: OpponentModel | None = None) -> None:
        self.player_id = player_id
        self.H = H
        self.W = W
        self.config = config or DeathtouchConfig()
        self.memory = DeathtouchMemory(model=model or OpponentModel())

    @property
    def phase(self) -> str:
        return self.memory.phase

    def observe(self, obs) -> None:
        self.memory.observe(obs)

    def decide(self, obs):
        return deathtouch_move(obs, self.memory, self.config)
