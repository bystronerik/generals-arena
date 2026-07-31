"""Metro — castle network + pressure (spec: docs/research/strategies/metro.md).

Grid-native redesign of the generals-bot Metro (city control) strategy. The
source thesis survives the ruleset change: *army spent on land is spent
once; army spent on production keeps paying.* The arena has no neutral
cities to capture, so the production programme **builds** castles instead
(``2 r c 0 0``), and "cash in" retargets **enemy** castles — capturing one
is a double production swing.

Priority ladder each turn:

1. **Killing blow** — take the enemy general if adjacent and affordable.
2. **Home defence** — an enemy stack near our general outranks everything.
3. **Cash in** — capture a visible enemy castle we can already afford.
4. **Claim / raid** — free land while opening or losing the land race.
5. **Expansion beat** — one turn in ``EXPAND_EVERY`` grows the base.
6. **Castle programme** — walk the main stack onto the chosen site, build.
7. **Pressure** — from turn 140 there is always a wave in the field.
8. **Expand / consolidate** fallbacks.
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


@dataclass(frozen=True)
class MetroParams:
    """Tunables. Source-tuned values carry over as hypotheses; the castle
    knobs replace the source's neutral-city knobs."""

    # --- opening -------------------------------------------------------
    open_until: int = 50
    """Turns of (nearly) pure land grabbing before the castle programme."""
    expand_every: int = 3
    """During the programme, spend one turn in N on plain expansion."""
    claim_parity: float = 0.95
    """While our land is below this multiple of theirs, free claims outrank
    the programme — a rout on land beats any production edge."""

    # --- castle programme ----------------------------------------------
    max_castles: int = 3
    """Stop opening new build projects once we hold this many (+ general =
    four producers)."""
    build_start: int = 60
    """First turn a build project may open."""
    forward_weight: float = 1.0
    """Army-equivalents charged per step of distance from the enemy anchor
    when scoring sites — pulls castles forward, where their production
    feeds the waves."""
    front_safety: int = 6
    """Never build within this BFS distance of a visible enemy cell."""
    build_keep: int = 3
    """Army banked beyond the price, left as the new castle's garrison."""
    walk_min: int = 6
    """Smallest stack that counts as "the main stack" for a collection walk."""
    give_up_ticks: int = 80
    """Abandon a build project that has not funded itself in this long."""

    # --- defence -------------------------------------------------------
    defend_radius: int = 8
    general_free_dist: int = 8
    """Below this enemy distance the general is locked as a source."""
    min_bank: int = 8
    """After the opening, only release the general once it holds this much."""
    castle_garrison: int = 8
    """Keep this much army on an owned castle before spending it elsewhere."""
    castle_safe_dist: int = 6
    """An owned castle with an enemy this close is a front-line bastion: its
    army stays put (a move sends all-but-one, so draining it flips it)."""

    # --- pressure ------------------------------------------------------
    pressure_from: int = 140
    """Turn from which we always keep a wave in the field. Army standing on
    a castle is worth nothing; only spent army takes land or generals."""
    pressure_ratio: float = 1.15
    strong_ratio: float = 1.6
    min_push: int = 18
    max_push_share: float = 0.55
    """Never wait for a spearhead bigger than this share of our whole army —
    one monster enemy stack must not freeze us into permanent saving."""
    wave_takeover: float = 1.6
    """How much bigger another stack must be before it takes the wave over."""


@dataclass
class MetroMemory:
    """Everything Metro remembers between turns."""

    opp: OpponentModel = field(default_factory=OpponentModel)
    belief: object = field(default_factory=_STRATEGY.BeliefState)
    my_general: Cell | None = None

    #: Current castle build site and the turn we committed to it.
    build_site: Cell | None = None
    site_since: int = 0

    #: Wave state (see the source's wave_tile/pushing rationale: without the
    #: latch, a regenerating home castle hijacks the advance every cycle).
    pushing: bool = False
    wave_tile: Cell | None = None

    castles_built: int = 0
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

    def set_site(self, site: Cell | None, turn: int) -> None:
        if site != self.build_site:
            self.build_site = site
            self.site_since = turn

    def waited(self, turn: int) -> int:
        if self.build_site is None:
            return 0
        return max(0, turn - self.site_since)


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


def owned_castles(obs) -> list[Cell]:
    return [
        (r, c)
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.owner_grid[r][c] == 1 and obs.type_grid[r][c] == 3
    ]


def enemy_castles(obs) -> list[Cell]:
    return [
        (r, c)
        for r in range(obs.H)
        for c in range(obs.W)
        if obs.owner_grid[r][c] == 2 and obs.type_grid[r][c] == 3
    ]


def _transit(obs):
    """Routing for moving our own army: never through enemy land (crossing
    it burns the pile the walk is supposed to deliver)."""

    def ok(r: int, c: int) -> bool:
        return is_passable(obs.type_grid[r][c]) and obs.owner_grid[r][c] != 2

    return ok


class MetroStrategy:
    """Metro's decision logic: ``(obs, memory) -> action``. Stateless."""

    def __init__(self, params: MetroParams | None = None) -> None:
        self.p = params or MetroParams()

    # ------------------------------------------------------------------
    def decide(self, obs, memory: MetroMemory):
        memory.observe(obs)
        move = self._decide_raw(obs, memory)
        if move is not None and self._legal(obs, move):
            return move
        fallback = self._any_move(obs, memory.my_general)
        if fallback is not None and self._legal(obs, fallback):
            return fallback
        return PASS

    def _decide_raw(self, obs, memory: MetroMemory):
        gen = memory.my_general

        move = self._killing_blow(obs, memory)
        if move is not None:
            return move

        if obs.turn >= 780 and gen is not None:
            chase = _STRATEGY.chase_defence(obs, gen)
            if chase is not None:
                return chase

        move = self._defend(obs, memory, gen)
        if move is not None:
            return move

        # Cash in: a visible enemy castle we can already afford is a double
        # production swing (their +1/2-turns becomes ours).
        move = self._cash_in(obs, memory, gen)
        if move is not None:
            return move

        opening = obs.turn < self.p.open_until
        if opening or self._losing_the_land_race(obs):
            move = self._claim(obs, memory, gen)
            if move is not None:
                return move
            move = self._raid(obs, memory, gen)
            if move is not None:
                return move

        if opening or obs.turn % self.p.expand_every == 0:
            move = self._expand(obs, memory, gen)
            if move is not None:
                return move

        move = self._castle_programme(obs, memory, gen)
        if move is not None:
            return move

        move = self._pressure(obs, memory, gen)
        if move is not None:
            return move

        move = self._expand(obs, memory, gen)
        if move is not None:
            return move

        return self._consolidate(obs, memory, gen)

    # ------------------------------------------------------------------
    # 1. Killing blow
    # ------------------------------------------------------------------
    def _killing_blow(self, obs, memory: MetroMemory):
        """The general is allowed as the source here: capturing theirs ends
        the game on the spot."""
        target = memory.belief.enemy_general
        if target is None:
            return None
        tr, tc = target
        if obs.owner_grid[tr][tc] == 1:
            return None
        deathtouch = obs.turn >= DEATHTOUCH_TURN
        defender = 1 if deathtouch else enemy_general_army(obs, target)
        best, best_army = None, -1
        for n in _neighbors(obs, target):
            if obs.owner_grid[n[0]][n[1]] != 1:
                continue
            army = obs.army_grid[n[0]][n[1]]
            ok = army >= 2 if deathtouch else army - 1 > defender
            if ok and army > best_army:
                best, best_army = n, army
        if best is None:
            return None
        return _move(best, target)

    # ------------------------------------------------------------------
    # 2. Defence
    # ------------------------------------------------------------------
    def _defend(self, obs, memory: MetroMemory, gen: Cell | None):
        """Metro is cash-poor right after a build, so the threat test is
        deliberately pessimistic: biggest enemy stack in range vs the
        general's own garrison, ignoring attrition on the way in."""
        if gen is None:
            return None
        enemies = visible_enemy_tiles(obs)
        if not enemies:
            return None
        dist = multi_bfs(obs, [gen])
        near = [c for c in enemies if dist[c[0]][c[1]] <= self.p.defend_radius]
        if not near:
            return None

        threat_cell = max(
            near,
            key=lambda c: (obs.army_grid[c[0]][c[1]], -dist[c[0]][c[1]]),
        )
        if dist[threat_cell[0]][threat_cell[1]] <= 2:
            kill = self._capture_cell(obs, threat_cell, avoid={gen})
            if kill is not None:
                return kill

        threat = obs.army_grid[threat_cell[0]][threat_cell[1]]
        if obs.army_grid[gen[0]][gen[1]] >= threat:
            return None  # the general already out-guns anything in range
        return self._gather(obs, gen, exclude=self._garrisoned(obs) | {gen})

    def _gather(self, obs, target: Cell, exclude: set[Cell] | None = None):
        """Funnel army to ``target``, preferring routes over friendly ground."""
        move = gather_step(
            obs, target, min_army=2, exclude=exclude, passable=_transit(obs)
        )
        if move is not None:
            return move
        return gather_step(obs, target, min_army=2, exclude=exclude)

    def _capture_cell(self, obs, cell: Cell, avoid: set[Cell] | None = None):
        """Take ``cell`` from an adjacent owned cell that can afford it."""
        avoid = avoid or set()
        defender = obs.army_grid[cell[0]][cell[1]]
        best, best_army = None, -1
        for n in _neighbors(obs, cell):
            if obs.owner_grid[n[0]][n[1]] != 1 or n in avoid:
                continue
            army = obs.army_grid[n[0]][n[1]]
            if army - 1 > defender and army > best_army:
                best, best_army = n, army
        return _move(best, cell) if best is not None else None

    # ------------------------------------------------------------------
    # 3. Cash in — enemy castles
    # ------------------------------------------------------------------
    def _cash_in(self, obs, memory: MetroMemory, gen: Cell | None):
        avoid: set[Cell] = set()
        if gen is not None and not self._general_usable(obs, memory, gen):
            avoid.add(gen)
        affordable = []
        for c in enemy_castles(obs):
            move = self._capture_cell(obs, c, avoid=avoid)
            if move is not None:
                affordable.append((obs.army_grid[c[0]][c[1]], c, move))
        if affordable:
            affordable.sort()
            return affordable[0][2]
        return None

    # ------------------------------------------------------------------
    # 4/5. Expansion
    # ------------------------------------------------------------------
    def _expand(self, obs, memory: MetroMemory, gen: Cell | None):
        return expansion_step(obs, reserve=self._reserve(obs, memory, gen))

    def _claim(self, obs, memory: MetroMemory, gen: Cell | None):
        """Expansion, but *only* when it takes a neutral cell this turn."""
        move = self._expand(obs, memory, gen)
        if move is None or move[0] != 0:
            return None
        dr, dc = DIRECTIONS[move[3]]
        er, ec = move[1] + dr, move[2] + dc
        if obs.owner_grid[er][ec] != 0 or obs.type_grid[er][ec] != 1:
            return None
        if obs.army_grid[move[1]][move[2]] - 1 <= obs.army_grid[er][ec]:
            return None
        return move

    def _raid(self, obs, memory: MetroMemory, gen: Cell | None):
        """Take a cheap enemy border cell (a two-land swing for almost
        nothing). Castles are excluded: those are the cash-in's business."""
        avoid: set[Cell] = set()
        if gen is not None and not self._general_usable(obs, memory, gen):
            avoid.add(gen)
        best, best_key = None, None
        for src in _owned_cells(obs):
            if src in avoid:
                continue
            attacking = obs.army_grid[src[0]][src[1]] - 1
            if attacking < 1:
                continue
            for tgt in _neighbors(obs, src):
                tr, tc = tgt
                if obs.owner_grid[tr][tc] != 2 or obs.type_grid[tr][tc] == 3:
                    continue
                defender = obs.army_grid[tr][tc]
                if attacking <= defender:
                    continue
                key = (defender, obs.army_grid[src[0]][src[1]])
                if best_key is None or key < best_key:
                    best_key = key
                    best = (src, tgt)
        return _move(best[0], best[1]) if best is not None else None

    def _garrisoned(self, obs) -> set[Cell]:
        """Owned castles near the enemy: locked as sources (bastions)."""
        castles = owned_castles(obs)
        if not castles:
            return set()
        enemies = visible_enemy_tiles(obs)
        if not enemies:
            return set()
        dist = multi_bfs(obs, enemies)
        return {
            c for c in castles if dist[c[0]][c[1]] <= self.p.castle_safe_dist
        }

    def _losing_the_land_race(self, obs) -> bool:
        return obs.my_land < obs.opp_land * self.p.claim_parity

    def _reserve(self, obs, memory: MetroMemory, gen: Cell | None) -> set[Cell]:
        """Cells expansion must not raid: general, war chests, bastions."""
        reserve: set[Cell] = set()
        if gen is not None and not self._general_usable(
            obs, memory, gen, for_expansion=True
        ):
            reserve.add(gen)
        site = memory.build_site
        if site is not None and obs.turn >= self.p.open_until:
            if (
                site != gen
                and obs.owner_grid[site[0]][site[1]] == 1
                and obs.army_grid[site[0]][site[1]] > 5
            ):
                reserve.add(site)
        reserve |= self._garrisoned(obs)
        for c in owned_castles(obs):
            if obs.army_grid[c[0]][c[1]] > self.p.castle_garrison:
                reserve.add(c)
        return reserve

    def _general_usable(self, obs, memory: MetroMemory, gen: Cell | None,
                        for_expansion: bool = False) -> bool:
        """Whether we may move army *off* the general this turn."""
        if gen is None or obs.owner_grid[gen[0]][gen[1]] != 1:
            return False
        if obs.army_grid[gen[0]][gen[1]] < 2:
            return False
        if memory.opp.closest_enemy_dist_now <= self.p.general_free_dist:
            return False
        if for_expansion:
            return True
        return obs.army_grid[gen[0]][gen[1]] >= self.p.min_bank

    # ------------------------------------------------------------------
    # 6. Castle programme (the redesign: build, do not capture)
    # ------------------------------------------------------------------
    def _enemy_anchor(self, obs, memory: MetroMemory) -> Cell | None:
        if memory.belief.enemy_general is not None:
            return memory.belief.enemy_general
        enemies = visible_enemy_tiles(obs)
        if enemies:
            return min(enemies)
        if memory.my_general is not None:
            return mirror_tile(obs, *memory.my_general)
        return None

    def _choose_site(self, obs, memory: MetroMemory, gen: Cell | None) -> Cell | None:
        """Pick the build site: an owned plain cell, cheap (the crowding
        surcharge makes spacing emerge naturally), pulled forward toward the
        enemy anchor, never on the front line."""
        anchor = self._enemy_anchor(obs, memory)
        forward = multi_bfs(obs, [anchor]) if anchor is not None else None
        enemies = visible_enemy_tiles(obs)
        danger = multi_bfs(obs, enemies) if enemies else None
        structures = own_structures(obs)
        best, best_key = None, None
        for r, c in _owned_cells(obs):
            if obs.type_grid[r][c] != 1 or (r, c) == gen:
                continue
            if danger is not None and danger[r][c] <= self.p.front_safety:
                continue
            fwd = 0
            if forward is not None and forward[r][c] < UNREACHABLE:
                fwd = forward[r][c]
            score = build_cost(obs, r, c, structures) + self.p.forward_weight * fwd
            key = (score, (r, c))
            if best_key is None or key < best_key:
                best_key = key
                best = (r, c)
        return best

    def _castle_programme(self, obs, memory: MetroMemory, gen: Cell | None):
        """Advance the build project: pick a site, walk the main stack onto
        it (collecting surplus en route), build once funded."""
        if obs.turn < self.p.build_start:
            return None
        if len(owned_castles(obs)) >= self.p.max_castles:
            memory.set_site(None, obs.turn)
            return None

        site = memory.build_site
        if site is not None:
            lost = (
                obs.owner_grid[site[0]][site[1]] != 1
                or obs.type_grid[site[0]][site[1]] != 1
            )
            if lost or memory.waited(obs.turn) > self.p.give_up_ticks:
                memory.set_site(None, obs.turn)
                site = None
        if site is None:
            site = self._choose_site(obs, memory, gen)
            memory.set_site(site, obs.turn)
        if site is None:
            return None

        cost = build_cost(obs, site[0], site[1])
        # Builds resolve before moves: the army must already stand there.
        if obs.army_grid[site[0]][site[1]] >= cost + self.p.build_keep:
            memory.set_site(None, obs.turn)
            memory.castles_built += 1
            return (2, site[0], site[1], 0, 0)

        exclude = self._garrisoned(obs)
        if gen is not None and not self._general_usable(obs, memory, gen):
            exclude.add(gen)
        walk = self._collection_walk(obs, site, exclude)
        if walk is not None:
            return walk
        return self._gather(obs, site, exclude=exclude | {site})

    def _collection_walk(self, obs, target: Cell, exclude: set[Cell]):
        """Walk the main stack to ``target``, absorbing surplus on the way —
        dragging every frontier cell to a rally instead strips the frontier
        to 1 army and silently kills expansion (the source's live traces
        showed exactly that)."""
        movable = [
            c
            for c in _owned_cells(obs)
            if c not in exclude and c != target and obs.army_grid[c[0]][c[1]] >= 2
        ]
        if not movable:
            return None
        stack = max(
            movable,
            key=lambda c: (obs.army_grid[c[0]][c[1]], (-c[0], -c[1])),
        )
        if obs.army_grid[stack[0]][stack[1]] < self.p.walk_min:
            return None  # nothing worth calling a main stack yet
        ok = _transit(obs)
        dist = multi_bfs(
            obs, [target], passable=lambda r, c: ok(r, c) or (r, c) == target
        )
        if dist[stack[0]][stack[1]] >= UNREACHABLE:
            return None
        best, best_key = None, None
        for n in _neighbors(obs, stack):
            if dist[n[0]][n[1]] >= dist[stack[0]][stack[1]]:
                continue
            if not (ok(*n) or n == target):
                continue
            own = 1 if obs.owner_grid[n[0]][n[1]] == 1 else 0
            key = (own, obs.army_grid[n[0]][n[1]] if own else 0)
            if best_key is None or key > best_key:
                best_key = key
                best = n
        return _move(stack, best) if best is not None else None

    # ------------------------------------------------------------------
    # 7. Pressure
    # ------------------------------------------------------------------
    def _pressure(self, obs, memory: MetroMemory, gen: Cell | None):
        """Spend the production lead: keep a wave in the enemy's half. This
        is what keeps Metro out of the draw-prone home-builder cluster."""
        ratio = memory.opp.army_ratio()
        held = len(owned_castles(obs))
        ready = (
            obs.turn >= self.p.pressure_from
            or ratio >= self.p.strong_ratio
            or (held >= 2 and ratio >= self.p.pressure_ratio)
        )
        if not ready:
            return None
        target = self._pressure_target(obs, memory)
        if target is None:
            return None

        exclude = self._garrisoned(obs)
        if gen is not None and not self._general_usable(obs, memory, gen):
            exclude.add(gen)
        movable = [
            c
            for c in _owned_cells(obs)
            if c not in exclude and obs.army_grid[c[0]][c[1]] > 1
        ]
        if not movable:
            memory.pushing = False
            memory.wave_tile = None
            return None

        spearhead = self._spearhead(obs, memory, movable)
        bar = self.p.min_push if memory.pushing else self._push_bar(obs, memory)
        if obs.army_grid[spearhead[0]][spearhead[1]] >= bar:
            step = next_step_toward(obs, spearhead, [target])
            if step is not None:
                memory.pushing = True
                memory.wave_tile = step
                return _move(spearhead, step)

        # Not big enough yet: consolidate at the cell nearest the target.
        memory.pushing = False
        memory.wave_tile = None
        dist = multi_bfs(obs, [target])
        reachable = [
            c for c in _owned_cells(obs) if dist[c[0]][c[1]] < UNREACHABLE
        ]
        if not reachable:
            return None
        rally = min(
            reachable,
            key=lambda c: (dist[c[0]][c[1]], -obs.army_grid[c[0]][c[1]], c),
        )
        return self._gather(obs, rally, exclude=exclude | {rally})

    def _spearhead(self, obs, memory: MetroMemory, movable: list[Cell]) -> Cell:
        """A wave already in the field keeps carrying the push, even once a
        castle back home holds more army."""
        biggest = max(
            movable,
            key=lambda c: (obs.army_grid[c[0]][c[1]], (-c[0], -c[1])),
        )
        wave = memory.wave_tile
        if (
            memory.pushing
            and wave is not None
            and wave in movable
            and obs.army_grid[wave[0]][wave[1]] >= self.p.min_push
            and obs.army_grid[biggest[0]][biggest[1]]
            < self.p.wave_takeover * obs.army_grid[wave[0]][wave[1]]
        ):
            return wave
        return biggest

    def _pressure_target(self, obs, memory: MetroMemory) -> Cell | None:
        """Their general, else their nearest land (eating territory strangles
        their land bonus and drags the general out of the fog)."""
        if memory.belief.enemy_general is not None:
            return memory.belief.enemy_general
        anchor = self._enemy_anchor(obs, memory)
        if anchor is not None:
            return anchor
        return None

    def _push_bar(self, obs, memory: MetroMemory) -> float:
        ceiling = max(self.p.min_push, self.p.max_push_share * obs.my_army)
        wanted = memory.opp.biggest_enemy_stack_now + 5
        return max(self.p.min_push, min(wanted, ceiling))

    # ------------------------------------------------------------------
    # 8. Fallbacks
    # ------------------------------------------------------------------
    def _consolidate(self, obs, memory: MetroMemory, gen: Cell | None):
        """Nothing to build or attack: tidy army onto a production cell."""
        rally = gen
        castles = owned_castles(obs)
        if castles:
            rally = min(castles, key=lambda c: (obs.army_grid[c[0]][c[1]], c))
        if rally is None:
            return None
        return self._gather(obs, rally, exclude={rally})

    def _any_move(self, obs, gen: Cell | None = None):
        """Last resort, restricted to moves that do not *lose* army."""
        best, best_key = None, None
        for cell in _owned_cells(obs):
            if obs.army_grid[cell[0]][cell[1]] < 2:
                continue
            for n in _neighbors(obs, cell):
                if not is_passable(obs.type_grid[n[0]][n[1]]):
                    continue
                if obs.owner_grid[n[0]][n[1]] == 1:
                    gain = 1  # harmless consolidation
                elif (
                    obs.army_grid[cell[0]][cell[1]] - 1
                    > obs.army_grid[n[0]][n[1]]
                ):
                    gain = 2  # actually takes the cell
                else:
                    continue
                key = (gain, 0 if cell == gen else 1, obs.army_grid[cell[0]][cell[1]])
                if best_key is None or key > best_key:
                    best_key = key
                    best = (cell, n)
        return _move(best[0], best[1]) if best is not None else None

    def _legal(self, obs, action) -> bool:
        """Reject anything the engine would drop, so a bug costs no tempo."""
        if action is None or len(action) != 5:
            return False
        p, r, c, d, _s = action
        if p == 1:
            return True
        if not (0 <= r < obs.H and 0 <= c < obs.W):
            return False
        if obs.owner_grid[r][c] != 1:
            return False
        if p == 2:
            return obs.type_grid[r][c] == 1
        if obs.army_grid[r][c] < 2:
            return False
        dr, dc = DIRECTIONS[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            return False
        return is_passable(obs.type_grid[nr][nc])


class MetroCore:
    """Stateful wrapper around :class:`MetroStrategy`, composable by proteus."""

    def __init__(self, player_id: int, H: int, W: int,
                 params: MetroParams | None = None,
                 model: OpponentModel | None = None) -> None:
        self.player_id = player_id
        self.H = H
        self.W = W
        self.strategy = MetroStrategy(params)
        self.memory = MetroMemory(opp=model or OpponentModel())

    def observe(self, obs) -> None:
        self.memory.observe(obs)

    def decide(self, obs):
        return self.strategy.decide(obs, self.memory)


class Agent:
    """Arena entrypoint."""

    def __init__(self, player_id: int, H: int, W: int) -> None:
        self._core = MetroCore(player_id, H, W)

    def act(self, obs):
        return self._core.decide(obs)

    def telemetry_extras(self) -> dict:
        mem = self._core.memory
        return {
            "castles_built": mem.castles_built,
            "pushing": 1 if mem.pushing else 0,
        }
