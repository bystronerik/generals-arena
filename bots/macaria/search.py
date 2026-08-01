"""macaria's tactical search: flat paranoid maximin over a pruned move matrix.

Interface (unchanged from the contract stub):

    TacticalSearch(params).improve(obs, core_move, core, deadline) -> Outcome

- `obs` is the bot's own fogged `Observation` (`bots/_common/wire.py`).
- `core_move` is the vendored core's already-legal action for this turn;
  returning it (or `None`) is always a valid outcome.
- `core` is the live `BlitzCore`. Its cross-turn belief is read
  (`memory.my_general`, `memory.belief.enemy_general`, `memory.target`,
  `memory.phase`); on an override, two board-tracking memory fields are
  REPAIRED — see `_repair_memory` for why that write is necessary.
- `deadline` is an absolute `time.monotonic()` timestamp, checked INSIDE the
  rollout loop (once per simulated ply), not just between iterations.

Why this algorithm and not a UCT tree
-------------------------------------
The realistic budget is ~55 ms of pure Python on a good turn — and possibly
almost nothing, since the core's own cost is wildly non-uniform (measured
p50 1-2 ms but max 85+ ms; the deadline is whatever is left). One rollout of
this forward model costs on the order of 50-150 us, so the sample budget is
a few hundred rollouts at best and a handful at worst. A UCT tree over even
10 root children would give each child a few dozen visits and each
grandchild almost none: far below where UCB's regret bounds pay for their
bookkeeping, with per-node allocation on top. At this sample count the
strongest use of the budget is to spend every rollout on a deliberately
chosen (my move, their reply) pair and none on exploration control:

- ROWS:   K pruned candidate moves of ours (core's move always row 0, PASS
          always present, plus attacks on nearby enemy stacks, sidesteps of
          our biggest threatened stack, the biggest stack's four moves, and
          adjacencies of a known enemy general).
- COLUMNS: M pruned opponent replies per row (PASS, each big visible stack
          stepping toward our general or onto our biggest adjacent stack,
          plus two row-specific replies: a chase onto our move's source and
          an attack on our move's destination).
- Each cell is ONE deterministic rollout: both sides play their first move,
  then a fixed greedy policy for the remaining plies; the leaf is scored by
  a material/threat evaluation. Row score = min over its replies (paranoid:
  the opponent is assumed to see our move), decision = argmax over rows.
- Iterative deepening on the horizon (`mcts_horizons`); the deepest pass
  that COMPLETED before the deadline decides. A pass cut mid-flight is
  discarded whole, so the decision never comes from a half-sampled matrix —
  when the budget is too small for even the first (cheapest) pass, the
  search declines and the core's move stands. Declining is free; a
  badly-sampled override is not.

The paranoid min is sound for a simultaneous-move game in the conservative
direction (it concedes the information race), and determinism means zero
variance — every rollout is spent on a distinct hypothesis, which flat
enumeration exploits and a bandit cannot beat until budgets are ~100x
larger.

Library survey (verdict recorded here; reasoning in the bot's doc)
------------------------------------------------------------------
PyPI `mcts` (1.0.4), `monte-carlo-tree-search` (2.1.0), `mcts-simple`
(1.1.0) and `mctspy` (0.1.1) were reviewed. All assume a perfect-
information, alternating-turn game exposing `get_possible_actions` /
`take_action` on a cheap, hashable state, and budget in wall-clock seconds
or whole playouts to terminal. This game is simultaneous-move (with a
resolution-order rule that IS the tactics: chasing > reinforcing > smaller
army), fog-of-war, and the budget is tens of milliseconds — every one of
those assumptions fails, and none of the packages ship in the sandbox
(`competition/requirements.txt`), so each would also need vendoring. The
cost of a search here is the forward model, which must be hand-built
against the engine's semantics regardless; the ~50 lines of control flow a
library would contribute are the wrong control flow. Verdict: implement
from first principles. (numba IS in the sandbox and could multiply the
rollout rate ~10x, but its first-call JIT costs seconds against a
150 ms/50-fault budget and the flat design is sample-sufficient without it
— noted as a future option, not taken.)

Forward model — what is simulated, exactly
------------------------------------------
Mirrors `competition-module/generals/core/game.py` step for step on the
visible board: move validity re-judged at execution time (a captured source
is a silent pass — this is what makes a chase onto an attacker's source a
real defence, per `modifiers/deathtouch.py`), move order by chasing >
reinforcing > smaller army with the seat-order tie (`core.player_id`),
strictly-more combat with ties to the defender, deathtouch from
`blitz_deathtouch_turn` (a valid execution onto the enemy general wins
regardless of armies), MUTUAL capture/touch detected as a draw (mirroring
the deathtouch modifier: when the first mover's win would be answered by
the second mover's still-valid capture of the other general, the exchange
is a draw, not a win), then `time += 1` and growth at the NEW time
(structures on even ticks, every owned cell at multiples of 50), stopping
at the turn-1200 hard draw.

Terrain is PERFECT information and modelled as such: `fog_cells` excludes
mountains/castles in the engine, so type 0 is fogged-but-passable ground
and type 5 is a fogged wall (a mountain — or an enemy castle built in fog,
which we conservatively also refuse to path through, since its garrison is
unknowable).

What it does NOT model, and why each omission is bounded at this horizon:

- FOG ARMY. Enemy cells we cannot see do not exist in the sim. The hidden
  total is exactly `opp_army - visible sum` (the scalars are unfogged), but
  it cannot be localised, and a constant added to every leaf changes no
  comparison — so it is omitted rather than guessed. Bounded: the search
  only fires on visible contact (scoping below), the horizon is <= ~10
  plies, and vision is a 3x3 dilation of ownership, so the first ring past
  our frontier is exact and the certainty horizon is one ply. The one
  fabrication allowed is the core's own: a once-seen, currently-fogged
  enemy general is injected with the same `1 + turn/2` garrison estimate
  the core's `enemy_general_army` uses — without it the finishing race the
  core is running could not be scored at all.
- CASTLE BUILDS, ours and theirs. A build spends 35+ army for 1 army per 2
  ticks; over <= 10 plies its swing is <= 5 army and only realizable by
  first banking 35, which the material eval already dominates.
- The opponent's REAL policy. Their first move is a small paranoid reply
  set; their continuation is one stack marching down the BFS gradient to
  our general. That is the most dangerous cheap model, not the most likely
  one — by design, since the min over replies is what a row must survive.
- Quiet strategy. The leaf eval is army diff + land diff + a threat penalty
  near our general. It can rank fights, trades and races; it cannot rank
  expansion plans. Note the eval is NOT anti-aggression: combat is
  army-diff-NEUTRAL (equal armies die on both sides) and land-positive, so
  attacking never loses points merely for being an attack — the known
  failure mode where a material chooser quietly turns the bot into a
  turtle does not apply to the diff form. What remains true is that quiet
  moves all score alike, which is why scoping refuses quiet positions and
  an override must beat the core's own row by a margin (ties keep the
  core).

Scoping — when the search may fire, and what it never overrides
---------------------------------------------------------------
Fires only on the mechanism triggers (measured union ~15-25% of turns):

- T1 finish window: enemy general location known and one of our 2+-army
  cells within `mcts_finish_radius` of it — the 0-20 turn window after a
  sighting decides most won games.
- T2 home tactical: a visible enemy stack of `mcts_min_enemy_army`+ within
  `mcts_scope_radius` (Manhattan) of our general. Active in the opening
  too — an early rush is exactly when the core's scalar threat model is
  weakest.
- T3 endgame: every turn from `mcts_endgame_from` (deathtouch era; the
  stack threshold drops to 2, the lethal minimum).
- T4 contested contact: a qualifying enemy stack within
  `mcts_scope_radius_move` of the core move's source or destination
  (post-opening only — the opening is near its arithmetic ceiling and no
  enemy is visible for most of it anyway).

Never overrides the core's `finish` move (a checked winning capture) or its
chase-defence at `blitz_chase_defend_from`+ (the provably-correct §07
defence).

Override hygiene: the core mutates its memory while DECIDING (the wave
engine pre-records the push destination in `mem.stack`; the opening chain
advances `chain_head`). An override would leave that memory describing a
game that never happened, and the core's own re-latch would then reset the
whole wave cycle. `_repair_memory` rewrites exactly the board-tracking
fields to match the move actually played — see its docstring for the
push/feed distinction.

Time control
------------
`agent.py` hands an absolute deadline (min of the whole-move cap remainder
and the search's own budget); the core's spend is highly non-uniform, so
the search assumes nothing about how much is left. `improve` declines
outright with less than `mcts_min_headroom_ms` on the clock; every rollout
checks the clock once per ply; a deepening pass that gets cut is discarded
whole. On any expiry the core's move stands. All knobs live in `params.py`
below the search marker.
"""
from __future__ import annotations

import time
from collections import deque
from typing import NamedTuple

PASS = (1, 0, 0, 0, 0)

#: Direction offsets, index-compatible with the engine's DIRECTIONS
#: (game.py: UP, DOWN, LEFT, RIGHT) and the wire protocol's dir field.
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

_UNREACH = 1 << 30

#: Core phase names this module keys on (see blitz_core; the strings are the
#: core's public trace vocabulary, not private state).
_PHASE_FINISH = "finish"
_PHASE_DEFEND = "defend"
_PHASE_ASSAULT = "assault"

#: Rollout outcomes (internal).
_NONE, _US, _THEM, _DRAW = 0, 1, 2, 3


class Outcome(NamedTuple):
    move: tuple[int, int, int, int, int] | None
    searched: bool
    iters: int


class _Ctx:
    """Root snapshot in flat arrays, plus everything precomputed per turn.

    Cells are flat indices ``idx = r * W + c``; a simulated move is
    ``(src_idx, dst_idx, split)`` and ``None`` means pass.
    """

    __slots__ = (
        "H", "W", "N", "turn", "seat0", "passable", "owner", "army",
        "my_gen", "opp_gen", "structures", "nbrs", "dist_gen", "dist_target",
        "my_army", "opp_army", "my_land", "opp_land", "opp_marcher", "my_big",
        "gen_box", "deathtouch_turn", "draw_turn", "stacks", "phase",
    )


def _bfs(N, nbrs, passable, sources):
    """Multi-source BFS over the flat grid; sources get 0 even if impassable."""
    dist = [_UNREACH] * N
    dq = deque()
    for s in sources:
        if dist[s]:
            dist[s] = 0
            dq.append(s)
    while dq:
        i = dq.popleft()
        d = dist[i] + 1
        for j, _dir in nbrs[i]:
            if dist[j] > d and passable[j]:
                dist[j] = d
                dq.append(j)
    return dist


def _we_first(owner, army, my, opp, seat0):
    """Mirror of game._determine_move_order: chasing > reinforcing > smaller
    army, seat order on a full tie. Pass ordering is irrelevant to outcomes
    (a pass changes nothing), so passes just yield the slot."""
    if my is None:
        return False
    if opp is None:
        return True
    ms, md, _ = my
    os_, od, _ = opp
    my_chase = md == os_
    opp_chase = od == ms
    if my_chase != opp_chase:
        return my_chase
    my_re = owner[md] == 1
    opp_re = owner[od] == 2
    if my_re != opp_re:
        return my_re
    a_my = army[ms]
    a_opp = army[os_]
    if a_my != a_opp:
        return a_my < a_opp
    return seat0


def _apply(owner, army, passable, mover, move, t, dt_turn, my_gen, opp_gen):
    """Execute one player's move, mirroring game._execute_move/_apply_move
    plus the deathtouch overlay. Mutates owner/army in place.

    Validity is re-judged HERE, at execution time, exactly as the engine
    does — that is what makes a chase that captures the attacker's source a
    real defence (the touch's source is no longer the attacker's: silent
    pass).

    Returns (winner, d_my_army, d_opp_army, d_my_land, d_opp_land);
    winner is 0 (none), 1 (us) or 2 (them).
    """
    if move is None:
        return 0, 0, 0, 0, 0
    src, dst, split = move
    if owner[src] != mover or not passable[dst]:
        return 0, 0, 0, 0, 0
    a = army[src]
    m = a // 2 if split else a - 1
    if m <= 0:
        return 0, 0, 0, 0, 0
    target_gen = opp_gen if mover == 1 else my_gen
    if dst == target_gen and target_gen >= 0 and t >= dt_turn:
        # §07 deathtouch: a valid execution onto the general wins outright.
        return mover, 0, 0, 0, 0
    down = owner[dst]
    if down == mover:
        army[src] = a - m
        army[dst] += m
        return 0, 0, 0, 0, 0
    tgt = army[dst]
    army[src] = a - m
    if m > tgt:  # strictly more: attacker takes the cell
        army[dst] = m - tgt
        owner[dst] = mover
        winner = mover if dst == target_gen and target_gen >= 0 else 0
        if mover == 1:
            return (winner, -tgt, -tgt if down == 2 else 0,
                    1, -1 if down == 2 else 0)
        return (winner, -tgt if down == 1 else 0, -tgt,
                -1 if down == 1 else 0, 1)
    army[dst] = tgt - m  # tie or worse: defender keeps the cell
    if mover == 1:
        return 0, -m, -m if down == 2 else 0, 0, 0
    return 0, -m if down == 1 else 0, -m, 0, 0


def _would_capture(owner, army, passable, mover, move, t, dt_turn, gen_idx):
    """Would `move`, applied to THIS state, still validly take `gen_idx`?

    Mirrors the deathtouch modifier's mutual-outcome test: after the first
    mover's win, the second mover's move is checked against the mid-state;
    if it would also capture (by army, or by touch at/after the threshold),
    the exchange is a DRAW, not a win. Scoring a drawn exchange as a win
    would make the search seek trades it does not actually win.
    """
    if move is None or gen_idx < 0:
        return False
    src, dst, split = move
    if dst != gen_idx or owner[src] != mover or not passable[dst]:
        return False
    a = army[src]
    m = a // 2 if split else a - 1
    if m <= 0:
        return False
    return t >= dt_turn or m > army[dst]


def _cont_my(ctx, owner, army, active, t, home):
    """Our greedy continuation: push the walker down a gradient — enemy-ward
    for "target" rows, toward our own general for "home" (gather) rows —
    refusing steps that would throw the stack away on a defender it cannot
    beat (unless it is the enemy general under deathtouch). Homeward walks
    merge every own cell they cross, which is what turns a walk into a
    gather."""
    if active < 0 or owner[active] != 1 or army[active] < 2:
        return None
    dist = ctx.dist_gen if home else ctx.dist_target
    if dist is None:
        return None
    bd = dist[active]
    if bd >= _UNREACH:
        return None
    m = army[active] - 1
    best = -1
    best_d = bd
    for j, _dir in ctx.nbrs[active]:
        dj = dist[j]
        if dj >= best_d:
            continue
        if owner[j] != 1 and army[j] >= m and not (
            j == ctx.opp_gen and t >= ctx.deathtouch_turn
        ):
            continue
        best_d = dj
        best = j
    if best < 0:
        return None
    return (active, best, 0)


def _cont_opp(ctx, owner, army, active):
    """Their greedy continuation (paranoid): the walker — or, if it died,
    their biggest root stack — marches down the gradient to our general,
    fighting whatever is in the way."""
    src = active
    if src < 0 or owner[src] != 2 or army[src] < 2:
        src = ctx.opp_marcher
        if src < 0 or owner[src] != 2 or army[src] < 2:
            return None
    dist = ctx.dist_gen
    bd = dist[src]
    if bd >= _UNREACH or bd == 0:
        return None
    best = -1
    best_d = bd
    for j, _dir in ctx.nbrs[src]:
        dj = dist[j]
        if dj < best_d:
            best_d = dj
            best = j
    if best < 0:
        return None
    return (src, best, 0)


class TacticalSearch:
    def __init__(self, params) -> None:
        self.params = params

    # ------------------------------------------------------------ entry
    def improve(self, obs, core_move, core, deadline: float) -> Outcome:
        p = self.params
        if time.monotonic() + p.mcts_min_headroom_ms / 1000.0 >= deadline:
            return Outcome(move=None, searched=False, iters=0)

        mem = core.memory
        # Never second-guess a checked winning capture or the §07 chase
        # defence — both are exact, and "improving" them can only lose.
        if mem.phase == _PHASE_FINISH:
            return Outcome(move=None, searched=False, iters=0)
        if mem.phase == _PHASE_DEFEND and obs.turn >= p.blitz_chase_defend_from:
            return Outcome(move=None, searched=False, iters=0)
        gen = mem.my_general
        if gen is None:
            return Outcome(move=None, searched=False, iters=0)

        stacks_rc = self._scope(obs, core_move, mem, gen)
        if stacks_rc is None:
            return Outcome(move=None, searched=False, iters=0)

        ctx = self._build_ctx(obs, core, stacks_rc)
        rows = self._root_moves(ctx, core_move)  # [(rep, intent), ...]
        base = self._base_replies(ctx)
        row_replies = [self._row_replies(ctx, base, rep) for rep, _ in rows]

        iters = 0
        decision = None
        headroom = p.mcts_min_headroom_ms / 1000.0
        for horizon in p.mcts_horizons:
            if time.monotonic() + headroom >= deadline:
                break
            scores: list[float] = []
            best_so_far = -1e18
            aborted = False
            for ri, (rep, intent) in enumerate(rows):
                row_min = 1e18
                for reply in row_replies[ri]:
                    score = self._rollout(
                        ctx, rep, reply, intent, horizon, deadline
                    )
                    if score is None:
                        aborted = True
                        break
                    iters += 1
                    if score < row_min:
                        row_min = score
                        # A non-core row already below the leader cannot win;
                        # its recorded (partial) min keeps it unselectable.
                        if ri > 0 and row_min <= best_so_far:
                            break
                if aborted:
                    break
                scores.append(row_min)
                if row_min > best_so_far:
                    best_so_far = row_min
            if aborted:
                break
            decision = scores  # the deepest COMPLETED pass decides

        if decision is None:
            # Ran, but not even the cheapest pass finished: no conclusion —
            # a half-sampled matrix must not outvote the core.
            return Outcome(move=None, searched=True, iters=iters)
        core_score = decision[0]
        best_i = max(range(len(decision)), key=lambda i: decision[i])
        if best_i != 0 and decision[best_i] > core_score + p.mcts_override_margin:
            self._repair_memory(ctx, mem, rows[0][0], rows[best_i][0])
            return Outcome(
                move=self._wire(ctx, rows[best_i][0]), searched=True, iters=iters
            )
        return Outcome(move=None, searched=True, iters=iters)

    # ------------------------------------------------------------ scoping
    def _scope(self, obs, core_move, mem, gen):
        """The mechanism triggers T1-T4 (see module docstring). Returns the
        visible enemy stack list [(army, r, c), ...] when the search should
        fire, else None."""
        p = self.params
        endgame = obs.turn >= p.mcts_endgame_from
        thr = 2 if endgame else p.mcts_min_enemy_army  # 2 = lethal minimum
        stacks_rc = []
        og, ag = obs.owner_grid, obs.army_grid
        for r in range(obs.H):
            orow, arow = og[r], ag[r]
            for c in range(obs.W):
                if orow[c] == 2 and arow[c] >= thr:
                    stacks_rc.append((arow[c], r, c))
        if not stacks_rc:
            return None  # nothing visible to search against

        if endgame:  # T3
            return stacks_rc

        eg = mem.belief.enemy_general
        if eg is not None:  # T1 finish window
            er, ec = eg
            fr = p.mcts_finish_radius
            for r in range(obs.H):
                orow, arow = og[r], ag[r]
                for c in range(obs.W):
                    if (
                        orow[c] == 1
                        and arow[c] >= 2
                        and abs(r - er) + abs(c - ec) <= fr
                    ):
                        return stacks_rc

        gr, gc = gen  # T2 home tactical (opening included: early rushes)
        rad = p.mcts_scope_radius
        for _, r, c in stacks_rc:
            if abs(r - gr) + abs(c - gc) <= rad:
                return stacks_rc

        # T4 contested contact around the core's own move — ASSAULT phase
        # only. During rebuild/rally the core is running its economy, which
        # the mechanism analysis puts outside a short search's reach; firing
        # there let the search bleed the wave cycle one marginal override at
        # a time (measured: 88 rebuild-phase overrides in one game). The
        # search's quiet-phase mandate is defence (T2) and finish (T1) only.
        if (
            mem.phase == _PHASE_ASSAULT
            and core_move[0] == 0
            and obs.turn >= p.blitz_opening_end
        ):
            sr, sc = core_move[1], core_move[2]
            dr, dc = _DIRS[core_move[3]]
            tr, tc = sr + dr, sc + dc
            rad = p.mcts_scope_radius_move
            for _, r, c in stacks_rc:
                if (
                    abs(r - sr) + abs(c - sc) <= rad
                    or abs(r - tr) + abs(c - tc) <= rad
                ):
                    return stacks_rc
        return None

    # ------------------------------------------------------- root snapshot
    def _build_ctx(self, obs, core, stacks_rc):
        p = self.params
        H, W = obs.H, obs.W
        N = H * W
        tg, og, ag = obs.type_grid, obs.owner_grid, obs.army_grid

        passable = [True] * N
        owner = [0] * N
        army = [0] * N
        structures: list[int] = []
        my_army = opp_army = my_land = opp_land = 0
        my_big = -1
        my_big_a = 1
        i = 0
        for r in range(H):
            trow, orow, arow = tg[r], og[r], ag[r]
            for c in range(W):
                t = trow[c]
                if t == 2 or t == 5:
                    passable[i] = False
                o = orow[c]
                a = arow[c]
                owner[i] = o
                army[i] = a
                if t == 3 or t == 4:
                    structures.append(i)
                if o == 1:
                    my_army += a
                    my_land += 1
                    if a > my_big_a:
                        my_big_a = a
                        my_big = i
                elif o == 2:
                    opp_army += a
                    opp_land += 1
                i += 1

        nbrs: list[list[tuple[int, int]]] = [None] * N  # type: ignore
        for r in range(H):
            base = r * W
            for c in range(W):
                lst = []
                if r > 0:
                    lst.append((base - W + c, 0))
                if r < H - 1:
                    lst.append((base + W + c, 1))
                if c > 0:
                    lst.append((base + c - 1, 2))
                if c < W - 1:
                    lst.append((base + c + 1, 3))
                nbrs[base + c] = lst

        mem = core.memory
        gr, gc = mem.my_general
        my_gen = gr * W + gc

        opp_gen = -1
        eg = mem.belief.enemy_general
        if eg is not None:
            gi = eg[0] * W + eg[1]
            if tg[eg[0]][eg[1]] == 0:
                # Once-seen, fogged now: inject the core's own garrison
                # estimate (strategy_common.enemy_general_army) so the
                # finishing race can be scored. The only fabricated state.
                est = 1 + obs.turn // p.blitz_general_growth_period
                owner[gi] = 2
                army[gi] = est
                structures.append(gi)
                opp_army += est
                opp_land += 1
            opp_gen = gi

        stacks_rc.sort(reverse=True)
        stacks = [(a, r * W + c) for a, r, c in stacks_rc[: p.mcts_enemy_stacks]]

        ctx = _Ctx()
        ctx.H, ctx.W, ctx.N = H, W, N
        ctx.turn = obs.turn
        ctx.seat0 = core.player_id == 0
        ctx.passable = passable
        ctx.owner = owner
        ctx.army = army
        ctx.my_gen = my_gen
        ctx.opp_gen = opp_gen
        ctx.structures = structures
        ctx.nbrs = nbrs
        ctx.my_army, ctx.opp_army = my_army, opp_army
        ctx.my_land, ctx.opp_land = my_land, opp_land
        ctx.stacks = stacks
        ctx.opp_marcher = stacks[0][1] if stacks else -1
        ctx.my_big = my_big
        ctx.phase = mem.phase
        ctx.deathtouch_turn = p.blitz_deathtouch_turn
        ctx.draw_turn = p.mcts_draw_turn

        ctx.dist_gen = _bfs(N, nbrs, passable, [my_gen])
        target = opp_gen
        if target < 0 and mem.target is not None:
            target = mem.target[0] * W + mem.target[1]
        ctx.dist_target = (
            _bfs(N, nbrs, passable, [target]) if target >= 0 else None
        )

        box = []
        tr = p.mcts_threat_radius
        for dr in range(-tr, tr + 1):
            for dc in range(-tr + abs(dr), tr - abs(dr) + 1):
                r, c = gr + dr, gc + dc
                if 0 <= r < H and 0 <= c < W and (dr or dc):
                    box.append((r * W + c, abs(dr) + abs(dc)))
        ctx.gen_box = box
        return ctx

    # -------------------------------------------------------- candidates
    def _root_moves(self, ctx, core_move):
        """Our K rows as (rep, intent). Core's move is always row 0; PASS
        always survives. `intent` steers the rollout continuation: "target"
        walkers push down the enemy-ward gradient, "home" walkers keep
        gathering toward our general."""
        p = self.params
        owner, army = ctx.owner, ctx.army
        rows: list = []
        seen: set = set()

        def add(rep, intent="target"):
            key = rep[:2] if rep else None
            if key in seen:
                return
            seen.add(key)
            rows.append((rep, intent))

        add(self._rep(ctx, core_move))
        add(None)  # hold

        # Attack each big enemy stack with our largest adjacent stack.
        for _, e in ctx.stacks:
            best, ba = -1, 1
            for j, _d in ctx.nbrs[e]:
                if owner[j] == 1 and army[j] > ba:
                    ba, best = army[j], j
            if best >= 0:
                add((best, e, 0))

        # Home-defence gather. The taxonomy's largest loss class (59%) is
        # the general falling while ahead, with a median 84 mobile army near
        # home against a 33-army raider — the army is spread one-per-cell
        # while the opponent arrives as one pile. This row starts the pile:
        # our biggest stack near home steps toward the general, and its
        # "home" continuation keeps gathering (merges pick up every cell it
        # crosses), so the matrix can see an 8-ply gather beat an 8-ply
        # raid. Re-chosen each turn while T2 keeps firing, the per-turn
        # winner composes into the multi-turn gather a single move cannot
        # express. (Honest limit: coherence depends on this row winning on
        # consecutive turns; the search cannot commit to a plan.)
        #
        # GATED on root-level danger: the row exists only when the best
        # discounted enemy pile already beats garrison + one interceptor.
        # Ungated, this row out-scored the core's rebuild expansion every
        # turn a raider merely EXISTED in radius (measured: 176 of 225
        # overrides in one game were rebuild-phase gathers), starving the
        # wave economy and dragging a 371-turn win out to 771 turns.
        dg = ctx.dist_gen
        threat0 = 0
        guard0 = 0
        for i, d in ctx.gen_box:
            o = owner[i]
            v = army[i] - d
            if o == 2 and v > threat0:
                threat0 = v
            elif o == 1 and v > guard0:
                guard0 = v
        danger = threat0 > army[ctx.my_gen] + guard0
        guard, ga_ = -1, 1
        rad = p.mcts_scope_radius
        if danger:
            for i in range(ctx.N):
                if owner[i] == 1 and army[i] > ga_ and 0 < dg[i] <= rad:
                    ga_, guard = army[i], i
        if guard >= 0:
            step, sd, sa = -1, dg[guard], -1
            for j, _d in ctx.nbrs[guard]:
                dj = dg[j]
                if dj < sd or (dj == sd and owner[j] == 1 and army[j] > sa):
                    sd, sa, step = dj, army[j] if owner[j] == 1 else -1, j
            if step >= 0:
                add((guard, step, 0), "home")

        # Our biggest stack that a neighbouring enemy stack could kill:
        # every legal sidestep (flee, merge, or counter — the matrix judges).
        threatened, ta = -1, 1
        for _, e in ctx.stacks:
            ea = army[e]
            for j, _d in ctx.nbrs[e]:
                if owner[j] == 1 and army[j] > ta and ea > army[j] - 1:
                    ta, threatened = army[j], j
        if threatened >= 0:
            for j, _d in ctx.nbrs[threatened]:
                if ctx.passable[j]:
                    add((threatened, j, 0))

        # The biggest stack's four moves — detours during the assault, or
        # repositioning when home is genuinely in danger. Not offered in
        # quiet phases: with the progress term these rows outbid the core's
        # rebuild economy, which is not the search's call to make.
        if (
            (ctx.phase == _PHASE_ASSAULT or danger)
            and ctx.my_big >= 0
            and army[ctx.my_big] >= 2
        ):
            for j, _d in ctx.nbrs[ctx.my_big]:
                if ctx.passable[j]:
                    add((ctx.my_big, j, 0))

        # Adjacent pressure on a known enemy general.
        if ctx.opp_gen >= 0:
            for j, _d in ctx.nbrs[ctx.opp_gen]:
                if owner[j] == 1 and army[j] >= 2:
                    add((j, ctx.opp_gen, 0))

        return rows[: p.mcts_max_root_moves]

    def _rep(self, ctx, wire):
        if wire is None or wire[0] != 0:
            return None
        sr, sc, d = wire[1], wire[2], wire[3]
        dr, dc = _DIRS[d]
        tr, tc = sr + dr, sc + dc
        if not (0 <= tr < ctx.H and 0 <= tc < ctx.W):
            return None
        return (sr * ctx.W + sc, tr * ctx.W + tc, wire[4])

    def _wire(self, ctx, rep):
        if rep is None:
            return PASS
        src, dst, split = rep
        sr, sc = divmod(src, ctx.W)
        tr, tc = divmod(dst, ctx.W)
        for d, (dr, dc) in enumerate(_DIRS):
            if sr + dr == tr and sc + dc == tc:
                return (0, sr, sc, d, split)
        return PASS  # unreachable by construction

    def _base_replies(self, ctx):
        """Replies shared by every row: PASS, each big stack stepping down
        the gradient to our general, and each big stack eating our largest
        adjacent cell."""
        owner, army = ctx.owner, ctx.army
        dg = ctx.dist_gen
        replies: list[tuple[int, int, int] | None] = [None]
        seen: set = set()
        for _, e in ctx.stacks:
            best, bd = -1, dg[e]
            atk, aa = -1, 1
            for j, _d in ctx.nbrs[e]:
                if dg[j] < bd:
                    bd, best = dg[j], j
                if owner[j] == 1 and army[j] > aa:
                    aa, atk = army[j], j
            for dst in (best, atk):
                if dst >= 0 and (e, dst) not in seen:
                    seen.add((e, dst))
                    replies.append((e, dst, 0))
        return replies

    def _row_replies(self, ctx, base, rep):
        """base + two row-specific replies: chase our move's source, attack
        our move's destination. Min over a row's own relevant dangers."""
        p = self.params
        if rep is None:
            return base[: p.mcts_opp_replies]
        out = list(base)
        seen = {r[:2] for r in base if r}
        owner, army = ctx.owner, ctx.army
        for cell in (rep[0], rep[1]):
            best, ba = -1, 1
            for j, _d in ctx.nbrs[cell]:
                if owner[j] == 2 and army[j] > ba:
                    ba, best = army[j], j
            if best >= 0 and (best, cell) not in seen:
                seen.add((best, cell))
                out.append((best, cell, 0))
        # Row-specific dangers matter more than the tail of the shared list.
        if len(out) > p.mcts_opp_replies:
            extras = len(out) - len(base)
            out = out[: p.mcts_opp_replies - extras] + out[len(base):]
        return out

    # ------------------------------------------------------ memory repair
    def _repair_memory(self, ctx, mem, core_rep, my_rep):
        """Make the core's memory describe the game actually being played.

        The core mutates memory while DECIDING, not after the move resolves:
        `assault_move` pre-records the push destination in `mem.stack` (and
        the rally collection-walk does the same), `opening_move` advances
        `chain_head`/`chain_visited`, and the defence branch rewrites
        `mem.stack` too. Overriding the move leaves those fields describing
        a fiction; worse, `ensure_stack`'s re-latch would then set
        `restacked=True` next turn and `start_wave` would reset the whole
        wave cycle — a silent, systematic regression channel, not a one-off.

        Only the two BOARD-tracking fields are repaired. The programme
        fields (`committed`, `strikes`, `feed_ticks`, `rally_until`,
        `rebuild_until`) describe which stretch of the wave cycle we are in;
        substituting one tactical move does not falsify them.

        Push vs feed: both a stack PUSH (stack at src, dst pre-recorded)
        and a FEED (small cell moving onto the stack at dst) end with
        `mem.stack == core dst`. They are told apart by ensure_stack's own
        latch invariant — the stack is the larger cell — so `army[src] >=
        army[dst]` means push (repair needed), else feed (`mem.stack`
        already points at the real, unmoved stack).
        """
        if core_rep is None:
            return
        src, dst, _split = core_rep
        W = ctx.W
        dst_rc = divmod(dst, W)
        if mem.stack == dst_rc:
            if ctx.owner[src] == 1 and ctx.army[src] >= ctx.army[dst]:
                if my_rep is not None and my_rep[0] == src:
                    # We moved the same stack elsewhere; assume the capture
                    # succeeds — the same optimism the core's push applies.
                    mem.stack = divmod(my_rep[1], W)
                else:
                    mem.stack = divmod(src, W)  # the push is not happening
        if mem.chain_head == dst_rc:
            # The chain step is not being taken. A None head makes
            # opening_move relaunch cleanly next turn (its own owner check
            # would fail anyway; this just skips the dangling state).
            mem.chain_head = None

    # ----------------------------------------------------------- rollout
    def _rollout(self, ctx, my0, opp0, intent, horizon, deadline):
        """One deterministic rollout; returns a leaf score, or None if the
        deadline fired mid-rollout (the whole pass is then discarded)."""
        p = self.params
        home = intent == "home"
        monotonic = time.monotonic
        owner = ctx.owner[:]
        army = ctx.army[:]
        passable = ctx.passable
        structures = ctx.structures
        my_gen, opp_gen = ctx.my_gen, ctx.opp_gen
        dt_turn = ctx.deathtouch_turn
        seat0 = ctx.seat0
        my_a, opp_a = ctx.my_army, ctx.opp_army
        my_l, opp_l = ctx.my_land, ctx.opp_land
        t = ctx.turn
        # RULES.md §04's two growth cadences. Hoisted out of the ply loop for
        # speed, and read from params rather than written as literals so the
        # simulation and the vendored core cannot drift apart: `blitz_core`
        # routes the same two constants through the same fields.
        growth_period = p.blitz_general_growth_period
        bonus_period = p.blitz_land_bonus_period
        outcome = _NONE
        my_active = -1
        opp_active = -1
        # Progress anchor: the SAME object in every row — the biggest stack
        # (blitz's tempo carrier), followed through the rollout. Anchoring
        # each row on its own walker instead biased the term systematically:
        # after a spent wave the leftover forward stack made PASS score
        # ~zero remaining distance while the core's rebuild expansion was
        # charged the full board width, and the search overrode the economy
        # every turn (measured: 216 rebuild-phase overrides in one game).
        my_pos = ctx.my_big
        mv_my, mv_opp = my0, opp0
        ply = 0
        for ply in range(horizon):
            # §08: the clock is read INSIDE the rollout loop — one long
            # rollout must never be able to blow the turn.
            if monotonic() >= deadline:
                return None
            if ply:
                mv_my = _cont_my(ctx, owner, army, my_active, t, home)
                mv_opp = _cont_opp(ctx, owner, army, opp_active)
            if _we_first(owner, army, mv_my, mv_opp, seat0):
                first, fm, second, sm = 1, mv_my, 2, mv_opp
            else:
                first, fm, second, sm = 2, mv_opp, 1, mv_my
            w, da, doa, dl, dol = _apply(
                owner, army, passable, first, fm, t, dt_turn, my_gen, opp_gen
            )
            my_a += da
            opp_a += doa
            my_l += dl
            opp_l += dol
            if w:
                # Mutual capture/touch is a DRAW (deathtouch modifier): test
                # the second mover's move against the mid-state.
                other_gen = my_gen if second == 1 else opp_gen
                if _would_capture(
                    owner, army, passable, second, sm, t, dt_turn, other_gen
                ):
                    outcome = _DRAW
                else:
                    outcome = w
                break
            w, da, doa, dl, dol = _apply(
                owner, army, passable, second, sm, t, dt_turn, my_gen, opp_gen
            )
            my_a += da
            opp_a += doa
            my_l += dl
            opp_l += dol
            if w:
                outcome = w
                break
            if mv_my is not None:
                d = mv_my[1]
                my_active = d if owner[d] == 1 and army[d] >= 2 else -1
                if mv_my[0] == my_pos and owner[d] == 1:
                    my_pos = d  # the tracked stack moved (and arrived)
            if mv_opp is not None:
                d = mv_opp[1]
                opp_active = d if owner[d] == 2 and army[d] >= 2 else -1
            # Engine step order: both actions, then time += 1, then growth
            # at the NEW time (game.step -> global_update).
            t += 1
            if t >= ctx.draw_turn:
                break
            if t % growth_period == 0:
                for i in structures:
                    o = owner[i]
                    if o == 1:
                        army[i] += 1
                        my_a += 1
                    elif o == 2:
                        army[i] += 1
                        opp_a += 1
            if t % bonus_period == 0:
                for i in range(ctx.N):
                    o = owner[i]
                    if o == 1:
                        army[i] += 1
                        my_a += 1
                    elif o == 2:
                        army[i] += 1
                        opp_a += 1

        if outcome == _US:
            return p.mcts_win_score - ply  # sooner is better
        if outcome == _THEM:
            return -p.mcts_win_score + ply  # later is less bad
        if outcome == _DRAW:
            return p.mcts_draw_score
        score = p.mcts_w_army * (my_a - opp_a) + p.mcts_w_land * (my_l - opp_l)
        # General safety: biggest enemy pile near home, discounted by its
        # distance, against the garrison PLUS one interceptor — the best own
        # pile in the box, same one-reinforcer pessimism as the core's
        # home_threat. Counting the interceptor is what lets a gathered
        # defence score as a defence without sitting on the general itself.
        threat = 0
        guard = 0
        for i, d in ctx.gen_box:
            o = owner[i]
            if o == 2:
                v = army[i] - d
                if v > threat:
                    threat = v
            elif o == 1:
                v = army[i] - d
                if v > guard:
                    guard = v
        defence = army[my_gen] + guard
        if threat > defence:
            score -= p.mcts_w_threat * (threat - defence)
        # Progress: without this, the paranoid min systematically prefers a
        # local trade to pushing PAST a defender (trades are eval-neutral,
        # approach earned nothing), and the search bleeds the assault's
        # tempo one override at a time — measured as games stretching by
        # hundreds of turns. Half an army-unit per remaining hop keeps
        # "closer to the target" worth something without outbidding a real
        # fight (w_threat prices a raider at 3x per army).
        dist_t = ctx.dist_target
        if dist_t is not None and my_pos >= 0 and owner[my_pos] == 1:
            d = dist_t[my_pos]
            if d < _UNREACH:
                score -= p.mcts_w_progress * d
        return score
