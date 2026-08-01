"""Monte Carlo tree search over a tactical window, under a hard wall-clock cap.

Read this before trusting a number that comes out of it.

Scope — why not the whole game
------------------------------
A full-game MCTS at 150 ms a turn in CPython is not a credible object. The
rollout cost is dominated by move generation, and the branching factor of the
real game is ~4 x (owned cells with 2+ army), which past the opening is 40-200
moves a ply. Measured on this machine (see docs/research/strategies/yankee.md),
one 12-ply rollout of the model below costs ~0.2-0.4 ms, so a 40 ms budget buys
O(100) iterations. Spread over a whole game that is noise; spent on the handful
of turns where the game is actually decided it is a real second opinion.

So the search runs only in a **tactical window**: an enemy stack within
`window` BFS steps of our general, or one of our stacks within `window` steps
of a *known* enemy general. Increment-2's measurement says how often that
fires. Everywhere else the core's heuristic move is returned untouched and the
search never starts.

The search never *replaces* the fallback blindly: the heuristic move is always
root child 0, so a search that learns nothing returns it by construction.

Time
----
`budget_ms` is a self-enforced cap on the search alone, and `Deadline` is
checked inside the rollout loop, once per simulated ply — not merely between
iterations, which would let one long rollout blow the budget. The caller
additionally passes the time already spent this turn so the cap shrinks to fit
what is left of the move (RULES.md §08: 150 ms, and 50 late replies forfeit).
On expiry the best root child so far is returned; with zero completed
iterations that is the heuristic move.

Fidelity to RULES.md
--------------------
Modelled exactly:

- §02 move order — `_order` implements chasing > reinforcing > smaller army,
  with the full tie going to player 0, which is why the search needs our real
  seat and not the perspective-relative owner code.
- §02 move semantics — all-but-one or half, one army always left behind,
  invalid moves are a silent pass.
- §05 combat — armies subtract, the attacker takes the cell only with strictly
  more, an exact tie leaves the defender in place.
- §04 growth — structures on even ticks, every cell every 50 ticks, applied
  after both moves, on the incremented tick.
- §07 deathtouch — from `deathtouch_turn` a move that executes onto the enemy
  general wins regardless of army.

**Not** modelled, and each one is a real gap:

- **Castle builds.** The search proposes none and simulates the opponent
  building none. Against a castle programme its opponent model is wrong.
- **Fog** — see below. This is the big one.
- **Rollout policy.** Both seats play a cheap greedy policy, not their real
  strategy, so a value estimate is only as good as that policy is
  representative. It is tuned to be *pessimistic for us*: the opponent marches
  its largest visible stack at our general.

The belief model, and where it is a guess
-----------------------------------------
The true enemy state is unobservable (§06). The search runs on this belief,
which is the observation taken literally:

- Cells we can see are exact — owner, army, type.
- **Fog cells are assumed empty and neutral.** The wire protocol reports army 0
  outside vision, and the search does not invent a distribution over what is
  there. So an enemy stack one step outside our vision does not exist as far as
  the search is concerned, and the search therefore *systematically
  underestimates incoming force*. This is a guess, and it is the least
  defensible thing in this file.
- **The opponent's off-screen army is not placed.** `obs.opp_army` says how
  much army the opponent has in total; the search knows the visible part and
  drops the rest rather than sampling positions for it. Sampling from
  `OpponentModel` was considered and not done: the model tracks aggregates and
  a latched nearest-approach, not a spatial distribution, so any sample would
  be a prior dressed as evidence. The honest version is "visible only", stated.
- **A fogged enemy general is not searched for.** With no known enemy general
  the kill window never opens and only the defensive window can fire.

The practical consequence: this search is trustworthy about *the fight it can
see* and blind about reinforcements arriving from fog. In the defensive window
— the one the cm_hunter losses live in — the threatening stack is visible by
construction, which is why that is the window it is scoped to.
"""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Wire protocol type codes (competition/protocol.py).
T_FOG = 0
T_PLAIN = 1
T_MOUNTAIN = 2
T_CASTLE = 3
T_GENERAL = 4
T_STRUCT_FOG = 5

PASS_ACTION = (1, 0, 0, 0, 0)

_WIN = 1.0
_LOSS = -1.0
_DRAW = 0.0


@dataclass
class SearchStats:
    """What the search actually did — reported, never inferred."""

    turns_searched: int = 0
    turns_skipped: int = 0
    iterations: int = 0
    max_iterations_one_turn: int = 0
    overrides: int = 0
    """Turns the search returned something other than the heuristic move."""
    budget_exhausted: int = 0
    latencies_ms: list = field(default_factory=list)

    def note(self, ms: float) -> None:
        self.latencies_ms.append(ms)


class _Sim:
    """The forward model: flat numpy board plus scalar updates.

    Board-sized numpy arrays for the predicates that touch every cell (find
    movable cells, apply the every-50 land bonus) and plain Python ints for the
    two-or-three cells a move actually changes. Mixing them is deliberate:
    `np.flatnonzero` over 441 int8s costs microseconds where a Python loop
    costs tens, and a single-element numpy store costs more than a list store.
    """

    __slots__ = (
        "owner", "army", "passable", "prod", "H", "W", "n",
        "my_gen", "op_gen", "turn", "seat", "deathtouch_turn", "winner",
    )

    def __init__(self, obs, seat: int, deathtouch_turn: int):
        H, W = obs.H, obs.W
        self.H, self.W, self.n = H, W, H * W
        types = np.asarray(obs.type_grid, dtype=np.int8).ravel()
        self.owner = np.asarray(obs.owner_grid, dtype=np.int8).ravel().copy()
        self.army = np.asarray(obs.army_grid, dtype=np.int32).ravel().copy()
        # Fog and structures-in-fog: mountains and fogged structures are walls
        # for planning, exactly as _common.strategy_common.is_passable has it.
        self.passable = (types != T_MOUNTAIN) & (types != T_STRUCT_FOG)
        self.prod = (types == T_GENERAL) | (types == T_CASTLE)
        self.turn = obs.turn
        self.seat = seat
        self.deathtouch_turn = deathtouch_turn
        self.winner = -1
        self.my_gen = -1
        self.op_gen = -1

    def clone(self) -> "_Sim":
        cp = _Sim.__new__(_Sim)
        cp.H, cp.W, cp.n = self.H, self.W, self.n
        cp.owner = self.owner.copy()
        cp.army = self.army.copy()
        cp.passable = self.passable          # immutable across a search
        cp.prod = self.prod                  # ditto
        cp.turn = self.turn
        cp.seat = self.seat
        cp.deathtouch_turn = self.deathtouch_turn
        cp.winner = self.winner
        cp.my_gen = self.my_gen
        cp.op_gen = self.op_gen
        return cp

    # ------------------------------------------------------------- mechanics
    def movable(self, side: int) -> np.ndarray:
        """Indices of `side`'s cells that can legally source a move (§02)."""
        return np.flatnonzero((self.owner == side) & (self.army >= 2))

    def _dest(self, idx: int, d: int) -> int:
        r, c = divmod(idx, self.W)
        dr, dc = DIRECTIONS[d]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.H and 0 <= nc < self.W):
            return -1
        return nr * self.W + nc

    def _valid(self, side: int, mv) -> bool:
        if mv is None or mv[0] != 0:
            return False
        idx, d, split = mv[1], mv[2], mv[3]
        if self.owner[idx] != side:
            return False
        src = int(self.army[idx])
        moving = src // 2 if split else src - 1
        if moving <= 0:
            return False
        dst = self._dest(idx, d)
        return dst >= 0 and bool(self.passable[dst])

    def _apply(self, side: int, mv) -> None:
        """One player's move, exactly game._apply_move (§02, §05, §07)."""
        idx, d, split = mv[1], mv[2], mv[3]
        dst = self._dest(idx, d)
        src_army = int(self.army[idx])
        moving = src_army // 2 if split else src_army - 1

        touch = (
            self.turn >= self.deathtouch_turn
            and dst == (self.op_gen if side == 1 else self.my_gen)
            and dst >= 0
        )

        if self.owner[dst] == side:
            self.army[dst] += moving
            self.army[idx] -= moving
            return

        defenders = int(self.army[dst])
        self.army[idx] -= moving
        wins = moving > defenders or touch
        self.army[dst] = abs(defenders - moving)
        if wins:
            captured_general = dst in (self.my_gen, self.op_gen)
            self.owner[dst] = side
            if touch:
                self.army[dst] = max(1, moving - defenders)
            if captured_general:
                self.winner = side

    def step(self, mv_me, mv_op) -> None:
        """Both moves plus growth — one engine tick."""
        me, op = 1, 2
        valid_me = self._valid(me, mv_me)
        valid_op = self._valid(op, mv_op)
        first = self._order(mv_me if valid_me else None, mv_op if valid_op else None)

        order = ((me, mv_me, valid_me), (op, mv_op, valid_op))
        if first == op:
            order = order[::-1]
        for side, mv, ok in order:
            if ok and self.winner < 0 and self._valid(side, mv):
                self._apply(side, mv)

        self.turn += 1
        if self.winner >= 0:
            return
        if self.turn % 50 == 0:
            self.army[self.owner == 1] += 1
            self.army[self.owner == 2] += 1
        if self.turn % 2 == 0:
            grow = self.prod & (self.owner != 0)
            self.army[grow] += 1

    def _order(self, mv_me, mv_op) -> int:
        """RULES.md §02: chasing > reinforcing > smaller army; ties to seat A.

        `self.seat` is our real player index, which the perspective-relative
        owner codes do not carry — without it the tie-break would be a coin
        flip in the model and a fact in the engine.
        """
        if mv_me is None and mv_op is None:
            return 1
        if mv_me is None:
            return 2          # only we pass: they resolve first
        if mv_op is None:
            return 1

        src_me, src_op = mv_me[1], mv_op[1]
        dst_me = self._dest(src_me, mv_me[2])
        dst_op = self._dest(src_op, mv_op[2])

        chase_me = dst_me == src_op
        chase_op = dst_op == src_me
        if chase_me != chase_op:
            return 1 if chase_me else 2

        reinforce_me = self.owner[dst_me] == 1 if dst_me >= 0 else False
        reinforce_op = self.owner[dst_op] == 2 if dst_op >= 0 else False
        if reinforce_me != reinforce_op:
            return 1 if reinforce_me else 2

        a_me, a_op = int(self.army[src_me]), int(self.army[src_op])
        if a_me != a_op:
            return 1 if a_me < a_op else 2
        # Full tie: the engine gives it to player 0.
        return 1 if self.seat == 0 else 2


def _bfs(sim: _Sim, sources) -> np.ndarray:
    """Steps from `sources` over passable cells; `n` where unreachable."""
    dist = np.full(sim.n, sim.n, dtype=np.int32)
    q = deque()
    for s in sources:
        if s >= 0 and dist[s] != 0:
            dist[s] = 0
            q.append(s)
    W, H, passable = sim.W, sim.H, sim.passable
    while q:
        i = q.popleft()
        nd = dist[i] + 1
        r, c = divmod(i, W)
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            j = nr * W + nc
            if dist[j] > nd and passable[j]:
                dist[j] = nd
                q.append(j)
    return dist


class _Node:
    __slots__ = ("move", "visits", "value", "children", "untried")

    def __init__(self, move):
        self.move = move
        self.visits = 0
        self.value = 0.0
        self.children: list = []
        self.untried: list | None = None


class TacticalSearch:
    """MCTS as a *filter* on the core's move, never as its replacement."""

    def __init__(self, seat: int, params, rng: random.Random | None = None):
        self.seat = seat
        self.p = params
        self.rng = rng or random.Random(0xA17E)
        self.stats = SearchStats()

    # ------------------------------------------------------------- interface
    def improve(self, obs, heuristic, my_gen, enemy_gen, elapsed_ms: float):
        """Return the move to play: `heuristic`, or a searched improvement.

        `elapsed_ms` is what the turn has already spent, so the cap can shrink
        to fit the remainder of §08's 150 ms rather than assume a fresh move.
        """
        p = self.p
        if not p.mcts_enabled or my_gen is None:
            self.stats.turns_skipped += 1
            return heuristic

        budget = min(p.mcts_budget_ms, p.mcts_latency_cap_ms - elapsed_ms)
        if budget <= 2.0:
            self.stats.turns_skipped += 1
            self.stats.budget_exhausted += 1
            return heuristic

        t0 = time.perf_counter()
        deadline = t0 + budget / 1000.0

        sim = _Sim(obs, self.seat, p.deathtouch_from)
        sim.my_gen = my_gen[0] * obs.W + my_gen[1]
        sim.op_gen = -1 if enemy_gen is None else enemy_gen[0] * obs.W + enemy_gen[1]

        to_home = _bfs(sim, [sim.my_gen])
        to_away = _bfs(sim, [sim.op_gen]) if sim.op_gen >= 0 else None

        if not self._in_window(sim, to_home, to_away):
            self.stats.turns_skipped += 1
            return heuristic

        roots = self._root_moves(sim, heuristic, to_home, to_away, obs)
        if len(roots) < 2:
            self.stats.turns_skipped += 1
            return heuristic

        best = self._search(sim, roots, to_home, to_away, deadline)
        self.stats.turns_searched += 1
        self.stats.note((time.perf_counter() - t0) * 1000.0)
        if best is None or best == roots[0]:
            return heuristic
        self.stats.overrides += 1
        return self._encode(sim, best)

    # -------------------------------------------------------------- windowing
    def _in_window(self, sim: _Sim, to_home, to_away) -> bool:
        """Is this a turn worth searching at all?

        Measured, not assumed. The first version of this test fired on *any*
        enemy stack within `window` of our general — 175 of 580 turns in a
        sample game — and cost 47 points of winrate against cm_hunter
        (0.917 -> 0.450 over 200 paired games; widening the window to 12 took
        it to 0.183, which is the monotone dose-response that says the search
        itself was the cause and not the noise).

        Two things were wrong and both are fixed by asking a narrower question.

        1. **Overriding a stateful core desynchronises it.** blitz and boom
           write their memory as a side effect of choosing — `mem.stack`,
           `mem.chain_head`, `mem.chain_visited` all record the move the core
           *believes* it just made. Replace that move and the core plans the
           next turn from a board that never happened. `ensure_stack` and the
           chain-head check re-derive from the board and so self-heal, but
           `chain_visited` does not, and every override pays something.
        2. **A 12-ply horizon cannot price giving up the attack.** Interrupting
           a wave looks free inside the window and costs the game 300 turns
           later; mean game length went 399 -> 683 turns, which is what
           "stopped attacking" looks like from outside.

        So the search now fires only where being wrong is already the
        alternative: a stack that can actually take our general, a §07 touch
        threat, or a kill we can actually land. Everywhere else the core keeps
        its own move and its own state.
        """
        w = self.p.mcts_window
        gen_army = int(sim.army[sim.my_gen])

        enemy = np.flatnonzero((sim.owner == 2) & (sim.army >= 2))
        if enemy.size:
            near = enemy[to_home[enemy] <= w]
            if near.size:
                if sim.turn >= self.p.deathtouch_from:
                    return True          # §07: any 2-stack that arrives is lethal
                # Otherwise only a stack that beats the general's own garrison
                # is a question worth 40 ms; a raid is the core's business.
                if int(sim.army[near].max()) - 1 > gen_army:
                    return True

        if to_away is not None:
            mine = np.flatnonzero((sim.owner == 1) & (sim.army >= 2))
            if mine.size:
                close = mine[to_away[mine] <= w]
                if close.size:
                    if sim.turn >= self.p.deathtouch_from:
                        return True
                    if int(sim.army[close].max()) - 1 > int(sim.army[sim.op_gen]):
                        return True
        return False

    # ------------------------------------------------------------ root moves
    def _root_moves(self, sim: _Sim, heuristic, to_home, to_away, obs) -> list:
        """The shortlist. Index 0 is always the heuristic move, so a search
        that learns nothing returns exactly what the core wanted."""
        cap = self.p.mcts_root_moves
        moves = [self._decode(sim, heuristic)]
        seen = {moves[0]}

        def add(mv):
            if mv is None or mv in seen or not sim._valid(1, mv):
                return
            seen.add(mv)
            moves.append(mv)

        mine = sim.movable(1)
        if not mine.size:
            return moves

        # 1. Kill the biggest enemy stack near home, from any neighbour.
        enemy = np.flatnonzero((sim.owner == 2) & (sim.army >= 1))
        if enemy.size:
            near = enemy[to_home[enemy] <= self.p.mcts_window]
            if near.size:
                threat = int(near[np.argmax(sim.army[near])])
                for d in range(4):
                    src = self._back(sim, threat, d)
                    if src >= 0 and sim.owner[src] == 1:
                        add((0, src, d, 0))

        # 2. Put army *on* the general. The losses this search exists for are
        #    games where our general held 1-9 army against a 20-stack while our
        #    own army stood two cells away, so this move must be reachable.
        for d in range(4):
            src = self._back(sim, sim.my_gen, d)
            if src >= 0 and sim.owner[src] == 1 and sim.army[src] >= 2:
                add((0, src, d, 0))

        # 3. Walk our biggest stack one step homeward, and one step at them.
        for src in mine[np.argsort(-sim.army[mine])][:3]:
            src = int(src)
            for field_ in (to_home, to_away):
                if field_ is None:
                    continue
                here = int(field_[src])
                for d in range(4):
                    dst = sim._dest(src, d)
                    if dst >= 0 and sim.passable[dst] and field_[dst] < here:
                        add((0, src, d, 0))

        # 4. The finish, including the §07 deathtouch, if it is on the board.
        if sim.op_gen >= 0:
            for d in range(4):
                src = self._back(sim, sim.op_gen, d)
                if src >= 0 and sim.owner[src] == 1:
                    add((0, src, d, 0))

        return moves[:cap]

    def _back(self, sim: _Sim, target: int, d: int) -> int:
        """The cell that reaches `target` by moving in direction `d`."""
        r, c = divmod(target, sim.W)
        dr, dc = DIRECTIONS[d]
        sr, sc = r - dr, c - dc
        if not (0 <= sr < sim.H and 0 <= sc < sim.W):
            return -1
        return sr * sim.W + sc

    # ---------------------------------------------------------------- search
    def _search(self, sim: _Sim, roots: list, to_home, to_away, deadline):
        root = _Node(None)
        root.children = [_Node(mv) for mv in roots]
        iters = 0
        c_uct = self.p.mcts_c
        max_iters = self.p.mcts_max_iters

        while iters < max_iters:
            if time.perf_counter() >= deadline:
                break
            child = self._select(root, c_uct)
            value = self._rollout(sim, child.move, to_home, to_away, deadline)
            if value is None:            # deadline hit mid-rollout
                break
            child.visits += 1
            child.value += value
            root.visits += 1
            iters += 1

        self.stats.iterations += iters
        self.stats.max_iterations_one_turn = max(
            self.stats.max_iterations_one_turn, iters
        )
        if iters < len(roots):
            # Fewer iterations than root moves: `_select` hands the first
            # unvisited child every time, so below this every "best" is an
            # artefact of visit order rather than a comparison. Fall back.
            return None
        best = max(root.children, key=lambda n: n.value / n.visits)
        return best.move

    def _select(self, root: _Node, c_uct: float) -> _Node:
        import math

        unvisited = [n for n in root.children if n.visits == 0]
        if unvisited:
            return unvisited[0]
        log_n = math.log(max(1, root.visits))
        return max(
            root.children,
            key=lambda n: n.value / n.visits + c_uct * math.sqrt(log_n / n.visits),
        )

    # --------------------------------------------------------------- rollout
    def _rollout(self, sim: _Sim, first_move, to_home, to_away, deadline):
        """One playout. Returns None if the deadline landed mid-rollout —
        never a partial value, which would bias whichever child was unlucky."""
        s = sim.clone()
        depth = self.p.mcts_rollout_depth
        mv_me = first_move
        for ply in range(depth):
            if ply and (ply & 1) == 0 and time.perf_counter() >= deadline:
                return None
            mv_op = self._policy(s, 2, to_home, to_away)
            s.step(mv_me, mv_op)
            if s.winner >= 0:
                return _WIN if s.winner == 1 else _LOSS
            mv_me = self._policy(s, 1, to_home, to_away)
        return self._evaluate(s, to_home)

    def _policy(self, s: _Sim, side: int, to_home, to_away):
        """Cheap greedy playout policy, deliberately pessimistic for us.

        Them: march the largest stack down the distance field to our general,
        taking whatever is in the way. Us: the mirror, toward theirs, except
        that a stack already adjacent to a beatable enemy cell takes it.
        `epsilon` of the time either side moves at random, which is what stops
        the rollout being one deterministic line and the tree being a ladder.
        """
        cells = s.movable(side)
        if not cells.size:
            return None
        if self.rng.random() < self.p.mcts_epsilon:
            src = int(cells[self.rng.randrange(cells.size)])
            return (0, src, self.rng.randrange(4), 0)

        src = int(cells[np.argmax(s.army[cells])])
        if side == 2:
            field_ = to_home
        elif to_away is not None:
            field_ = to_away
        else:
            # No known enemy general. Walking our own stacks *down* `to_home`
            # would make the playout a turtle by construction and the value of
            # every root move the value of hiding — which is exactly the bias
            # that cost 47 points in the first measured version. March at the
            # threat instead: it is the only enemy object the search can see.
            field_ = self._threat_field(s, to_home)
            if field_ is None:
                return None
        here = int(field_[src])
        best_d, best_key = None, None
        attacking = int(s.army[src]) - 1
        for d in range(4):
            dst = s._dest(src, d)
            if dst < 0 or not s.passable[dst]:
                continue
            if s.owner[dst] != side and int(s.army[dst]) >= attacking:
                continue
            key = (int(field_[dst]), -int(s.army[dst]) if s.owner[dst] == side else 0)
            if best_key is None or key < best_key:
                best_key, best_d = key, d
        if best_d is None or (side == 1 and here == 0):
            return None
        return (0, src, best_d, 0)

    def _threat_field(self, s: _Sim, to_home):
        """Distance to the enemy stack nearest our general — the thing to kill."""
        enemy = np.flatnonzero((s.owner == 2) & (s.army >= 2))
        if not enemy.size:
            return None
        return _bfs(s, [int(enemy[np.argmin(to_home[enemy])])])

    def _evaluate(self, s: _Sim, to_home) -> float:
        """Bounded value in (-1, 1) from our seat's point of view.

        A rollout that reaches this did **not** resolve the fight, so the job
        here is to say who is better placed, not to reward a posture. The
        first version scored "army standing on our general" at half the total
        weight, which made every root move that carried army home look good
        and every attacking move look bad — the search preferred hiding, and
        the winrate said so.

        What survives is the part that is a fact rather than a preference:
        whether the threat is still alive, and who holds the board.
        """
        mine = s.owner == 1
        theirs = s.owner == 2
        my_army = int(s.army[mine].sum())
        op_army = int(s.army[theirs].sum())
        my_land = int(mine.sum())
        op_land = int(theirs.sum())

        # Danger, not comfort: an enemy stack near home that our general
        # cannot stop. Killing it and out-running it score the same, which is
        # the point — the search should be free to answer either way.
        gen_army = int(s.army[s.my_gen]) if s.my_gen >= 0 else 0
        enemy = np.flatnonzero(theirs & (s.army >= 2))
        danger = 0.0
        if enemy.size:
            close = enemy[to_home[enemy] <= 4]
            if close.size:
                worst = int(s.army[close].max())
                if worst - 1 > gen_army:
                    danger = min(1.0, (worst - gen_army) / float(worst + 1))

        army_edge = (my_army - op_army) / float(max(1, my_army + op_army))
        land_edge = (my_land - op_land) / float(max(1, my_land + op_land))
        raw = 0.45 * army_edge + 0.30 * land_edge - 0.25 * danger
        return max(-0.95, min(0.95, raw))

    # -------------------------------------------------------------- encoding
    def _decode(self, sim: _Sim, action):
        """Unified 5-int action -> internal `(0, idx, dir, split)`.

        A pass or a build has no internal form; both become `None`, which the
        search treats as "we do nothing this ply" rather than dropping the
        root child, so a core that wants to pass still gets its opinion
        represented at the root.
        """
        if action is None or action[0] != 0:
            return None
        return (0, action[1] * sim.W + action[2], action[3], action[4])

    def _encode(self, sim: _Sim, mv):
        if mv is None:
            return PASS_ACTION
        r, c = divmod(mv[1], sim.W)
        return (0, r, c, mv[2], mv[3])
