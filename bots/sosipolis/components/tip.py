"""Assault tip mass: select, path cost, exclusive feed (Kubic-style waves)."""
from __future__ import annotations

from components.army import (
    Action,
    bfs_dist,
    march_dist,
    gather_toward,
    is_wall,
    move_action,
    neighbors,
    step_toward,
)
from params import Params


Cell = tuple[int, int]


def total_owned_army(obs) -> int:
    total = 0
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] == 1:
                total += obs.army_grid[r][c]
    return total


def _blocked(obs):
    return lambda r, c: is_wall(obs.type_grid, r, c)


def path_cells_to_goal(obs, tip: Cell, goal: Cell) -> list[Cell]:
    """Greedy shortest-path cells from tip to goal (inclusive of goal)."""
    blocked = _blocked(obs)
    dist = bfs_dist(obs.H, obs.W, [goal], blocked)
    if tip not in dist:
        return []
    path: list[Cell] = []
    cur = tip
    guard = obs.H * obs.W + 2
    while cur != goal and guard > 0:
        guard -= 1
        path.append(cur)
        best = None
        best_d = dist[cur]
        for nr, nc in neighbors(obs.H, obs.W, *cur):
            if (nr, nc) not in dist:
                continue
            if dist[(nr, nc)] < best_d:
                best_d = dist[(nr, nc)]
                best = (nr, nc)
        if best is None:
            break
        cur = best
    path.append(goal)
    return path


def path_finish_need(obs, tip: Cell, goal: Cell, params: Params) -> int:
    """Army the tip must spend to walk the path and capture the general."""
    if obs.turn >= params.DEATHTOUCH_TURN:
        base = 1  # one unit executes; leave-1 needs tip army >= 2
    else:
        base = params.FINISH_MARGIN
        if obs.owner_grid[goal[0]][goal[1]] == 2:
            base += obs.army_grid[goal[0]][goal[1]]

    path = path_cells_to_goal(obs, tip, goal)
    if not path:
        blocked = _blocked(obs)
        dist = bfs_dist(obs.H, obs.W, [goal], blocked)
        d = dist.get(tip, obs.H + obs.W)
        return base + params.STRIKE_PATH_BUFFER * d

    cost = base
    # Skip tip cell; charge each step onto the next cell.
    for cell in path[1:]:
        r, c = cell
        owner = obs.owner_grid[r][c]
        if cell == goal:
            continue  # already in base
        if owner == 2:
            cost += obs.army_grid[r][c] + 1
        elif owner == 0:
            cost += 1
        # owned path is free
    hops = max(0, len(path) - 1)
    cost += params.STRIKE_PATH_BUFFER * hops
    cost += params.STRIKE_REGEN_SLACK * (hops // 2)
    return cost


def tip_mass_target(obs, tip: Cell, goal: Cell, params: Params) -> int:
    """March bar: contested path cost and absolute min tip (Kubic ~50–55)."""
    path_need = path_finish_need(obs, tip, goal, params)
    return max(path_need, params.STRIKE_MIN_TIP)


def tip_feed_target(obs, tip: Cell, goal: Cell, params: Params) -> int:
    """Exclusive-feed bar: also pull toward STRIKE_TIP_ARMY_FRAC of owned army."""
    frac_need = int(params.STRIKE_TIP_ARMY_FRAC * total_owned_army(obs))
    return max(tip_mass_target(obs, tip, goal, params), frac_need)


def finish_cost_exact(obs, tip: Cell, goal: Cell, params: Params) -> int:
    """Army the tip needs to walk the path and take the general — no slack.

    `path_finish_need` adds `STRIKE_PATH_BUFFER` a hop plus a regen allowance.
    That is right for planning a long march and wrong for deciding whether a
    kill two steps away is already on: it priced a 2-hop finish at ~10 and the
    tip sat feeding, seven army against a general holding two, until the
    general had grown past it.
    """
    if obs.turn >= params.DEATHTOUCH_TURN:
        base = 1
    else:
        base = params.FINISH_MARGIN
        if obs.owner_grid[goal[0]][goal[1]] == 2:
            base += obs.army_grid[goal[0]][goal[1]]

    path = path_cells_to_goal(obs, tip, goal)
    if not path or path[-1] != goal:
        return 1 << 20
    cost = base
    for cell in path[1:]:
        if cell == goal:
            continue
        r, c = cell
        owner = obs.owner_grid[r][c]
        if owner == 2:
            cost += obs.army_grid[r][c] + 1
        elif owner == 0:
            cost += 1
        # own land is free — the stack absorbs it
    return cost


def can_finish_now(obs, tip: Cell | None, goal: Cell | None, params: Params) -> bool:
    """True when this stack, as it stands, can walk in and take the general."""
    if tip is None or goal is None:
        return False
    army = obs.army_grid[tip[0]][tip[1]]
    if army <= 1:
        return False
    return army - 1 >= finish_cost_exact(obs, tip, goal, params)


def tip_is_ready(obs, tip: Cell | None, goal: Cell, params: Params) -> bool:
    """True when the tip can march (path + operating mass). Frac is feed-only."""
    if tip is None:
        return False
    army = obs.army_grid[tip[0]][tip[1]]
    if army <= 1:
        return False
    return army - 1 >= tip_mass_target(obs, tip, goal, params)


def tip_below_sight_floor(obs, tip: Cell | None, params: Params) -> bool:
    """Hard §2.6 gate: tip army below TIP_AT_SIGHT_FLOOR needs exclusive feed."""
    if tip is None:
        return True
    return obs.army_grid[tip[0]][tip[1]] < params.TIP_AT_SIGHT_FLOOR


def tip_needs_feed(obs, tip: Cell | None, goal: Cell, params: Params) -> bool:
    if tip is None:
        return False
    army = obs.army_grid[tip[0]][tip[1]]
    if army <= 1:
        return True
    if army < params.TIP_AT_SIGHT_FLOOR:
        return True
    return army - 1 < tip_feed_target(obs, tip, goal, params)


def touches_enemy(obs, r: int, c: int) -> bool:
    return any(
        obs.owner_grid[nr][nc] == 2
        for nr, nc in neighbors(obs.H, obs.W, r, c)
    )


def select_mass_tip(
    obs,
    goal: Cell,
    params: Params,
    cached: Cell | None,
    cached_turn: int,
    prefer_front: bool = False,
) -> Cell | None:
    """Mass-first tip on a path to goal; do not lock a thin tip.

    `prefer_front` restricts the pool to stacks already touching enemy land.
    Kubic's attack source is the largest such stack on 94.5-95.6% of its
    attacks; ours starts a wave adjacent to the enemy only 43% of the time,
    so the mass we build is behind the front and pays the walk again.
    """
    blocked = _blocked(obs)
    dist = bfs_dist(obs.H, obs.W, [goal], blocked)
    max_d = params.STRIKE_TIP_MAX_DIST

    candidates: list[tuple[int, int, int, int]] = []
    front: list[tuple[int, int, int, int]] = []
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                continue
            if (r, c) not in dist:
                continue
            d = dist[(r, c)]
            army = obs.army_grid[r][c]
            candidates.append((army, -d, r, c))
            if prefer_front and touches_enemy(obs, r, c):
                front.append((army, -d, r, c))
    if prefer_front and front:
        candidates = front
    if not candidates:
        return None
    candidates.sort(reverse=True)

    # Prefer mass within max dist; else best mass overall.
    near = [c for c in candidates if -c[1] <= max_d]
    pool = near if near else candidates

    best_army, neg_d, br, bc = pool[0]
    best = (br, bc)

    if (
        cached is not None
        and obs.owner_grid[cached[0]][cached[1]] == 1
        and obs.army_grid[cached[0]][cached[1]] > 1
        and cached in dist
        and obs.turn - cached_turn < params.STRIKE_TIP_HOLD
    ):
        cached_army = obs.army_grid[cached[0]][cached[1]]
        # Keep hold only if tip is already substantial and not dominated.
        min_hold = max(params.STRIKE_MIN_TIP // 2, 12)
        if cached_army >= min_hold and cached_army * 5 >= best_army * 4:
            return cached

    return best


def attack_move(obs, tip: Cell | None, goal: Cell | None, params: Params):
    """One forward attack from the front stack — Kubic's engagement rule.

    Three measured details of Kubic's attack, which only mean anything
    together (see 055):

    - Source is the largest stack already adjacent to enemy land (94.5-95.6%);
      that is `select_mass_tip(prefer_front=True)`, upstream of this.
    - Destination is the enemy neighbour minimising BFS distance to the
      believed general (~87% pre-sight, ~93% post-sight).
    - It does not attack at a losing margin (`moved > defender` in 95-97%,
      `P(attack | negative margin) ~= 1%`; ours measured 3.1%, and on seed 1
      alone that threw away 520 army).

    Forward only: the step must reduce distance to the objective. Waves that
    walk back are the behaviour this whole line of work started from.

    Being a rule rather than an MCTS root also removes the decision from the
    deadline: the same position now yields the same attack whatever the search
    budget happened to be that tick.
    """
    if tip is None or goal is None:
        return None
    army = obs.army_grid[tip[0]][tip[1]]
    if army <= 1:
        return None
    blocked = _blocked(obs)
    dist = bfs_dist(obs.H, obs.W, [goal], blocked)
    here = dist.get(tip)
    if here is None:
        return None
    best: Cell | None = None
    best_key = None
    for nr, nc in neighbors(obs.H, obs.W, *tip):
        if obs.owner_grid[nr][nc] != 2:
            continue
        if army - 1 <= obs.army_grid[nr][nc]:
            continue  # losing margin — Kubic does this ~1% of the time
        d = dist.get((nr, nc))
        if d is None:
            continue
        if params.CONTACT_ATTACK_FORWARD_ONLY and d >= here:
            continue
        key = (d, obs.army_grid[nr][nc], nr, nc)
        if best_key is None or key < best_key:
            best_key = key
            best = (nr, nc)
    if best is None:
        return None
    return move_action(tip[0], tip[1], best[0], best[1], 0)


def strike_march(obs, tip: Cell | None, goal: Cell | None, params: Params):
    """Deterministic near-BFS march to the remembered general.

    Kubic post-sight is not a search: it marches the remembered cell at a
    median path overhead of 11-20% and its first wave kills ~85%. Ours ran
    StrikeMCTS, which spent its entire STRIKE_MARCH_BUDGET_MS on every single
    strike tick (measured: median 55 ms, max 56, against a 55 ms budget). An
    iteration count that depends on machine load makes the same position play
    differently from run to run — the reason a no-op computation change moved
    a game in the coverage sweep.

    Cheapest route rather than shortest (`march_dist`), and never a step into
    a cell this stack cannot take.
    """
    if tip is None or goal is None:
        return None
    army = obs.army_grid[tip[0]][tip[1]]
    if army <= 1:
        return None
    blocked = _blocked(obs)
    dist = march_dist(obs, [goal], blocked, params.MARCH_COST_CAP)
    here = dist.get(tip)
    if here is None:
        return None
    best: Cell | None = None
    best_key = None
    for nr, nc in neighbors(obs.H, obs.W, *tip):
        d = dist.get((nr, nc))
        if d is None or d >= here:
            continue
        if obs.owner_grid[nr][nc] == 2 and army - 1 <= obs.army_grid[nr][nc]:
            continue
        key = (d, nr, nc)
        if best_key is None or key < best_key:
            best_key = key
            best = (nr, nc)
    if best is None:
        return None
    return move_action(tip[0], tip[1], best[0], best[1], 0)


def feed_tip_action(obs, tip: Cell) -> Action | None:
    """One exclusive feed move: gather or step the largest off-tip stack onto tip."""
    blocked = _blocked(obs)
    gact = gather_toward(obs, tip, blocked)
    if gact is not None:
        return gact
    stacks: list[tuple[int, int, int]] = []
    for r in range(obs.H):
        for c in range(obs.W):
            if (r, c) == tip:
                continue
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                continue
            stacks.append((obs.army_grid[r][c], r, c))
    stacks.sort(reverse=True)
    for _, r, c in stacks:
        act = step_toward(obs, (r, c), tip, blocked)
        if act is not None:
            return act
    return None


def path_feed_roots(obs, tip: Cell, feed_bonus: float) -> list[tuple[Action, float]]:
    """Root set used by MCTS while the tip is underfed."""
    blocked = _blocked(obs)
    out: list[tuple[Action, float]] = []
    gact = gather_toward(obs, tip, blocked)
    if gact is not None:
        out.append((gact, feed_bonus))
    stacks: list[tuple[int, int, int]] = []
    for r in range(obs.H):
        for c in range(obs.W):
            if (r, c) == tip:
                continue
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                continue
            stacks.append((obs.army_grid[r][c], r, c))
    stacks.sort(reverse=True)
    for army, r, c in stacks[:8]:
        act = step_toward(obs, (r, c), tip, blocked)
        if act is None:
            continue
        out.append((act, feed_bonus * 0.9 + army * 0.3))
    return out
