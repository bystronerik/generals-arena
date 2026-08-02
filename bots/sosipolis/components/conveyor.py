"""Shared clock / chain / prune helpers for purpose MCTS (not a live policy)."""
from __future__ import annotations

from components.army import (
    Action,
    bfs_dist,
    is_wall,
    move_action,
    neighbors,
)
from params import DIRECTIONS, Params, T_CASTLE, T_GENERAL


Cell = tuple[int, int]
ClockPhase = str  # "gather" | "wave"


def mod50_phase(turn: int, params: Params) -> ClockPhase:
    residue = turn % 50
    if params.GATHER_PHASE_LO <= residue <= params.GATHER_PHASE_HI:
        return "gather"
    return "wave"


def action_src_dst(action: Action) -> tuple[Cell, Cell] | None:
    if action[0] != 0:
        return None
    _, r, c, d, _ = action
    if not (0 <= d < len(DIRECTIONS)):
        return None
    dr, dc = DIRECTIONS[d]
    return (r, c), (r + dr, c + dc)


def chain_roots(obs, chain_head: Cell | None) -> list[Action]:
    """Leave-1 steps from the active chain head when movable."""
    if chain_head is None:
        return []
    r, c = chain_head
    if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
        return []
    out: list[Action] = []
    for nr, nc in neighbors(obs.H, obs.W, r, c):
        if is_wall(obs.type_grid, nr, nc):
            continue
        owner = obs.owner_grid[nr][nc]
        if owner != 1 and obs.army_grid[r][c] - 1 <= obs.army_grid[nr][nc]:
            continue
        out.append(move_action(r, c, nr, nc, 0))
    return out


def closes_on(obs, src: Cell, dst: Cell, goal: Cell | None) -> bool:
    if goal is None:
        return False
    before = abs(src[0] - goal[0]) + abs(src[1] - goal[1])
    after = abs(dst[0] - goal[0]) + abs(dst[1] - goal[1])
    return after < before


def closes_bfs(obs, src: Cell, dst: Cell, goal: Cell | None) -> bool:
    if goal is None:
        return False
    blocked = lambda rr, cc: is_wall(obs.type_grid, rr, cc)
    dist = bfs_dist(obs.H, obs.W, [goal], blocked)
    if src not in dist or dst not in dist:
        return closes_on(obs, src, dst, goal)
    return dist[dst] < dist[src]


def prune_by_clock(
    obs,
    moves: list[tuple[Action, float]],
    phase: ClockPhase,
    objective: Cell | None,
    muster: Cell | None,
) -> list[tuple[Action, float]]:
    """Hard filter: gather stays on own land toward muster; wave closes on objective."""
    kept: list[tuple[Action, float]] = []
    for action, prior in moves:
        pair = action_src_dst(action)
        if pair is None:
            continue
        src, dst = pair
        if not (0 <= dst[0] < obs.H and 0 <= dst[1] < obs.W):
            continue
        owner = obs.owner_grid[dst[0]][dst[1]]
        army = obs.army_grid[src[0]][src[1]]
        dest_army = obs.army_grid[dst[0]][dst[1]]

        if phase == "gather":
            if owner != 1:
                continue
            if muster is not None and not closes_on(obs, src, dst, muster):
                # Allow structure drain moves (src is general/castle) even if not closing.
                st = obs.type_grid[src[0]][src[1]]
                if st not in (T_GENERAL, T_CASTLE):
                    continue
            kept.append((action, prior))
            continue

        # wave
        if owner != 1:
            if army - 1 <= dest_army:
                continue
            # Capture allowed as routing side-effect when closing or adjacent win.
            if objective is not None and not closes_bfs(obs, src, dst, objective):
                # Still allow winning attacks on path-adjacent enemies.
                if owner != 2:
                    continue
            kept.append((action, prior))
            continue
        # Own-cell step: only if closing on objective.
        if objective is not None and closes_bfs(obs, src, dst, objective):
            kept.append((action, prior))
    return kept


def chain_prior(
    action: Action,
    chain_head: Cell | None,
    params: Params,
) -> float:
    pair = action_src_dst(action)
    if pair is None or chain_head is None:
        return 1.0
    src, _ = pair
    if src == chain_head:
        return params.CHAIN_CONTINUE_PRIOR
    return params.CHAIN_BREAK_PRIOR


def apply_chain_priors(
    moves: list[tuple[Action, float]],
    chain_head: Cell | None,
    params: Params,
) -> list[tuple[Action, float]]:
    return [
        (action, prior * chain_prior(action, chain_head, params))
        for action, prior in moves
    ]


def update_chain(state, action: Action) -> None:
    """Record chain head and last out-direction after a committed move."""
    if action[0] != 0:
        return
    pair = action_src_dst(action)
    if pair is None:
        return
    src, dst = pair
    state.chain_head = dst
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    state.last_out_dir[src] = (dr, dc)


def has_leave1_move(obs) -> bool:
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1 or obs.army_grid[r][c] <= 1:
                continue
            for nr, nc in neighbors(obs.H, obs.W, r, c):
                if is_wall(obs.type_grid, nr, nc):
                    continue
                return True
    return False


def opening_mask_ok(obs, state, dst: Cell, params: Params) -> bool:
    """Opening: stay near home half before flood; after flood, allow frontier push."""
    home = state.memory.own_general
    if home is None:
        return True
    if obs.owner_grid[dst[0]][dst[1]] == 2:
        return False
    # After flood start, only block enemy captures / walls already filtered.
    if obs.turn >= params.OPEN_FLOOD_START:
        return True
    mid_r = obs.H // 2
    mid_c = obs.W // 2
    home_d = abs(dst[0] - home[0]) + abs(dst[1] - home[1])
    center_d = abs(home[0] - mid_r) + abs(home[1] - mid_c)
    if home_d > center_d + 2:
        return False
    return True


def prune_opening(
    obs,
    state,
    moves: list[tuple[Action, float]],
    params: Params,
) -> list[tuple[Action, float]]:
    kept: list[tuple[Action, float]] = []
    for action, prior in moves:
        pair = action_src_dst(action)
        if pair is None:
            continue
        _, dst = pair
        if not opening_mask_ok(obs, state, dst, params):
            continue
        kept.append((action, prior))
    return kept


def resolve_objective(obs, state) -> Cell | None:
    """Fog frontier pre-contact → ContactMCTS probe → hunt → remembered general."""
    if state.memory.enemy_general is not None:
        return state.memory.enemy_general
    # Contact phase: ContactMCTS owns the probe waypoint via contact_commitment.
    if getattr(state, "phase", None) == "contact":
        commit = getattr(state, "contact_commitment", None)
        if commit is not None:
            return commit.macro.waypoint
        return _fog_frontier(obs, state)
    hunt = state.memory.hunt_cell
    if hunt is not None:
        return hunt
    return _fog_frontier(obs, state)


def resolve_muster(obs, state, tip: Cell | None = None) -> Cell | None:
    """Gather rally: strike tip, else chain head, else own general."""
    if tip is not None:
        return tip
    if state.chain_head is not None:
        r, c = state.chain_head
        if obs.owner_grid[r][c] == 1:
            return state.chain_head
    return state.memory.own_general


def _fog_frontier(obs, state) -> Cell | None:
    home = state.memory.own_general
    best = None
    best_s = -1.0
    for r in range(obs.H):
        for c in range(obs.W):
            if obs.owner_grid[r][c] != 1:
                continue
            for nr, nc in neighbors(obs.H, obs.W, r, c):
                if is_wall(obs.type_grid, nr, nc):
                    continue
                if obs.owner_grid[nr][nc] == 1:
                    continue
                if (nr, nc) in state.dead_pockets and obs.owner_grid[nr][nc] != 2:
                    continue
                s = state.sections.score_cell(nr, nc)
                if not state.memory.ever_seen[nr][nc]:
                    s += 0.5
                if home is not None:
                    # Prefer frontier away from home (Kubic pre-contact).
                    s += 0.02 * (abs(nr - home[0]) + abs(nc - home[1]))
                if s > best_s:
                    best_s = s
                    best = (nr, nc)
    return best


def prefer_chain_roots(
    obs,
    chain_head: Cell | None,
    scored: list[tuple[Action, float]],
    params: Params,
) -> list[tuple[Action, float]]:
    """Prefer chain-head continues that already passed scoring/prune.

    Never re-inject raw continues that failed the clock/objective filter — that
    locked the tip onto sideways snakes and delayed contact→sight.
    """
    continues = chain_roots(obs, chain_head)
    if not continues:
        return apply_chain_priors(scored, chain_head, params)
    cont_set = set(continues)
    preferred = [
        (a, p * params.CHAIN_CONTINUE_PRIOR)
        for a, p in scored
        if a in cont_set
    ]
    if preferred:
        return preferred
    # Chain cannot close on the objective this tick: allow a break from scored.
    return apply_chain_priors(scored, chain_head, params)
