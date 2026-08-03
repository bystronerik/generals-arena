"""Opening tempo gates (Kubic MEASURED beats) + SearchMCTS under opening mask."""
from __future__ import annotations

from components.army import Action, is_wall, largest_owned_stack, move_action, neighbors, pass_action
from components.clock import Deadline
from components.expand import expand_move
from params import Params


def opening_should_pass(obs, params: Params) -> bool:
    """Forced early passes only; keep staging moving toward flood."""
    t = obs.turn
    if t <= 2:
        return True
    return False


def opening_first_step(obs, state) -> Action | None:
    """t=3 style: general → best neutral neighbour (MCTS may refine later)."""
    home = state.memory.own_general
    if home is None:
        return None
    hr, hc = home
    if obs.owner_grid[hr][hc] != 1 or obs.army_grid[hr][hc] <= 1:
        return None
    best = None
    best_s = -1.0
    for nr, nc in neighbors(obs.H, obs.W, hr, hc):
        if is_wall(obs.type_grid, nr, nc):
            continue
        if obs.owner_grid[nr][nc] != 0:
            continue
        s = state.sections.score_cell(nr, nc)
        if not state.memory.ever_seen[nr][nc]:
            s += 0.5
        if s > best_s:
            best_s = s
            best = (nr, nc)
    if best is None:
        return None
    return move_action(hr, hc, best[0], best[1], 0)


def decide_opening(obs, state, search_mcts, deadline: Deadline, params: Params) -> Action | None:
    """Kubic §3: a script, not a search. Land by t=50 decides the game.

    The old path handed the opening to SearchMCTS, whose objective is the fog
    frontier *away from home*. That drew a one-cell-wide tendril across the
    map and left every capturable cell five to seven steps from the only stack
    that could take it. Expansion is a transit problem, not a search problem —
    see components/expand.py.
    """
    from components.search_mcts import SearchStats

    if obs.turn > params.OPEN_END:
        return None
    # Clear prior-turn UCT diag unless this tick runs SearchMCTS.
    search_mcts.stats = SearchStats()
    if opening_should_pass(obs, params):
        return pass_action()
    if obs.turn == 3:
        first = opening_first_step(obs, state)
        if first is not None:
            return first
    move = expand_move(obs, state, params)
    if move is not None:
        return move
    # Nothing to expand into: fall back to the search under the opening mask.
    state.clock_phase = "wave"
    return search_mcts.search(obs, state, deadline, opening=True)
