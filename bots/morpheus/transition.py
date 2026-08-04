"""Bundle-safe competition transition.

Mirrors competition-module composition:

    builds first → deathtouch-wrapped game.step

No competition-module import. Truncation at turn 1200 is a driver check
(``at_truncation``), not part of ``transition`` — same as ``make_transition``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from state import GameInfo, GameState

# Action encoding — matches protocol and engine.
PASS_ACTION = np.array([1, 0, 0, 0, 0], dtype=np.int32)
BUILD = 2
DIRECTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)

# Competition modifiers (GeneralsEnv mode="competition").
BASE_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2
_RADIUS = (PROXIMITY_PENALTY - 1) // PROXIMITY_DECAY
DEATHTOUCH_TURN = 800
TRUNCATION_TURN = 1200


def at_truncation(state: GameState) -> bool:
    """Driver-boundary hard draw: true when time has reached the cap."""
    return state.winner < 0 and state.time >= TRUNCATION_TURN


def build_cost_grid(state: GameState, player_idx: int) -> np.ndarray:
    """(H, W) live castle price for ``player_idx``."""
    H, W = state.armies.shape
    own = state.ownership[player_idx]
    structures = ((state.castles | state.generals) & own).astype(np.int32)
    padded = np.pad(structures, _RADIUS)

    cost = np.full((H, W), BASE_COST, dtype=np.int32)
    for di in range(-_RADIUS, _RADIUS + 1):
        for dj in range(-_RADIUS, _RADIUS + 1):
            surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                shifted = padded[
                    _RADIUS + di : _RADIUS + di + H,
                    _RADIUS + dj : _RADIUS + dj + W,
                ]
                cost = cost + surcharge * shifted
    return cost


def _apply_one_build(
    state: GameState,
    player_idx: int,
    action: np.ndarray,
    *,
    cost_grid: Optional[np.ndarray] = None,
) -> GameState:
    action = np.asarray(action, dtype=np.int32)
    if int(action[0]) != BUILD:
        return state

    H, W = state.armies.shape
    r, c = int(action[1]), int(action[2])
    in_bounds = 0 <= r < H and 0 <= c < W
    rs = min(max(r, 0), H - 1)
    cs = min(max(c, 0), W - 1)

    owns = bool(state.ownership[player_idx, rs, cs])
    plain = (not bool(state.generals[rs, cs])) and (not bool(state.castles[rs, cs]))
    grid = cost_grid if cost_grid is not None else build_cost_grid(state, player_idx)
    cost = int(grid[rs, cs])
    affords = int(state.armies[rs, cs]) >= cost
    alive = state.winner < 0
    valid = in_bounds and owns and plain and affords and alive

    if not valid:
        return state

    armies = state.armies.copy()
    castles = state.castles.copy()
    armies[rs, cs] -= cost
    castles[rs, cs] = True
    return state._replace(armies=armies, castles=castles)


def apply_build_actions(
    state: GameState, actions: np.ndarray
) -> tuple[GameState, np.ndarray]:
    """Resolve both builds; rewrite every BUILD action to pass."""
    actions = np.asarray(actions, dtype=np.int32)
    for player_idx in (0, 1):
        if int(actions[player_idx, 0]) == BUILD:
            state = _apply_one_build(state, player_idx, actions[player_idx])

    out = actions.copy()
    for i in range(2):
        if int(out[i, 0]) == BUILD:
            out[i] = PASS_ACTION
    return state, out


def _determine_move_order(state: GameState, actions: np.ndarray) -> int:
    """Chase > reinforce > smaller source; seat 0 on full tie; only-P0-pass → 1."""
    actions = np.asarray(actions, dtype=np.int32)
    pass_0, row_0, col_0, dir_0, _ = (int(x) for x in actions[0])
    pass_1, row_1, col_1, dir_1, _ = (int(x) for x in actions[1])

    # Bitwise form matches JAX ``pass_0 & ~pass_1`` on int32 action fields.
    only_p0_passes = bool(np.int32(pass_0) & ~np.int32(pass_1))

    di_0, dj_0 = row_0 + int(DIRECTIONS[dir_0, 0]), col_0 + int(DIRECTIONS[dir_0, 1])
    di_1, dj_1 = row_1 + int(DIRECTIONS[dir_1, 0]), col_1 + int(DIRECTIONS[dir_1, 1])

    p0_chasing = di_0 == row_1 and dj_0 == col_1
    p1_chasing = di_1 == row_0 and dj_1 == col_0

    # Numpy negative indices wrap like JAX; positive OOB is avoided by protocol
    # actions. Pass [1,0,0,0,0] yields di=-1 and relies on wrap for reinforce.
    p0_reinforcing = bool(state.ownership[0, di_0, dj_0])
    p1_reinforcing = bool(state.ownership[1, di_1, dj_1])

    army_0 = int(state.armies[row_0, col_0])
    army_1 = int(state.armies[row_1, col_1])

    p1_wins_by_chase = p1_chasing and not p0_chasing
    tie_on_chase = p0_chasing == p1_chasing
    p1_wins_by_reinforce = tie_on_chase and p1_reinforcing and not p0_reinforcing
    tie_on_reinforce = p0_reinforcing == p1_reinforcing
    p1_wins_by_army = tie_on_chase and tie_on_reinforce and (army_1 < army_0)

    if p1_wins_by_chase or p1_wins_by_reinforce or p1_wins_by_army or only_p0_passes:
        return 1
    return 0


def _army_to_move(source_army: int, split_army: int) -> int:
    raw = source_army // 2 if split_army == 1 else source_army - 1
    return max(0, min(raw, source_army - 1))


def _apply_move(
    state: GameState,
    player_idx: int,
    si: int,
    sj: int,
    di: int,
    dj: int,
    army_to_move: int,
) -> GameState:
    armies = state.armies.copy()
    ownership = state.ownership.copy()
    ownership_neutral = state.ownership_neutral.copy()
    winner = state.winner

    target_owner_0 = bool(ownership[0, di, dj])
    target_owner_1 = bool(ownership[1, di, dj])
    target_neutral = bool(ownership_neutral[di, dj])
    moving_to_own = (player_idx == 0 and target_owner_0) or (
        player_idx == 1 and target_owner_1
    )

    if moving_to_own:
        armies[di, dj] += army_to_move
        armies[si, sj] -= army_to_move
    else:
        target_army = int(armies[di, dj])
        attacker_wins = army_to_move > target_army
        remaining = abs(target_army - army_to_move)
        armies[di, dj] = remaining
        armies[si, sj] -= army_to_move

        if attacker_wins:
            ownership[player_idx, di, dj] = True
            if target_owner_0 and player_idx == 1:
                ownership[0, di, dj] = False
            if target_owner_1 and player_idx == 0:
                ownership[1, di, dj] = False
            if target_neutral:
                ownership_neutral[di, dj] = False

            if bool(state.generals[di, dj]) and not moving_to_own:
                winner = player_idx

    return state._replace(
        armies=armies,
        ownership=ownership,
        ownership_neutral=ownership_neutral,
        winner=winner,
    )


def _execute_move(
    state: GameState,
    player_idx: int,
    si: int,
    sj: int,
    direction: int,
    split_army: int,
) -> GameState:
    H, W = state.armies.shape
    in_bounds = 0 <= si < H and 0 <= sj < W
    di = si + int(DIRECTIONS[direction, 0])
    dj = sj + int(DIRECTIONS[direction, 1])
    dest_in_bounds = 0 <= di < H and 0 <= dj < W

    if not (in_bounds and dest_in_bounds):
        return state

    owns_source = bool(state.ownership[player_idx, si, sj])
    source_army = int(state.armies[si, sj])
    army_to_move = _army_to_move(source_army, split_army)
    valid = (
        owns_source
        and army_to_move > 0
        and bool(state.passable[di, dj])
    )
    if not valid:
        return state
    return _apply_move(state, player_idx, si, sj, di, dj, army_to_move)


def execute_action(state: GameState, player_idx: int, action: np.ndarray) -> GameState:
    action = np.asarray(action, dtype=np.int32)
    pass_turn, si, sj, direction, split_army = (int(x) for x in action)
    if pass_turn == 1:
        return state
    return _execute_move(state, player_idx, si, sj, direction, split_army)


def _transfer_loser_cells_to_winner(state: GameState) -> GameState:
    winner_idx = int(state.winner)
    loser_idx = 1 - winner_idx
    ownership = state.ownership.copy()
    ownership[winner_idx] = ownership[winner_idx] | ownership[loser_idx]
    ownership[loser_idx] = np.zeros_like(ownership[loser_idx], dtype=bool)
    ownership_neutral = state.ownership_neutral & ~state.ownership[loser_idx]
    return state._replace(ownership=ownership, ownership_neutral=ownership_neutral)


def global_update(state: GameState) -> GameState:
    """Growth at the post-increment time: structures on even ticks, all@50."""
    time = state.time
    armies = state.armies.copy()

    if time % 50 == 0:
        armies = armies + state.ownership[0].astype(np.int32) + state.ownership[1].astype(
            np.int32
        )

    if time % 2 == 0:
        structure_mask = (state.generals | state.castles).astype(np.int32)
        armies = (
            armies
            + structure_mask * state.ownership[0].astype(np.int32)
            + structure_mask * state.ownership[1].astype(np.int32)
        )

    return state._replace(armies=armies)


def get_info(state: GameState) -> GameInfo:
    armies = state.armies
    ownership = state.ownership
    return GameInfo(
        army=np.array(
            [
                int(np.sum(armies * ownership[0])),
                int(np.sum(armies * ownership[1])),
            ],
            dtype=np.int64,
        ),
        land=np.array(
            [int(np.sum(ownership[0])), int(np.sum(ownership[1]))],
            dtype=np.int64,
        ),
        is_done=state.winner >= 0,
        winner=state.winner,
        time=state.time,
    )


def step_base(state: GameState, actions: np.ndarray) -> tuple[GameState, GameInfo]:
    """Base game.step without build or deathtouch modifiers."""
    actions = np.asarray(actions, dtype=np.int32)
    done_before = state.winner >= 0

    first = _determine_move_order(state, actions)
    second = 1 - first
    state = execute_action(state, first, actions[first])
    state = execute_action(state, second, actions[second])

    if not done_before:
        state = state._replace(time=state.time + 1)

    if state.winner >= 0:
        state = _transfer_loser_cells_to_winner(state)
    else:
        state = global_update(state)

    return state, get_info(state)


def _executes_onto_general(
    state: GameState, player_idx: int, action: np.ndarray
) -> bool:
    """True iff a valid move from this state lands on the enemy general."""
    action = np.asarray(action, dtype=np.int32)
    pass_turn, si, sj, direction, split_army = (int(x) for x in action)
    H, W = state.armies.shape

    in_bounds = 0 <= si < H and 0 <= sj < W
    di = si + int(DIRECTIONS[direction, 0])
    dj = sj + int(DIRECTIONS[direction, 1])
    dest_in_bounds = 0 <= di < H and 0 <= dj < W

    if not (in_bounds and dest_in_bounds):
        return False

    owns_source = bool(state.ownership[player_idx, si, sj])
    source_army = int(state.armies[si, sj])
    army_to_move = _army_to_move(source_army, split_army)
    valid = (
        owns_source
        and army_to_move > 0
        and bool(state.passable[di, dj])
    )
    g = state.general_positions[1 - player_idx]
    return pass_turn == 0 and valid and di == int(g[0]) and dj == int(g[1])


def step_deathtouch(
    state: GameState,
    actions: np.ndarray,
    turn: int = DEATHTOUCH_TURN,
) -> tuple[GameState, GameInfo]:
    """Deathtouch wrapper around ``step_base`` (builds already rewritten)."""
    actions = np.asarray(actions, dtype=np.int32)
    active = state.winner < 0 and state.time >= turn

    first = _determine_move_order(state, actions)
    t_first = _executes_onto_general(state, first, actions[first])
    mid = execute_action(state, first, actions[first])
    t_second = _executes_onto_general(mid, 1 - first, actions[1 - first])

    if first == 0:
        touch = np.array([t_first, t_second], dtype=bool)
    else:
        touch = np.array([t_second, t_first], dtype=bool)
    touch = touch & active

    new_state, _ = step_base(state, actions)

    both_captured = (
        state.winner < 0 and mid.winner >= 0 and new_state.winner != mid.winner
    )
    both = bool((touch[0] and touch[1]) or both_captured)
    one = bool((touch[0] ^ touch[1]) and not both_captured)
    toucher = 0 if touch[0] else 1

    if both:
        winner = -1
    elif one:
        winner = toucher
    else:
        winner = new_state.winner

    new_state = new_state._replace(winner=winner)

    need_transfer = one and winner >= 0 and state.winner < 0
    if need_transfer:
        new_state = _transfer_loser_cells_to_winner(new_state)

    info = get_info(new_state)
    info = info._replace(is_done=info.is_done or both)
    return new_state, info


def transition(
    state: GameState, actions: np.ndarray
) -> tuple[GameState, GameInfo]:
    """Exact competition transition: builds, then deathtouch-wrapped step.

    Truncation is not applied here — call ``at_truncation`` at the driver.
    Pass and move turns skip castle-cost grids. Turns before deathtouch call
    ``step_base`` directly.
    """
    actions = np.asarray(actions, dtype=np.int32)
    has_build = int(actions[0, 0]) == BUILD or int(actions[1, 0]) == BUILD
    if has_build:
        state, actions = apply_build_actions(state, actions)
    if state.winner < 0 and state.time >= DEATHTOUCH_TURN:
        return step_deathtouch(state, actions, DEATHTOUCH_TURN)
    return step_base(state, actions)
