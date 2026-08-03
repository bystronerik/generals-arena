"""3970-logit action codec and legal masks.

Layout: channel-major over the padded ``21 × 21`` board::

    index = channel * 441 + row * 21 + col   # channels 0..8
    pass  = 3969

Direction order matches the competition protocol: up, down, left, right.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from memory import (
    TYPE_MOUNTAIN,
    TYPE_STRUCTURE_FOG,
    VisibleMemory,
)
from state import GameState
from transition import (
    BASE_COST,
    BUILD,
    DIRECTIONS,
    PASS_ACTION,
    PROXIMITY_DECAY,
    PROXIMITY_PENALTY,
    build_cost_grid,
)

PAD = 21
N_CELLS = PAD * PAD  # 441
N_CHANNELS = 9
N_ACTIONS = N_CHANNELS * N_CELLS + 1  # 3970
PASS_INDEX = N_ACTIONS - 1

CH_MOVE_FULL = (0, 1, 2, 3)  # up down left right, all-but-one
CH_MOVE_HALF = (4, 5, 6, 7)  # up down left right, half
CH_BUILD = 8


def encode_action(action: Sequence[int]) -> int:
    """Map a wire ``(pass, row, col, direction, split)`` to a logit index."""
    pass_f, row, col, direction, split = (int(x) for x in action)
    if pass_f == 1:
        return PASS_INDEX
    if pass_f == 2:
        return CH_BUILD * N_CELLS + int(row) * PAD + int(col)
    # Move.
    channel = int(direction) + (4 if int(split) == 1 else 0)
    return channel * N_CELLS + int(row) * PAD + int(col)


def decode_action(index: int) -> tuple[int, int, int, int, int]:
    """Map a logit index to a wire action tuple."""
    index = int(index)
    if index == PASS_INDEX:
        return (1, 0, 0, 0, 0)
    if not (0 <= index < PASS_INDEX):
        raise IndexError(f"action index out of range: {index}")
    channel, rem = divmod(index, N_CELLS)
    row, col = divmod(rem, PAD)
    if channel == CH_BUILD:
        return (2, row, col, 0, 0)
    if channel in CH_MOVE_FULL:
        return (0, row, col, channel, 0)
    if channel in CH_MOVE_HALF:
        return (0, row, col, channel - 4, 1)
    raise IndexError(f"invalid action channel: {channel}")


def _own_structures_from_obs(obs, memory: VisibleMemory) -> np.ndarray:
    """Own general + own castles (visible owner 1 and latched own general)."""
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    types = np.asarray(obs.type_grid, dtype=np.int32)
    own = owners == 1
    # Visible own castles and general, plus latched own general cell.
    structures = memory.own_general.copy()
    structures |= own & (types == 3)  # castle
    structures |= own & (types == 4)  # general (visible)
    structures |= own & memory.known_castle
    return structures


def live_build_cost(obs, memory: VisibleMemory) -> np.ndarray:
    """Exact live build-cost grid for the perspective player."""
    H, W = int(obs.H), int(obs.W)
    structures = _own_structures_from_obs(obs, memory)
    ownership = np.zeros((2, H, W), dtype=bool)
    ownership[0] = np.asarray(obs.owner_grid, dtype=np.int32) == 1
    # Fold every own structure into ``castles`` so
    # ``(castles | generals) & own`` equals the pricing mask.
    state = GameState(
        armies=np.asarray(obs.army_grid, dtype=np.int32),
        ownership=ownership,
        ownership_neutral=np.zeros((H, W), dtype=bool),
        generals=np.zeros((H, W), dtype=bool),
        castles=structures.astype(bool),
        mountains=memory.known_mountain,
        passable=~memory.known_mountain,
        general_positions=np.array([[0, 0], [-1, -1]], dtype=np.int32),
        time=int(obs.turn),
        winner=-1,
        pool_idx=0,
    )
    return build_cost_grid(state, 0)


def legal_mask(
    obs,
    memory: VisibleMemory,
    *,
    cost_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean mask of length 3970. Pass is always legal."""
    H, W = int(obs.H), int(obs.W)
    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    armies = np.asarray(obs.army_grid, dtype=np.int32)
    if cost_grid is None:
        cost_grid = live_build_cost(obs, memory)

    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[PASS_INDEX] = True

    # --- Moves ---
    for r in range(H):
        for c in range(W):
            if owners[r, c] != 1:
                continue
            src_army = int(armies[r, c])
            if src_army < 2:
                continue
            allow_half = src_army > 2
            for d in range(4):
                nr = r + int(DIRECTIONS[d, 0])
                nc = c + int(DIRECTIONS[d, 1])
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                dest_t = int(types[nr, nc])
                if dest_t in (TYPE_MOUNTAIN, TYPE_STRUCTURE_FOG):
                    continue
                full_idx = encode_action((0, r, c, d, 0))
                mask[full_idx] = True
                if allow_half:
                    half_idx = encode_action((0, r, c, d, 1))
                    mask[half_idx] = True

    # --- Builds ---
    for r in range(H):
        for c in range(W):
            if owners[r, c] != 1:
                continue
            if memory.own_general[r, c] or memory.known_enemy_general[r, c]:
                continue
            if memory.known_castle[r, c] or memory.known_mountain[r, c]:
                continue
            if not memory.known_passable_base[r, c]:
                # Unexplored cell: not yet proven plain. Refuse build.
                continue
            cost = int(cost_grid[r, c])
            if int(armies[r, c]) < cost:
                continue
            mask[encode_action((2, r, c, 0, 0))] = True

    return mask


def iter_legal_actions(
    obs, memory: VisibleMemory, *, cost_grid: Optional[np.ndarray] = None
) -> Iterable[tuple[int, int, int, int, int]]:
    mask = legal_mask(obs, memory, cost_grid=cost_grid)
    for idx in np.flatnonzero(mask):
        yield decode_action(int(idx))


def action_effects_match_pass(
    state: GameState, player_idx: int, action: Sequence[int]
) -> bool:
    """True when ``action`` matches a double-pass board result (invalid or pass).

    Compares armies, ownership, and castles after one joint step with the enemy
    passing. Growth and time advance on both paths, so a no-op move equals pass.
    """
    from transition import transition

    both_pass = np.stack([PASS_ACTION, PASS_ACTION]).astype(np.int32)
    with_action = both_pass.copy()
    with_action[player_idx] = np.asarray(action, dtype=np.int32)
    after_pass, _ = transition(state, both_pass)
    after_act, _ = transition(state, with_action)
    return (
        np.array_equal(after_pass.armies, after_act.armies)
        and np.array_equal(after_pass.ownership, after_act.ownership)
        and np.array_equal(after_pass.castles, after_act.castles)
    )


# Re-export cost constants for tests / measurement scripts.
__all__ = [
    "PAD",
    "N_CELLS",
    "N_CHANNELS",
    "N_ACTIONS",
    "PASS_INDEX",
    "CH_BUILD",
    "encode_action",
    "decode_action",
    "live_build_cost",
    "legal_mask",
    "iter_legal_actions",
    "action_effects_match_pass",
    "BASE_COST",
    "PROXIMITY_PENALTY",
    "PROXIMITY_DECAY",
    "BUILD",
]
