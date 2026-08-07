"""Build the Morpheus ``49 × 21 × 21`` observation tensor.

Perspective-relative: both seats use the same builder. Belief planes come from
an injected :class:`BeliefSummary` (Part 05 owns particle aggregation).

Specification choices locked by Part 03 (see observation-tensor.md):

- ``belief_enemy_army_mean`` = ``army_value(raw particle mean)``.
- ``belief_enemy_army_std`` = ``army_value(raw particle std)``.
- Previous-action planes encode the *perspective player's* last action only.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

import numpy as np

from memory import (
    OWNER_ENEMY,
    OWNER_ME,
    OWNER_NEUTRAL,
    TYPE_FOG,
    TYPE_STRUCTURE_FOG,
    VisibleMemory,
    update_memory,
)

Array = np.ndarray

PAD = 21
N_PLANES = 49
ARMY_SCALE = 4096  # initial guess; model manifest records replacements
TRUNCATION_TURN = 1200
DEATHTOUCH_TURN = 800
LN2 = math.log(2.0)

# Plane indices (0-based; spec table is 1-based).
P_BOARD_MASK = 0
P_VISIBLE_NOW = 1
P_FOG_NONSTRUCTURE = 2
P_FOG_STRUCTURE = 3
P_KNOWN_MOUNTAIN = 4
P_KNOWN_PASSABLE = 5
P_KNOWN_CASTLE = 6
P_OWN_GENERAL = 7
P_KNOWN_ENEMY_GENERAL = 8
P_OWNED_NOW = 9
P_ENEMY_VISIBLE = 10
P_NEUTRAL_VISIBLE = 11
P_OWNED_ARMY = 12
P_ENEMY_ARMY_VISIBLE = 13
P_EVER_VISIBLE = 14
P_SIGHT_AGE = 15
P_REMEMBERED_OWNED = 16
P_REMEMBERED_ENEMY = 17
P_REMEMBERED_NEUTRAL = 18
P_REMEMBERED_ENEMY_ARMY = 19
P_REMEMBERED_OWN_CASTLE = 20
P_REMEMBERED_ENEMY_CASTLE = 21
P_BELIEF_ENEMY_OWNER = 22
P_BELIEF_ENEMY_ARMY_MEAN = 23
P_BELIEF_ENEMY_ARMY_STD = 24
P_BELIEF_ENEMY_GENERAL = 25
P_BELIEF_ENEMY_CASTLE_OWNER = 26
P_BELIEF_ENEMY_VISIBILITY = 27
P_BELIEF_OWNER_ENTROPY = 28
P_PREV_MOVE_SOURCE = 29
P_PREV_MOVE_DEST = 30
P_PREV_MOVE_KIND = 31
P_PREV_BUILD_CELL = 32
P_ROW_COORD = 33
P_COL_COORD = 34
P_ROW_FROM_GENERAL = 35
P_COL_FROM_GENERAL = 36
P_TURN_FRACTION = 37
P_PRE_DEATHTOUCH = 38
P_DEATHTOUCH_ACTIVE = 39
P_STRUCTURE_GROWTH_NEXT = 40
P_BULK_GROWTH_COUNTDOWN = 41
P_OWN_LAND_FRACTION = 42
P_ENEMY_LAND_FRACTION = 43
P_OWN_ARMY_TOTAL = 44
P_ENEMY_ARMY_TOTAL = 45
P_LAND_MARGIN = 46
P_ARMY_MARGIN = 47
P_BELIEF_ESS = 48

PLANE_NAMES: tuple[str, ...] = (
    "board_mask",
    "visible_now",
    "fog_nonstructure_now",
    "fog_structure_now",
    "known_mountain",
    "known_passable_base",
    "known_castle",
    "own_general",
    "known_enemy_general",
    "owned_now",
    "enemy_visible",
    "neutral_visible",
    "owned_army",
    "enemy_army_visible",
    "ever_visible",
    "sight_age",
    "remembered_owned",
    "remembered_enemy",
    "remembered_neutral",
    "remembered_enemy_army",
    "remembered_own_castle",
    "remembered_enemy_castle",
    "belief_enemy_owner",
    "belief_enemy_army_mean",
    "belief_enemy_army_std",
    "belief_enemy_general",
    "belief_enemy_castle_owner",
    "belief_enemy_visibility",
    "belief_owner_entropy",
    "previous_move_source",
    "previous_move_destination",
    "previous_move_kind",
    "previous_build_cell",
    "row_coordinate",
    "column_coordinate",
    "row_from_own_general",
    "column_from_own_general",
    "turn_fraction",
    "pre_deathtouch_fraction",
    "deathtouch_active",
    "structure_growth_next",
    "bulk_growth_countdown",
    "own_land_fraction",
    "enemy_land_fraction",
    "own_army_total",
    "enemy_army_total",
    "land_margin",
    "army_margin",
    "belief_ess",
)


def army_value(x, scale: float = ARMY_SCALE) -> Array:
    """``clip(log1p(max(x, 0)) / log1p(scale), 0, 1)``."""
    x = np.asarray(x, dtype=np.float64)
    denom = math.log1p(scale)
    out = np.log1p(np.maximum(x, 0.0)) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class BeliefSummary(NamedTuple):
    """Particle aggregates injected into belief planes (raw army units)."""

    enemy_owner: Array  # (H, W) float probability
    enemy_army_mean: Array  # (H, W) raw mean army
    enemy_army_std: Array  # (H, W) raw army std
    enemy_general: Array  # (H, W) probability
    enemy_castle_owner: Array  # (H, W) probability
    enemy_visibility: Array  # (H, W) probability
    ess_fraction: float  # ESS / particle count in [0, 1]


def zero_belief(H: int, W: int) -> BeliefSummary:
    z = np.zeros((H, W), dtype=np.float32)
    return BeliefSummary(
        enemy_owner=z.copy(),
        enemy_army_mean=z.copy(),
        enemy_army_std=z.copy(),
        enemy_general=z.copy(),
        enemy_castle_owner=z.copy(),
        enemy_visibility=z.copy(),
        ess_fraction=0.0,
    )


def binary_entropy_bits(p: Array) -> Array:
    """Binary entropy of ``p`` in bits (divided by ``ln(2)``). Zero at 0 and 1."""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros(p.shape, dtype=np.float32)
    mid = (p > 0.0) & (p < 1.0)
    if not np.any(mid):
        return out
    pm = p[mid]
    h = -pm * np.log(pm) - (1.0 - pm) * np.log(1.0 - pm)
    out[mid] = (h / LN2).astype(np.float32)
    return out


def _pad_plane(plane: Array, H: int, W: int) -> Array:
    out = np.zeros((PAD, PAD), dtype=np.float32)
    out[:H, :W] = np.asarray(plane, dtype=np.float32)
    return out


def _fill_constant(board_mask: Array, value: float) -> Array:
    out = np.zeros((PAD, PAD), dtype=np.float32)
    out[board_mask > 0] = np.float32(value)
    return out


def _own_general_rc(memory: VisibleMemory) -> tuple[int, int]:
    pos = np.argwhere(memory.own_general)
    if len(pos) == 0:
        return 0, 0
    return int(pos[0, 0]), int(pos[0, 1])


def _paint_previous_action(
    tensor: Array,
    action: Optional[Sequence[int]],
    H: int,
    W: int,
) -> None:
    """Paint perspective-player previous-action planes (indices 29-32)."""
    if action is None:
        return
    pass_f, row, col, direction, split = (int(x) for x in action)
    if pass_f == 1:
        return
    if pass_f == 2:
        if 0 <= row < H and 0 <= col < W:
            tensor[P_PREV_BUILD_CELL, row, col] = 1.0
        return
    # Move.
    if not (0 <= row < H and 0 <= col < W):
        return
    tensor[P_PREV_MOVE_SOURCE, row, col] = 1.0
    drc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
    dr, dc = int(drc[0]), int(drc[1])
    dest_r, dest_c = row + dr, col + dc
    if 0 <= dest_r < H and 0 <= dest_c < W:
        tensor[P_PREV_MOVE_DEST, dest_r, dest_c] = 1.0
        tensor[P_PREV_MOVE_KIND, dest_r, dest_c] = 0.5 if split == 1 else 1.0


def build_tensor(
    obs,
    memory: VisibleMemory,
    *,
    belief: Optional[BeliefSummary] = None,
    previous_action: Optional[Sequence[int]] = None,
    army_scale: float = ARMY_SCALE,
) -> Array:
    """Return ``(49, 21, 21)`` float32. ``memory`` must already include ``obs``."""
    H, W = int(obs.H), int(obs.W)
    assert memory.H == H and memory.W == W
    if belief is None:
        belief = zero_belief(H, W)

    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    armies = np.asarray(obs.army_grid, dtype=np.float64)
    turn = int(obs.turn)

    board_mask_hw = np.ones((H, W), dtype=np.float32)
    board_mask = _pad_plane(board_mask_hw, H, W)

    visible_now = ((types != TYPE_FOG) & (types != TYPE_STRUCTURE_FOG)).astype(
        np.float32
    )
    fog0 = (types == TYPE_FOG).astype(np.float32)
    fog5 = (types == TYPE_STRUCTURE_FOG).astype(np.float32)

    owned = (owners == OWNER_ME).astype(np.float32)
    enemy = (owners == OWNER_ENEMY).astype(np.float32)
    # Neutral on a passable visible cell (not mountain / not fogged structure).
    passable_vis = visible_now.astype(bool) & ~memory.known_mountain
    neutral = (
        (owners == OWNER_NEUTRAL) & passable_vis & (types != TYPE_FOG)
    ).astype(np.float32)

    owned_army = army_value(armies * (owners == OWNER_ME), army_scale)
    enemy_army = army_value(armies * (owners == OWNER_ENEMY), army_scale)

    # Sight age: 0 before first sight; else (turn - last_seen) / 1200.
    last = memory.last_seen_turn.astype(np.float64)
    age = np.zeros((H, W), dtype=np.float32)
    seen = memory.ever_visible
    age[seen] = np.clip((turn - last[seen]) / float(TRUNCATION_TURN), 0.0, 1.0).astype(
        np.float32
    )

    rem_owner = memory.remembered_owner
    rem_owned = (memory.ever_visible & (rem_owner == OWNER_ME)).astype(np.float32)
    rem_enemy = (memory.ever_visible & (rem_owner == OWNER_ENEMY)).astype(np.float32)
    rem_neutral = (memory.ever_visible & (rem_owner == OWNER_NEUTRAL)).astype(
        np.float32
    )
    rem_enemy_army = np.zeros((H, W), dtype=np.float32)
    enemy_mem = memory.ever_visible & (rem_owner == OWNER_ENEMY)
    rem_enemy_army[enemy_mem] = army_value(
        memory.remembered_army[enemy_mem], army_scale
    )

    rem_own_castle = (
        memory.ever_visible
        & memory.remembered_was_castle
        & (memory.remembered_castle_owner == OWNER_ME)
    ).astype(np.float32)
    rem_enemy_castle = (
        memory.ever_visible
        & memory.remembered_was_castle
        & (memory.remembered_castle_owner == OWNER_ENEMY)
    ).astype(np.float32)

    p_enemy = np.asarray(belief.enemy_owner, dtype=np.float32)
    belief_mean = army_value(belief.enemy_army_mean, army_scale)
    belief_std = army_value(belief.enemy_army_std, army_scale)
    belief_entropy = binary_entropy_bits(p_enemy)

    gr, gc = _own_general_rc(memory)
    rows = np.arange(H, dtype=np.float32)[:, None]
    cols = np.arange(W, dtype=np.float32)[None, :]
    row_coord = np.broadcast_to(rows / 20.0, (H, W)).astype(np.float32)
    col_coord = np.broadcast_to(cols / 20.0, (H, W)).astype(np.float32)
    row_from = ((rows - gr) / 20.0).astype(np.float32)
    row_from = np.broadcast_to(row_from, (H, W)).copy()
    col_from = ((cols - gc) / 20.0).astype(np.float32)
    col_from = np.broadcast_to(col_from, (H, W)).copy()

    my_land = float(obs.my_land)
    opp_land = float(obs.opp_land)
    my_army = float(obs.my_army)
    opp_army = float(obs.opp_army)
    land_margin = (my_land - opp_land) / (my_land + opp_land + 1.0)
    army_margin = (my_army - opp_army) / (my_army + opp_army + 1.0)

    turn_f = turn / float(TRUNCATION_TURN)
    pre_dt = float(np.clip((DEATHTOUCH_TURN - turn) / float(DEATHTOUCH_TURN), 0.0, 1.0))
    dt_active = 1.0 if turn >= DEATHTOUCH_TURN else 0.0
    struct_next = 1.0 if ((turn + 1) % 2 == 0) else 0.0
    bulk = ((50 - ((turn + 1) % 50)) % 50) / 49.0

    tensor = np.zeros((N_PLANES, PAD, PAD), dtype=np.float32)
    planes_hw = [
        (P_BOARD_MASK, board_mask_hw),
        (P_VISIBLE_NOW, visible_now),
        (P_FOG_NONSTRUCTURE, fog0),
        (P_FOG_STRUCTURE, fog5),
        (P_KNOWN_MOUNTAIN, memory.known_mountain.astype(np.float32)),
        (P_KNOWN_PASSABLE, memory.known_passable_base.astype(np.float32)),
        (P_KNOWN_CASTLE, memory.known_castle.astype(np.float32)),
        (P_OWN_GENERAL, memory.own_general.astype(np.float32)),
        (P_KNOWN_ENEMY_GENERAL, memory.known_enemy_general.astype(np.float32)),
        (P_OWNED_NOW, owned),
        (P_ENEMY_VISIBLE, enemy),
        (P_NEUTRAL_VISIBLE, neutral),
        (P_OWNED_ARMY, owned_army),
        (P_ENEMY_ARMY_VISIBLE, enemy_army),
        (P_EVER_VISIBLE, memory.ever_visible.astype(np.float32)),
        (P_SIGHT_AGE, age),
        (P_REMEMBERED_OWNED, rem_owned),
        (P_REMEMBERED_ENEMY, rem_enemy),
        (P_REMEMBERED_NEUTRAL, rem_neutral),
        (P_REMEMBERED_ENEMY_ARMY, rem_enemy_army),
        (P_REMEMBERED_OWN_CASTLE, rem_own_castle),
        (P_REMEMBERED_ENEMY_CASTLE, rem_enemy_castle),
        (P_BELIEF_ENEMY_OWNER, p_enemy),
        (P_BELIEF_ENEMY_ARMY_MEAN, belief_mean),
        (P_BELIEF_ENEMY_ARMY_STD, belief_std),
        (P_BELIEF_ENEMY_GENERAL, np.asarray(belief.enemy_general, dtype=np.float32)),
        (
            P_BELIEF_ENEMY_CASTLE_OWNER,
            np.asarray(belief.enemy_castle_owner, dtype=np.float32),
        ),
        (
            P_BELIEF_ENEMY_VISIBILITY,
            np.asarray(belief.enemy_visibility, dtype=np.float32),
        ),
        (P_BELIEF_OWNER_ENTROPY, belief_entropy),
        (P_ROW_COORD, row_coord),
        (P_COL_COORD, col_coord),
        (P_ROW_FROM_GENERAL, row_from),
        (P_COL_FROM_GENERAL, col_from),
    ]
    for idx, plane in planes_hw:
        tensor[idx] = _pad_plane(plane, H, W)

    # Constants across the playable board; padding stays zero.
    constants = {
        P_TURN_FRACTION: turn_f,
        P_PRE_DEATHTOUCH: pre_dt,
        P_DEATHTOUCH_ACTIVE: dt_active,
        P_STRUCTURE_GROWTH_NEXT: struct_next,
        P_BULK_GROWTH_COUNTDOWN: bulk,
        P_OWN_LAND_FRACTION: my_land / 441.0,
        P_ENEMY_LAND_FRACTION: opp_land / 441.0,
        P_OWN_ARMY_TOTAL: float(army_value(my_army, army_scale)),
        P_ENEMY_ARMY_TOTAL: float(army_value(opp_army, army_scale)),
        P_LAND_MARGIN: land_margin,
        P_ARMY_MARGIN: army_margin,
        P_BELIEF_ESS: float(belief.ess_fraction),
    }
    for idx, value in constants.items():
        tensor[idx] = _fill_constant(board_mask, value)

    _paint_previous_action(tensor, previous_action, H, W)
    # Ensure previous-action paint does not leak into padding.
    for idx in (
        P_PREV_MOVE_SOURCE,
        P_PREV_MOVE_DEST,
        P_PREV_MOVE_KIND,
        P_PREV_BUILD_CELL,
    ):
        tensor[idx] *= board_mask

    return tensor


def observation_tensor(
    obs,
    memory: VisibleMemory,
    *,
    belief: Optional[BeliefSummary] = None,
    previous_action: Optional[Sequence[int]] = None,
    army_scale: float = ARMY_SCALE,
) -> tuple[Array, VisibleMemory]:
    """Update memory from ``obs``, then build the tensor."""
    mem = update_memory(memory, obs)
    return build_tensor(
        obs,
        mem,
        belief=belief,
        previous_action=previous_action,
        army_scale=army_scale,
    ), mem
