"""Persistent visible memory for Morpheus.

Update memory from each perspective-relative observation before the tensor
builder runs. Static terrain and latched generals never expire. Dynamic owner
and army memory keep the last seen value and age.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

Array = np.ndarray

# Competition type codes (perspective-relative wire).
TYPE_FOG = 0
TYPE_PLAIN = 1
TYPE_MOUNTAIN = 2
TYPE_CASTLE = 3
TYPE_GENERAL = 4
TYPE_STRUCTURE_FOG = 5

OWNER_NEUTRAL = 0
OWNER_ME = 1
OWNER_ENEMY = 2


class VisibleMemory(NamedTuple):
    """Persistent facts and last-seen dynamic values on the true H×W board."""

    H: int
    W: int
    known_mountain: Array  # (H, W) bool
    known_passable_base: Array  # (H, W) bool — non-mountain, non-castle terrain
    known_castle: Array  # (H, W) bool
    own_general: Array  # (H, W) bool — at most one cell
    known_enemy_general: Array  # (H, W) bool — latched or all false
    ever_visible: Array  # (H, W) bool
    last_seen_turn: Array  # (H, W) int32; -1 before first sight
    remembered_owner: Array  # (H, W) int8; 0/1/2 at last sight
    remembered_army: Array  # (H, W) int32; army at last sight
    remembered_was_castle: Array  # (H, W) bool; castle flag at last sight
    remembered_castle_owner: Array  # (H, W) int8; owner when last seen as castle


def empty_memory(H: int, W: int) -> VisibleMemory:
    """Blank memory before the first observation."""
    shape = (H, W)
    return VisibleMemory(
        H=H,
        W=W,
        known_mountain=np.zeros(shape, dtype=bool),
        known_passable_base=np.zeros(shape, dtype=bool),
        known_castle=np.zeros(shape, dtype=bool),
        own_general=np.zeros(shape, dtype=bool),
        known_enemy_general=np.zeros(shape, dtype=bool),
        ever_visible=np.zeros(shape, dtype=bool),
        last_seen_turn=np.full(shape, -1, dtype=np.int32),
        remembered_owner=np.zeros(shape, dtype=np.int8),
        remembered_army=np.zeros(shape, dtype=np.int32),
        remembered_was_castle=np.zeros(shape, dtype=bool),
        remembered_castle_owner=np.zeros(shape, dtype=np.int8),
    )


def _as_grids(obs) -> tuple[Array, Array, Array, int]:
    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    armies = np.asarray(obs.army_grid, dtype=np.int32)
    turn = int(obs.turn)
    return types, owners, armies, turn


def update_memory(memory: VisibleMemory, obs) -> VisibleMemory:
    """Fold one observation into persistent memory.

    Visible cells replace dynamic memory. Type ``5`` on a cell that was never
    passable is a mountain (competition maps start with no castles). A later
    ``0 → 5`` change marks a new castle.
    """
    H, W = memory.H, memory.W
    assert int(obs.H) == H and int(obs.W) == W
    types, owners, armies, turn = _as_grids(obs)

    known_mountain = memory.known_mountain.copy()
    known_passable_base = memory.known_passable_base.copy()
    known_castle = memory.known_castle.copy()
    own_general = memory.own_general.copy()
    known_enemy_general = memory.known_enemy_general.copy()
    ever_visible = memory.ever_visible.copy()
    last_seen_turn = memory.last_seen_turn.copy()
    remembered_owner = memory.remembered_owner.copy()
    remembered_army = memory.remembered_army.copy()
    remembered_was_castle = memory.remembered_was_castle.copy()
    remembered_castle_owner = memory.remembered_castle_owner.copy()

    visible = (types != TYPE_FOG) & (types != TYPE_STRUCTURE_FOG)

    # --- Fog / structure-in-fog terrain inference (also on invisible cells) ---
    fog0 = types == TYPE_FOG
    # Type 0 proves the cell has no mountain or castle.
    known_passable_base[fog0] = True
    known_mountain[fog0] = False
    # Do not clear known_castle on type 0: a castle that became fogged as type 0
    # is impossible under the encoder (castles in fog are type 5). Type 0 on a
    # former unknown cell only asserts passable base.

    struct_fog = types == TYPE_STRUCTURE_FOG
    # First-frame / never-passable type 5 → mountain; else → castle.
    was_passable = (
        known_passable_base | known_castle | memory.ever_visible
    )
    new_castle = struct_fog & was_passable
    new_mountain = struct_fog & ~was_passable
    known_castle[new_castle] = True
    known_mountain[new_castle] = False
    known_passable_base[new_castle] = False
    known_mountain[new_mountain] = True
    known_castle[new_mountain] = False
    known_passable_base[new_mountain] = False

    # --- Visible terrain ---
    known_mountain[types == TYPE_MOUNTAIN] = True
    known_passable_base[types == TYPE_MOUNTAIN] = False
    known_castle[types == TYPE_MOUNTAIN] = False

    known_passable_base[types == TYPE_PLAIN] = True
    known_mountain[types == TYPE_PLAIN] = False
    known_castle[types == TYPE_PLAIN] = False

    known_castle[types == TYPE_CASTLE] = True
    known_mountain[types == TYPE_CASTLE] = False
    known_passable_base[types == TYPE_CASTLE] = False

    # Generals: latch both seats; plain/castle flags clear on that cell.
    own_gen_vis = (types == TYPE_GENERAL) & (owners == OWNER_ME)
    enemy_gen_vis = (types == TYPE_GENERAL) & (owners == OWNER_ENEMY)
    own_general[own_gen_vis] = True
    known_enemy_general[enemy_gen_vis] = True
    for gen_mask in (own_gen_vis, enemy_gen_vis):
        known_mountain[gen_mask] = False
        known_passable_base[gen_mask] = False
        known_castle[gen_mask] = False

    # --- Dynamic last-seen (visible cells only) ---
    ever_visible[visible] = True
    last_seen_turn[visible] = turn
    remembered_owner[visible] = owners[visible].astype(np.int8)
    remembered_army[visible] = armies[visible]
    is_castle_now = types == TYPE_CASTLE
    remembered_was_castle[visible] = is_castle_now[visible]
    # Castle owner memory updates when the visible cell is a castle.
    castle_vis = visible & is_castle_now
    remembered_castle_owner[castle_vis] = owners[castle_vis].astype(np.int8)

    return VisibleMemory(
        H=H,
        W=W,
        known_mountain=known_mountain,
        known_passable_base=known_passable_base,
        known_castle=known_castle,
        own_general=own_general,
        known_enemy_general=known_enemy_general,
        ever_visible=ever_visible,
        last_seen_turn=last_seen_turn,
        remembered_owner=remembered_owner,
        remembered_army=remembered_army,
        remembered_was_castle=remembered_was_castle,
        remembered_castle_owner=remembered_castle_owner,
    )
