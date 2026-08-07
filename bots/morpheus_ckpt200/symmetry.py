"""Board symmetries for Morpheus tensors, actions, memory, and policy targets.

The eight dihedral transforms act on the padded ``21 × 21`` square. Coordinates,
direction channels, previous-action planes, generals, memory, belief planes, and
policy logits remap together. The pass logit never moves.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np

from action import (
    N_ACTIONS,
    PAD,
    PASS_INDEX,
    decode_action,
    encode_action,
)
from memory import VisibleMemory

Array = np.ndarray

# Direction indices: 0 up, 1 down, 2 left, 3 right.
_DIR_UP, _DIR_DOWN, _DIR_LEFT, _DIR_RIGHT = 0, 1, 2, 3


class Symmetry(NamedTuple):
    """One element of the dihedral group D4 on the padded board."""

    name: str
    transform_rc: Callable[[int, int], tuple[int, int]]
    transform_dir: Callable[[int], int]
    inverse_name: str


def _rot90_rc(r: int, c: int) -> tuple[int, int]:
    # (r, c) -> (c, PAD-1-r)
    return c, PAD - 1 - r


def _rot180_rc(r: int, c: int) -> tuple[int, int]:
    return PAD - 1 - r, PAD - 1 - c


def _rot270_rc(r: int, c: int) -> tuple[int, int]:
    # (r, c) -> (PAD-1-c, r)
    return PAD - 1 - c, r


def _flip_h_rc(r: int, c: int) -> tuple[int, int]:
    # Reflect over vertical axis: columns flip.
    return r, PAD - 1 - c


def _flip_v_rc(r: int, c: int) -> tuple[int, int]:
    return PAD - 1 - r, c


def _rot90_dir(d: int) -> int:
    # up->right, right->down, down->left, left->up
    return (_DIR_RIGHT, _DIR_LEFT, _DIR_UP, _DIR_DOWN)[d]


def _rot180_dir(d: int) -> int:
    return (_DIR_DOWN, _DIR_UP, _DIR_RIGHT, _DIR_LEFT)[d]


def _rot270_dir(d: int) -> int:
    # up->left, left->down, down->right, right->up
    return (_DIR_LEFT, _DIR_RIGHT, _DIR_DOWN, _DIR_UP)[d]


def _flip_h_dir(d: int) -> int:
    # left <-> right
    return (_DIR_UP, _DIR_DOWN, _DIR_RIGHT, _DIR_LEFT)[d]


def _flip_v_dir(d: int) -> int:
    # up <-> down
    return (_DIR_DOWN, _DIR_UP, _DIR_LEFT, _DIR_RIGHT)[d]


def _compose_rc(f, g):
    def composed(r: int, c: int) -> tuple[int, int]:
        r2, c2 = g(r, c)
        return f(r2, c2)

    return composed


def _compose_dir(f, g):
    def composed(d: int) -> int:
        return f(g(d))

    return composed


SYMMETRIES: tuple[Symmetry, ...] = (
    Symmetry("id", lambda r, c: (r, c), lambda d: d, "id"),
    Symmetry("rot90", _rot90_rc, _rot90_dir, "rot270"),
    Symmetry("rot180", _rot180_rc, _rot180_dir, "rot180"),
    Symmetry("rot270", _rot270_rc, _rot270_dir, "rot90"),
    Symmetry("flip_h", _flip_h_rc, _flip_h_dir, "flip_h"),
    Symmetry("flip_v", _flip_v_rc, _flip_v_dir, "flip_v"),
    Symmetry(
        "rot90_flip_h",
        _compose_rc(_flip_h_rc, _rot90_rc),
        _compose_dir(_flip_h_dir, _rot90_dir),
        "rot90_flip_h",  # involution
    ),
    Symmetry(
        "rot270_flip_h",
        _compose_rc(_flip_h_rc, _rot270_rc),
        _compose_dir(_flip_h_dir, _rot270_dir),
        "rot270_flip_h",  # involution
    ),
)

_BY_NAME = {s.name: s for s in SYMMETRIES}

_INVERSE = {
    "id": "id",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_h": "flip_h",
    "flip_v": "flip_v",
    "rot90_flip_h": "rot90_flip_h",
    "rot270_flip_h": "rot270_flip_h",
}


def get_symmetry(name: str) -> Symmetry:
    return _BY_NAME[name]


def inverse_symmetry(name: str) -> Symmetry:
    return _BY_NAME[_INVERSE[name]]


def transform_plane(plane: Array, sym: Symmetry) -> Array:
    """Apply a spatial transform to a ``(PAD, PAD)`` plane."""
    out = np.zeros_like(plane)
    for r in range(PAD):
        for c in range(PAD):
            nr, nc = sym.transform_rc(r, c)
            out[nr, nc] = plane[r, c]
    return out


def transform_tensor(tensor: Array, sym: Symmetry) -> Array:
    """Remap all 49 spatial planes, then rewrite absolute/relative coordinates."""
    from tensor import (
        P_BOARD_MASK,
        P_COL_COORD,
        P_COL_FROM_GENERAL,
        P_OWN_GENERAL,
        P_ROW_COORD,
        P_ROW_FROM_GENERAL,
    )

    assert tensor.shape == (49, PAD, PAD)
    out = np.zeros_like(tensor)
    for p in range(49):
        out[p] = transform_plane(tensor[p], sym)

    board = out[P_BOARD_MASK] > 0
    out[P_ROW_COORD] = np.zeros((PAD, PAD), dtype=np.float32)
    out[P_COL_COORD] = np.zeros((PAD, PAD), dtype=np.float32)
    out[P_ROW_FROM_GENERAL] = np.zeros((PAD, PAD), dtype=np.float32)
    out[P_COL_FROM_GENERAL] = np.zeros((PAD, PAD), dtype=np.float32)

    pos = np.argwhere(out[P_OWN_GENERAL] > 0.5)
    if len(pos):
        gr, gc = int(pos[0, 0]), int(pos[0, 1])
    else:
        gr, gc = 0, 0

    rs, cs = np.where(board)
    out[P_ROW_COORD, rs, cs] = rs.astype(np.float32) / 20.0
    out[P_COL_COORD, rs, cs] = cs.astype(np.float32) / 20.0
    out[P_ROW_FROM_GENERAL, rs, cs] = (rs.astype(np.float32) - gr) / 20.0
    out[P_COL_FROM_GENERAL, rs, cs] = (cs.astype(np.float32) - gc) / 20.0
    return out


def transform_action_tuple(
    action: Sequence[int], sym: Symmetry
) -> tuple[int, int, int, int, int]:
    """Remap a wire action. Pass is unchanged."""
    pass_f, row, col, direction, split = (int(x) for x in action)
    if pass_f == 1:
        return (1, 0, 0, 0, 0)
    nr, nc = sym.transform_rc(row, col)
    if pass_f == 2:
        return (2, nr, nc, 0, 0)
    nd = sym.transform_dir(direction)
    return (0, nr, nc, nd, split)


def transform_policy(logits: Array, sym: Symmetry) -> Array:
    """Remap a length-3970 policy vector (or logit vector)."""
    assert logits.shape == (N_ACTIONS,)
    out = np.zeros_like(logits)
    out[PASS_INDEX] = logits[PASS_INDEX]
    for idx in range(PASS_INDEX):
        action = decode_action(idx)
        new_action = transform_action_tuple(action, sym)
        out[encode_action(new_action)] = logits[idx]
    return out


def transform_memory(memory: VisibleMemory, sym: Symmetry) -> VisibleMemory:
    """Remap every spatial memory field on the padded board, then crop to H×W.

    Memory lives on the true H×W board in the upper-left. Transforms that move
    content outside ``[:H, :W]`` are still valid on the padded tensor path; this
    helper is for square boards (H = W = PAD) or tests that keep content inside
    the true board after the transform.
    """
    H, W = memory.H, memory.W

    def xform(field: Array) -> Array:
        padded = np.zeros((PAD, PAD), dtype=field.dtype)
        padded[:H, :W] = field
        moved = transform_plane(padded, sym)
        return moved[:H, :W].astype(field.dtype, copy=False)

    return VisibleMemory(
        H=H,
        W=W,
        known_mountain=xform(memory.known_mountain),
        known_passable_base=xform(memory.known_passable_base),
        known_castle=xform(memory.known_castle),
        own_general=xform(memory.own_general),
        known_enemy_general=xform(memory.known_enemy_general),
        ever_visible=xform(memory.ever_visible),
        last_seen_turn=xform(memory.last_seen_turn),
        remembered_owner=xform(memory.remembered_owner),
        remembered_army=xform(memory.remembered_army),
        remembered_was_castle=xform(memory.remembered_was_castle),
        remembered_castle_owner=xform(memory.remembered_castle_owner),
    )


def all_symmetry_names() -> tuple[str, ...]:
    return tuple(s.name for s in SYMMETRIES)
