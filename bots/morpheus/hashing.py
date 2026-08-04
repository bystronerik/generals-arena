"""Information-set and enemy-table digests for Morpheus search.

Keys are full SHA-256 digests. See docs/bots/morpheus/search.md Part 06
executable definitions for the collision policy and payload layout.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Optional, Sequence

import numpy as np

from action import encode_action
from memory import VisibleMemory
from observe import observation_hash

Array = np.ndarray
ZERO_DIGEST = b"\x00" * 32


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def memory_digest(memory: VisibleMemory) -> bytes:
    """SHA-256 over every VisibleMemory array in declaration order, then H, W."""
    parts = [
        np.ascontiguousarray(memory.known_mountain).tobytes(),
        np.ascontiguousarray(memory.known_passable_base).tobytes(),
        np.ascontiguousarray(memory.known_castle).tobytes(),
        np.ascontiguousarray(memory.own_general).tobytes(),
        np.ascontiguousarray(memory.known_enemy_general).tobytes(),
        np.ascontiguousarray(memory.ever_visible).tobytes(),
        np.ascontiguousarray(memory.last_seen_turn).tobytes(),
        np.ascontiguousarray(memory.remembered_owner).tobytes(),
        np.ascontiguousarray(memory.remembered_army).tobytes(),
        np.ascontiguousarray(memory.remembered_was_castle).tobytes(),
        np.ascontiguousarray(memory.remembered_castle_owner).tobytes(),
        struct.pack("<ii", int(memory.H), int(memory.W)),
    ]
    return _sha256(*parts)


def roll_history_digest(
    prev: bytes,
    action: Sequence[int],
    obs_hash: bytes,
) -> bytes:
    """Advance the rolling action-observation history digest."""
    if len(prev) != 32:
        raise ValueError("history digest must be 32 bytes")
    action_i = struct.pack("<i", int(encode_action(action)))
    return _sha256(prev, action_i, obs_hash)


def info_state_key(
    turn: int,
    memory: VisibleMemory,
    obs_hash: bytes,
    history_digest: bytes = ZERO_DIGEST,
) -> bytes:
    """Node key: turn || memory digest || observation hash || history digest."""
    return _sha256(
        struct.pack("<i", int(turn)),
        memory_digest(memory),
        obs_hash,
        history_digest,
    )


def enemy_info_hash(enemy_obs, enemy_memory: VisibleMemory) -> bytes:
    """Enemy table key from fogged enemy observation and enemy memory."""
    return _sha256(observation_hash(enemy_obs), memory_digest(enemy_memory))


def child_edge_key(action: Sequence[int], obs_hash: bytes) -> bytes:
    """Child map key: Morpheus action index plus resulting observation hash."""
    return _sha256(struct.pack("<i", int(encode_action(action))), obs_hash)


def digests_equal(a: Optional[bytes], b: Optional[bytes]) -> bool:
    if a is None or b is None:
        return False
    return a == b
