"""Tactical helpers for Morpheus search and deterministic play vs expanders.

Pass is omitted whenever any other playable action exists. Until an enemy is
visible, moves onto the own general or castle are banned, and idle piles on
those structures are pushed out toward fog. Direction toward the seek target
is rewarded on own land; retreats are punished; enemy takes rank highest.
Land lead is not a hunt gate.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from action import (
    PASS_INDEX,
    decode_action,
    encode_action,
    legal_mask,
)
from memory import (
    OWNER_ENEMY,
    OWNER_NEUTRAL,
    TYPE_CASTLE,
    TYPE_FOG,
    TYPE_GENERAL,
    TYPE_MOUNTAIN,
    TYPE_STRUCTURE_FOG,
    VisibleMemory,
)
from observe import visibility_mask
from transition import DIRECTIONS

Array = np.ndarray
Action5 = tuple[int, int, int, int, int]

# Early soft explore window (urgency still rises after this until contact).
EXPLORE_TURNS = 20
# Fog-hunt urgency keeps rising until first enemy sight.
FOG_URGENCY_TURN_SCALE = 40.0
# Tip armies below this are slow explorers — prefer a formed wave.
EXPLORE_WAVE_MIN = 3
# Remember this many recent army moves; reverse along any of those edges is banned.
OSCILLATION_HISTORY = 8

MoveSegment = tuple[tuple[int, int], tuple[int, int]]


def move_dest(action: Action5) -> Optional[tuple[int, int, int, int]]:
    """Return ``(sr, sc, tr, tc)`` for a move action, else ``None``."""
    if int(action[0]) != 0:
        return None
    sr, sc, d = int(action[1]), int(action[2]), int(action[3])
    if not (0 <= d < 4):
        return None
    tr = sr + int(DIRECTIONS[d, 0])
    tc = sc + int(DIRECTIONS[d, 1])
    return sr, sc, tr, tc


def move_segment(action: Action5) -> Optional[MoveSegment]:
    """Return ``((sr, sc), (tr, tc))`` for a move action."""
    ends = move_dest(action)
    if ends is None:
        return None
    sr, sc, tr, tc = ends
    return (sr, sc), (tr, tc)


def is_reverse_segment(a: MoveSegment, b: MoveSegment) -> bool:
    return a[0] == b[1] and a[1] == b[0]


def is_reverse_move(action: Action5, prev: Optional[Action5]) -> bool:
    """True when ``action`` walks the same edge as ``prev`` in reverse."""
    if prev is None:
        return False
    cur = move_segment(action)
    old = move_segment(prev)
    if cur is None or old is None:
        return False
    return is_reverse_segment(cur, old)


def _oscillation_history(
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> list[MoveSegment]:
    if recent_actions:
        seq: Sequence[Action5] = recent_actions
    elif prev_action is not None:
        seq = (prev_action,)
    else:
        seq = ()
    out: list[MoveSegment] = []
    for act in seq:
        seg = move_segment(act)
        if seg is not None:
            out.append(seg)
    if len(out) > OSCILLATION_HISTORY:
        return out[-OSCILLATION_HISTORY:]
    return out


def blocks_oscillation(
    action: Action5,
    prev_action: Optional[Action5],
    obs,
    *,
    recent_actions: Sequence[Action5] = (),
    force_allow: bool = False,
) -> bool:
    """True when the move reverses a recent own-corridor edge.

    Checks the last ``OSCILLATION_HISTORY`` army moves, not only the previous
    step, so A→B→C then C→B→A ping-pong is banned. Reverse onto enemy land or
    a cell that unlocks new vision stays legal. ``force_allow`` is last resort.
    """
    if force_allow:
        return False
    seg = move_segment(action)
    if seg is None:
        return False
    _src, (tr, tc) = seg
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    H, W = int(obs.H), int(obs.W)
    if not (0 <= tr < H and 0 <= tc < W):
        return True
    if int(owners[tr, tc]) == OWNER_ENEMY:
        return False
    # New vision is progress, not wasted oscillation.
    if newly_revealed_cells(obs, tr, tc) > 0:
        return False
    history = _oscillation_history(prev_action, recent_actions)
    for old in history:
        if is_reverse_segment(seg, old):
            return True
    return False


# Soft size bonus for tip-feed / explore (not for commit attacks).
WAVE_ARMY_SOFT_CAP = 20
# Own-land pile merges at/above this are stacking waste.
STACK_GATHER_BAN = 16
# Large stacks may march onto thin own cells toward the enemy.
COMMIT_DEST_ARMY_MAX = 8
# Attack scoring still cares about size up to this.
ATTACK_ARMY_CAP = 200
# Encourage castle builds through this turn; mild penalty after.
CASTLE_EARLY_UNTIL = 200
# Pre-contact: armies at/above this on gen/castle should leave, not idle.
# Keep a real garrison — emptying the general loses to rush bots.
STRUCTURE_IDLE_ARMY = 18
# Below this share of total army in the king stack, sweep land into it.
GATHER_SHARE_MIN = 0.35
# Tip armies below this fraction of the king must gather, not freestyle.
COMMIT_ARMY_FRAC = 0.35
# Absolute floor: a stack below this fraction of *total* army is a tip when
# the board is dispersed (max alone is a bad commit signal).
COMMIT_TOTAL_FRAC = 0.15
# Path distance (BFS) at which an enemy threatens the general enough to
# yank the king home. Manhattan-8 was too soft and stalled every push.
DEFEND_PATH_NEAR = 4
DEFEND_PATH_THIN = 6
# Garrison that still needs reinforcement when an enemy is within THIN range.
DEFEND_GEN_THIN = 20
# Minimum army left on the general when evacuating a fat pile is impossible
# in one step (moves leave 1); instead delay evacuate until above this.
GENERAL_EVACUATE_MIN = 18


def castle_timing_weight(turn: int) -> float:
    """Reward early castles; gently penalize late builds."""
    t = max(int(turn), 0)
    if t <= CASTLE_EARLY_UNTIL:
        # ~2.3 at opening → ~1.4 at turn 200.
        return 2.3 - 0.9 * (float(t) / float(CASTLE_EARLY_UNTIL))
    over = float(t - CASTLE_EARLY_UNTIL)
    return max(0.2, 1.2 - over / 280.0)


def explore_wave_weight(army: int) -> float:
    """Prefer a formed wave for fog carving; 1–2 army tips are slow."""
    a = max(int(army), 1)
    if a < EXPLORE_WAVE_MIN:
        return 0.12
    return 1.0 + 0.9 * float(np.log1p(min(a, 80)))


def fog_urgency(turn: int, *, enemy_seen: bool) -> float:
    """How hard to push fog reveal. Rises with turn until first contact."""
    if enemy_seen:
        return 1.0
    t = max(int(turn), 0)
    return 1.0 + (float(t) / FOG_URGENCY_TURN_SCALE) ** 1.15


def wave_weight(army: int) -> float:
    """Tiny size bonus up to a useful wave; flat after that."""
    a = max(min(int(army), WAVE_ARMY_SOFT_CAP), 1)
    return 1.0 + 0.25 * float(np.log1p(a))


def attack_weight(army: int) -> float:
    """Size bonus for committing a real stack into attack / approach."""
    a = max(min(int(army), ATTACK_ARMY_CAP), 1)
    return 1.0 + 0.85 * float(np.log1p(a))


def largest_own_army(obs) -> int:
    _types, owners, armies = _as_grids(obs)
    own = armies[owners == 1]
    if own.size == 0:
        return 0
    return int(own.max())


def king_cell(obs) -> Optional[tuple[int, int]]:
    """Cell holding the largest own army (ties: first in row-major order)."""
    _types, owners, armies = _as_grids(obs)
    own = owners == 1
    if not np.any(own):
        return None
    max_a = int(armies[own].max())
    locs = np.argwhere(own & (armies == max_a))
    return int(locs[0, 0]), int(locs[0, 1])


def rally_cell(obs, memory: VisibleMemory) -> Optional[tuple[int, int]]:
    """Stack target for land gather.

    After contact, pick the owned non-structure cell closest to the seek
    target (ties: larger army). That forms a front stack instead of feeding
    a parked general pile.
    """
    _types, owners, armies = _as_grids(obs)
    own = owners == 1
    if not np.any(own):
        return None
    if enemy_is_visible(obs, memory):
        off_struct = own & ~own_structure_mask(obs, memory)
        if np.any(off_struct):
            target = enemy_seek_target(obs, memory)
            best_pos: Optional[tuple[int, int]] = None
            best_key: Optional[tuple] = None
            for r, c in np.argwhere(off_struct):
                r, c = int(r), int(c)
                a = int(armies[r, c])
                if target is None:
                    key: tuple = (a,)
                else:
                    dist = abs(r - target[0]) + abs(c - target[1])
                    key = (-dist, a)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pos = (r, c)
            if best_pos is not None:
                return best_pos
    return king_cell(obs)


def general_threat_path(
    obs, memory: VisibleMemory
) -> tuple[Optional[tuple[int, int]], int, int]:
    """Return ``(gen_cell, nearest_enemy_path, gen_army)``.

    ``nearest_enemy_path`` is ``10**9`` when no enemy is visible.
    """
    types, owners, armies = _as_grids(obs)
    gen = np.argwhere(
        np.asarray(memory.own_general, dtype=bool)
        | ((types == TYPE_GENERAL) & (owners == 1))
    )
    if gen.size == 0:
        return None, 10**9, 0
    gr, gc = int(gen[0, 0]), int(gen[0, 1])
    gen_army = int(armies[gr, gc])
    if not enemy_is_visible(obs, memory):
        return (gr, gc), 10**9, gen_army
    enemy_cells = np.argwhere(owners == OWNER_ENEMY)
    if enemy_cells.size == 0:
        return (gr, gc), 10**9, gen_army
    gen_dist = path_distance_field(obs, [(gr, gc)])
    nearest_path = 10**9
    for r, c in enemy_cells:
        d = int(gen_dist[int(r), int(c)])
        if 0 <= d < nearest_path:
            nearest_path = d
    if nearest_path >= 10**9:
        nearest_path = min(
            int(abs(int(r) - gr) + abs(int(c) - gc)) for r, c in enemy_cells
        )
    return (gr, gc), int(nearest_path), gen_army


def general_is_threatened(obs, memory: VisibleMemory) -> bool:
    """True when an enemy is on a short path to our general."""
    _gen, nearest, gen_army = general_threat_path(obs, memory)
    if _gen is None:
        return False
    return nearest <= DEFEND_PATH_NEAR or (
        nearest <= DEFEND_PATH_THIN and gen_army < DEFEND_GEN_THIN
    )


def army_concentration(obs) -> tuple[float, int, int]:
    """Return ``(max_share, max_army, total_army)`` on own land."""
    _types, owners, armies = _as_grids(obs)
    own = owners == 1
    if not np.any(own):
        return 0.0, 0, 0
    total = int(armies[own].sum())
    max_a = int(armies[own].max())
    if total <= 0:
        return 0.0, max_a, 0
    return float(max_a) / float(total), max_a, total


def is_committed_army(
    army: int, max_army: int, total: int, share: float
) -> bool:
    """True when this stack may freestyle (attack / march); else gather.

    When the board is dispersed, ``max_army`` alone is a bad signal: a tip of
    11 with total 200 looks "committed" at 0.35*max. Require a real fraction
    of total army, or the unique king while share is still low.
    """
    a = int(army)
    tot = max(int(total), 1)
    mx = max(int(max_army), 1)
    if share < GATHER_SHARE_MIN:
        # Only the king may press, and only once it holds a real fighting lump.
        return a >= mx and a >= max(STACK_GATHER_BAN, int(COMMIT_TOTAL_FRAC * tot))
    return a >= COMMIT_ARMY_FRAC * mx or a >= COMMIT_TOTAL_FRAC * tot


def tip_thrash_factor(
    src_army: int, max_army: int, total: int | None = None
) -> float:
    """Downweight tiny tip attacks while a much larger stack sits idle."""
    src = max(int(src_army), 1)
    big = max(int(max_army), 1)
    tot = int(total) if total is not None else big
    share = float(big) / float(max(tot, 1))
    if is_committed_army(src, big, tot, share):
        return 1.0
    if big < STACK_GATHER_BAN and share >= GATHER_SHARE_MIN:
        return 1.0
    return float(0.08 + 0.6 * (src / max(big, 1)))


def move_progress(
    sr: int,
    sc: int,
    tr: int,
    tc: int,
    target: Optional[tuple[int, int]],
) -> float:
    """Manhattan progress toward ``target`` (+1 closer, -1 farther, 0 lateral)."""
    if target is None:
        return 0.0
    before = abs(int(sr) - target[0]) + abs(int(sc) - target[1])
    after = abs(int(tr) - target[0]) + abs(int(tc) - target[1])
    return float(before - after)


def _path_passable_mask(obs) -> Array:
    """Cells an army can step onto (not mountain / structure-fog)."""
    types, _owners, _armies = _as_grids(obs)
    return (types != TYPE_MOUNTAIN) & (types != TYPE_STRUCTURE_FOG)


def path_distance_field(
    obs,
    goals: Sequence[tuple[int, int]],
) -> Array:
    """BFS distance to the nearest goal through passable cells. -1 = unreachable."""
    H, W = int(obs.H), int(obs.W)
    dist = np.full((H, W), -1, dtype=np.int32)
    passable = _path_passable_mask(obs)
    q: deque[tuple[int, int]] = deque()
    for gr, gc in goals:
        gr, gc = int(gr), int(gc)
        if not (0 <= gr < H and 0 <= gc < W):
            continue
        if not bool(passable[gr, gc]):
            continue
        if dist[gr, gc] == 0:
            continue
        dist[gr, gc] = 0
        q.append((gr, gc))
    while q:
        r, c = q.popleft()
        base = int(dist[r, c])
        for d in range(4):
            nr = r + int(DIRECTIONS[d, 0])
            nc = c + int(DIRECTIONS[d, 1])
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not bool(passable[nr, nc]):
                continue
            if dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = base + 1
            q.append((nr, nc))
    return dist


def seek_goals(obs, memory: VisibleMemory) -> list[tuple[int, int]]:
    """Cells to drive toward: enemy general, else all visible enemy land."""
    types, owners, _ = _as_grids(obs)
    latched = np.argwhere(memory.known_enemy_general)
    if latched.size:
        return [(int(latched[0, 0]), int(latched[0, 1]))]
    enemy_gen = np.argwhere((types == TYPE_GENERAL) & (owners == OWNER_ENEMY))
    if enemy_gen.size:
        return [(int(enemy_gen[0, 0]), int(enemy_gen[0, 1]))]
    enemy = np.argwhere(owners == OWNER_ENEMY)
    if enemy.size:
        return [(int(r), int(c)) for r, c in enemy]
    # Pre-contact: opposite corner from own general (fog hunt beacon).
    own = np.argwhere(memory.own_general)
    if own.size == 0:
        own = np.argwhere((types == TYPE_GENERAL) & (owners == 1))
    if own.size == 0:
        return []
    H, W = int(obs.H), int(obs.W)
    gr, gc = int(own[0, 0]), int(own[0, 1])
    return [(H - 1 - gr, W - 1 - gc)]


def path_progress(
    sr: int,
    sc: int,
    tr: int,
    tc: int,
    dist_field: Optional[Array],
    *,
    fallback_target: Optional[tuple[int, int]] = None,
) -> float:
    """Path-aware progress: drop in BFS distance to seek goals.

    Falls back to Manhattan when the field is missing or a cell is
    unreachable (so fog carving that opens a route still gets a signal).
    """
    if dist_field is None:
        return move_progress(sr, sc, tr, tc, fallback_target)
    H, W = int(dist_field.shape[0]), int(dist_field.shape[1])
    if not (0 <= sr < H and 0 <= sc < W and 0 <= tr < H and 0 <= tc < W):
        return move_progress(sr, sc, tr, tc, fallback_target)
    before = int(dist_field[sr, sc])
    after = int(dist_field[tr, tc])
    if before < 0 and after < 0:
        return move_progress(sr, sc, tr, tc, fallback_target)
    if before < 0:
        # Stepping onto the connected component of the goals.
        return 2.0 if after >= 0 else move_progress(sr, sc, tr, tc, fallback_target)
    if after < 0:
        return -2.0
    return float(before - after)


def direction_bias(progress: float, dest_owner: int) -> float:
    """Reward steps toward the seek target; punish retreats; mild on lateral.

    Own-land marches toward the enemy stay useful (build / approach). Enemy
    takes are strongest. Only steps that move *away* from the target are
    crushed — not every interior move.
    """
    p = float(progress)
    own = int(dest_owner)
    if own == OWNER_ENEMY:
        return 4.0 + 2.0 * max(p, 0.0)
    if p > 0.0:
        if own == 1:
            return 1.5 + 1.2 * p
        return 1.8 + 1.4 * p
    if p < 0.0:
        return 0.06
    if own == 1:
        return 0.25
    return 0.55


def stack_gather_factor(
    src_army: int,
    dest_army: int,
    dest_owner: int,
    *,
    progress: float = 0.0,
) -> float:
    """Own-land friction: soft on forward marches, hard on retreat dumps."""
    if int(dest_owner) != 1:
        return 1.0
    src = max(int(src_army), 1)
    dest = max(int(dest_army), 0)
    p = float(progress)
    if p > 0.0:
        # Toward enemy: allow corridor gathers / front consolidation.
        return float(1.0 / (1.0 + (dest / 25.0) ** 1.5))
    if p < 0.0:
        # Away from enemy: crush pile dumps and retreat shuffles.
        return float(0.03 / (1.0 + dest / 4.0))
    # Lateral: keep the old pile-merge ban.
    feed = 1.0 / (1.0 + (dest / 6.0) ** 2)
    if src >= STACK_GATHER_BAN and dest > COMMIT_DEST_ARMY_MAX:
        return float(feed * 0.05)
    return float(feed)


def is_overstack_own_gather(
    action: Action5,
    obs,
    *,
    target: Optional[tuple[int, int]] = None,
) -> bool:
    """True when a large stack dumps onto a fat own pile without progress."""
    ends = move_dest(action)
    if ends is None:
        return False
    sr, sc, tr, tc = ends
    _types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    if not (0 <= tr < H and 0 <= tc < W):
        return False
    if int(owners[tr, tc]) != 1:
        return False
    if int(armies[sr, sc]) < STACK_GATHER_BAN:
        return False
    if int(armies[tr, tc]) <= COMMIT_DEST_ARMY_MAX:
        return False
    # Forward consolidation toward the seek target is not waste.
    if move_progress(sr, sc, tr, tc, target) > 0.0:
        return False
    return True


def _as_grids(obs) -> tuple[Array, Array, Array]:
    return (
        np.asarray(obs.type_grid, dtype=np.int32),
        np.asarray(obs.owner_grid, dtype=np.int32),
        np.asarray(obs.army_grid, dtype=np.int32),
    )


def _is_passable_type(cell_type: int) -> bool:
    return int(cell_type) not in (TYPE_MOUNTAIN, TYPE_STRUCTURE_FOG)


def _is_visible_neutral(owner: int, cell_type: int) -> bool:
    return int(owner) == OWNER_NEUTRAL and int(cell_type) not in (
        TYPE_FOG,
        TYPE_STRUCTURE_FOG,
    )


def enemy_is_visible(obs, memory: VisibleMemory) -> bool:
    """True once any enemy-owned cell is in view or the enemy general is latched."""
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    if np.any(owners == OWNER_ENEMY):
        return True
    return bool(np.any(memory.known_enemy_general))


def own_structure_mask(obs, memory: VisibleMemory) -> Array:
    """Own general and own castles (latched or currently visible)."""
    types, owners, _ = _as_grids(obs)
    own = owners == 1
    gen = np.asarray(memory.own_general, dtype=bool) | (
        (types == TYPE_GENERAL) & own
    )
    castle = np.asarray(memory.known_castle, dtype=bool) | (types == TYPE_CASTLE)
    return gen | (castle & own)


def structure_idle_army(obs, memory: VisibleMemory) -> int:
    """Largest army sitting on own general/castle."""
    _types, _owners, armies = _as_grids(obs)
    structs = own_structure_mask(obs, memory)
    if not np.any(structs):
        return 0
    return int(armies[structs].max())


def play_mask(
    obs,
    memory: VisibleMemory,
    *,
    cost_grid: Optional[Array] = None,
) -> Array:
    """Legal mask with Morpheus play rules applied.

    - Pass is illegal when any non-pass action exists.
    - Until an enemy cell is visible, moves onto the own general or own castle
      are illegal (do not stack idle piles on structures). Leaving a structure
      onto own land stays legal so a large stack can evacuate toward fog.
    """
    base = legal_mask(obs, memory, cost_grid=cost_grid)
    mask = np.asarray(base, dtype=bool).copy()
    nonpass = mask.copy()
    nonpass[PASS_INDEX] = False
    if np.any(nonpass):
        mask[PASS_INDEX] = False

    if enemy_is_visible(obs, memory):
        if not np.any(mask):
            return np.asarray(base, dtype=bool).copy()
        return mask

    H, W = int(obs.H), int(obs.W)
    own_struct = own_structure_mask(obs, memory)

    for idx in np.flatnonzero(mask):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        action = decode_action(idx)
        kind = int(action[0])
        if kind == 2:
            continue
        if kind != 0:
            continue
        sr, sc, d = int(action[1]), int(action[2]), int(action[3])
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            mask[idx] = False
            continue
        # Ban stacking onto general/castle before contact.
        if own_struct[tr, tc]:
            mask[idx] = False

    if not np.any(mask):
        restored = np.asarray(base, dtype=bool).copy()
        nonpass = restored.copy()
        nonpass[PASS_INDEX] = False
        if np.any(nonpass):
            restored[PASS_INDEX] = False
            return restored
        return restored
    return mask


def general_capture_indices(obs, memory: VisibleMemory, mask: Array) -> list[int]:
    """Legal move indices whose destination is a visible enemy general."""
    types, owners, _ = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    out: list[int] = []
    enemy_gen = (types == TYPE_GENERAL) & (owners == 2)
    enemy_gen |= memory.known_enemy_general & (owners == 2)
    for r, c in np.argwhere(enemy_gen):
        r, c = int(r), int(c)
        for d in range(4):
            sr = r - int(DIRECTIONS[d, 0])
            sc = c - int(DIRECTIONS[d, 1])
            if not (0 <= sr < H and 0 <= sc < W):
                continue
            if owners[sr, sc] != 1:
                continue
            for split in (0, 1):
                idx = encode_action((0, sr, sc, d, split))
                if mask[idx]:
                    out.append(idx)
    return out


def visible_enemy_source_interaction_indices(
    obs, memory: VisibleMemory, mask: Array
) -> list[int]:
    """Legal moves onto or from a cell adjacent to a visible enemy army source."""
    _types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    enemy_sources = (owners == 2) & (armies >= 1)
    if not np.any(enemy_sources):
        return []
    interact_dest = enemy_sources.copy()
    for r, c in np.argwhere(enemy_sources):
        r, c = int(r), int(c)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                interact_dest[nr, nc] = True
    out: list[int] = []
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        action = decode_action(idx)
        if int(action[0]) != 0:
            continue
        sr, sc, d = int(action[1]), int(action[2]), int(action[3])
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        if interact_dest[tr, tc] or interact_dest[sr, sc]:
            out.append(idx)
    return out


def frontier_expand_indices(obs, memory: VisibleMemory, mask: Array) -> list[int]:
    """Legal moves whose destination is currently unowned (neutral)."""
    _types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    scored: list[tuple[int, int]] = []
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        action = decode_action(idx)
        if int(action[0]) != 0:
            continue
        sr, sc, d = int(action[1]), int(action[2]), int(action[3])
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        if int(owners[tr, tc]) != OWNER_NEUTRAL:
            continue
        scored.append((int(armies[sr, sc]), idx))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [idx for _army, idx in scored]


def enemy_attack_indices(obs, memory: VisibleMemory, mask: Array) -> list[int]:
    """Legal moves whose destination is enemy-owned land."""
    _types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    scored: list[tuple[int, int]] = []
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        action = decode_action(idx)
        if int(action[0]) != 0:
            continue
        sr, sc, d = int(action[1]), int(action[2]), int(action[3])
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        if int(owners[tr, tc]) != OWNER_ENEMY:
            continue
        scored.append((int(armies[sr, sc]), idx))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [idx for _army, idx in scored]


def enemy_seek_target(
    obs,
    memory: VisibleMemory,
    belief: Optional[Any] = None,
) -> Optional[tuple[int, int]]:
    """Drive toward latched/visible enemy general, else nearest enemy land.

    Prefers the enemy cell closest (Manhattan) to the largest own stack so the
    king stack does not aim at a centroid behind mountains.
    """
    del belief
    goals = seek_goals(obs, memory)
    if not goals:
        return None
    if len(goals) == 1:
        return goals[0]
    _types, owners, armies = _as_grids(obs)
    own = owners == 1
    if not np.any(own):
        return goals[0]
    max_a = int(armies[own].max())
    locs = np.argwhere(own & (armies == max_a))
    kr, kc = int(locs[0, 0]), int(locs[0, 1])
    return min(goals, key=lambda g: abs(g[0] - kr) + abs(g[1] - kc))


def apply_pre_contact_prior(
    prior: Array,
    obs,
    memory: VisibleMemory,
    *,
    mask: Optional[Array] = None,
    belief: Optional[Any] = None,
) -> Array:
    """Reshape root prior: expand pre-contact, seek after contact."""
    prior_a = np.asarray(prior, dtype=np.float64).reshape(-1).copy()
    if mask is None:
        mask = play_mask(obs, memory)
    mask_a = np.asarray(mask, dtype=bool).reshape(-1)
    prior_a = np.where(mask_a, np.maximum(prior_a, 0.0), 0.0)

    _types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    out = np.zeros_like(prior_a)
    seen_enemy = enemy_is_visible(obs, memory)

    for idx in general_capture_indices(obs, memory, mask_a):
        action = decode_action(idx)
        sr, sc = int(action[1]), int(action[2])
        out[int(idx)] = 1_000_000.0 * float(max(int(armies[sr, sc]), 1))

    turn = int(getattr(obs, "turn", 0))
    castle_w = castle_timing_weight(turn)
    for idx in np.flatnonzero(mask_a):
        idx = int(idx)
        action = decode_action(idx)
        if int(action[0]) != 2:
            continue
        r, c = int(action[1]), int(action[2])
        army = float(max(int(armies[r, c]), 1))
        nn = max(float(prior_a[idx]), 1e-6)
        # Prefer cells that can still hold a useful remnant after the spend.
        out[idx] = max(
            out[idx],
            nn * castle_w * (10.0 + 0.2 * min(army, 100.0)),
        )

    if seen_enemy:
        target = enemy_seek_target(obs, memory, belief=belief)
        goals = seek_goals(obs, memory)
        dist_field = path_distance_field(obs, goals) if goals else None
        king = king_cell(obs)
        king_dist = path_distance_field(obs, [king]) if king is not None else None
        share, max_own, tot = army_concentration(obs)
        gen_known = enemy_general_visible(obs, memory)
        for idx in np.flatnonzero(mask_a):
            idx = int(idx)
            if idx == PASS_INDEX:
                continue
            action = decode_action(idx)
            if int(action[0]) == 2:
                continue  # already scored with castle timing
            if int(action[0]) != 0:
                out[idx] = max(out[idx], 0.01)
                continue
            sr, sc, d = int(action[1]), int(action[2]), int(action[3])
            tr = sr + int(DIRECTIONS[d, 0])
            tc = sc + int(DIRECTIONS[d, 1])
            if not (0 <= tr < H and 0 <= tc < W):
                continue
            army_i = int(armies[sr, sc])
            army_w = wave_weight(army_i)
            atk_w = attack_weight(army_i)
            dest_owner = int(owners[tr, tc])
            dest_army = int(armies[tr, tc])
            reveal = newly_revealed_cells(obs, tr, tc)
            efficiency = float(reveal) / float(explore_cost(dest_army))
            progress = path_progress(
                sr, sc, tr, tc, dist_field, fallback_target=target
            )
            bias = direction_bias(progress, dest_owner)
            surplus = float(max(army_i - dest_army - 1, 0))
            nn = max(float(prior_a[idx]), 1e-6)
            thrash = tip_thrash_factor(army_i, max_own, tot)
            gather = stack_gather_factor(
                army_i, dest_army, dest_owner, progress=progress
            )

            if dest_owner == OWNER_ENEMY and surplus > 0.0:
                # Strong attack reward: raw NN rarely proposes takes (probe ~0.8%).
                score = (
                    nn
                    * atk_w
                    * thrash
                    * bias
                    * (
                        120.0
                        + 40.0 * max(progress, 0.0)
                        + 8.0 * float(reveal)
                        + 3.5 * min(surplus, 80.0)
                    )
                )
                if gen_known:
                    score *= 1.35
            elif dest_owner == OWNER_NEUTRAL:
                # Fog carve that shortens the path to enemy is a real attack prep.
                score = (
                    nn
                    * atk_w
                    * thrash
                    * bias
                    * (
                        45.0
                        + 30.0 * max(progress, 0.0)
                        + 12.0 * efficiency
                        + 8.0 * float(reveal)
                    )
                )
            elif dest_owner == 1:
                # Own corridor: reward path-toward-enemy; crush retreat.
                # Also reward hinterland tips gathering into the king stack.
                k_prog = 0.0
                if king_dist is not None and not is_committed_army(
                    army_i, max_own, tot, share
                ):
                    k_prog = path_progress(
                        sr, sc, tr, tc, king_dist, fallback_target=king
                    )
                if k_prog > 0.0 and share < GATHER_SHARE_MIN:
                    score = (
                        nn
                        * army_w
                        * (14.0 + 22.0 * k_prog)
                        * stack_gather_factor(
                            army_i, dest_army, 1, progress=k_prog
                        )
                    )
                elif progress > 0.0:
                    score = (
                        nn
                        * atk_w
                        * thrash
                        * bias
                        * (18.0 + 32.0 * progress)
                        * gather
                    )
                else:
                    # Raw NN often picks retreat; keep mass near zero here.
                    score = nn * 0.008 * army_w * bias * gather
            else:
                score = nn * 0.01 * army_w * gather
            out[idx] = max(out[idx], score)
    else:
        # No enemy yet: hunt fog with a formed wave; urgency rises with turn.
        # Idle piles on general/castle are heavily discouraged.
        turn = int(getattr(obs, "turn", 0))
        urgency = fog_urgency(turn, enemy_seen=False)
        own_struct = own_structure_mask(obs, memory)
        fog_target = enemy_seek_target(obs, memory, belief=belief)
        fog_goals = seek_goals(obs, memory)
        fog_dist = path_distance_field(obs, fog_goals) if fog_goals else None
        frontier = frontier_expand_indices(obs, memory, mask_a)
        frontier_sources: set[tuple[int, int]] = set()
        for idx in frontier:
            action = decode_action(idx)
            sr, sc, d = int(action[1]), int(action[2]), int(action[3])
            tr = sr + int(DIRECTIONS[d, 0])
            tc = sc + int(DIRECTIONS[d, 1])
            frontier_sources.add((sr, sc))
            reveal = newly_revealed_cells(obs, tr, tc) if 0 <= tr < H and 0 <= tc < W else 0
            dest_army = int(armies[tr, tc]) if 0 <= tr < H and 0 <= tc < W else 0
            eff = float(reveal) / float(explore_cost(dest_army))
            src_army = int(armies[sr, sc])
            ew = explore_wave_weight(src_army)
            nn = max(float(prior_a[int(idx)]), 1e-6)
            progress = path_progress(
                sr, sc, tr, tc, fog_dist, fallback_target=fog_target
            )
            bias = direction_bias(progress, OWNER_NEUTRAL)
            score = (
                nn
                * ew
                * urgency
                * bias
                * (4.0 + 12.0 * float(reveal) + 7.0 * eff + 6.0 * max(progress, 0.0))
            )
            # Leaving a fat structure into fog is the right pre-contact move.
            if bool(own_struct[sr, sc]) and src_army >= STRUCTURE_IDLE_ARMY:
                score *= 2.5 + 0.04 * float(min(src_army, 80))
            out[int(idx)] = max(out[int(idx)], score)
        # Own corridor: reward toward fog target; punish retreat / circles.
        for idx in np.flatnonzero(mask_a):
            idx = int(idx)
            if idx == PASS_INDEX:
                continue
            action = decode_action(idx)
            if int(action[0]) != 0:
                continue
            sr, sc, d = int(action[1]), int(action[2]), int(action[3])
            tr = sr + int(DIRECTIONS[d, 0])
            tc = sc + int(DIRECTIONS[d, 1])
            if not (0 <= tr < H and 0 <= tc < W):
                continue
            if int(owners[tr, tc]) != 1:
                continue
            src_army = int(armies[sr, sc])
            dest_army = int(armies[tr, tc])
            nn = max(float(prior_a[idx]), 1e-6)
            progress = path_progress(
                sr, sc, tr, tc, fog_dist, fallback_target=fog_target
            )
            bias = direction_bias(progress, 1)
            gather = stack_gather_factor(
                src_army, dest_army, 1, progress=progress
            )
            tip_bonus = 2.0 if (tr, tc) in frontier_sources else 1.0
            if bool(own_struct[sr, sc]) and src_army >= STRUCTURE_IDLE_ARMY:
                if progress < 0.0:
                    out[idx] = max(out[idx], nn * 0.01)
                    continue
                out[idx] = max(
                    out[idx],
                    nn
                    * urgency
                    * tip_bonus
                    * bias
                    * (5.0 + 0.1 * float(min(src_army, 100)) + 8.0 * max(progress, 0.0))
                    * gather,
                )
                continue
            if progress > 0.0:
                ew = explore_wave_weight(src_army)
                out[idx] = max(
                    out[idx],
                    nn
                    * ew
                    * urgency
                    * tip_bonus
                    * bias
                    * (0.8 + 4.0 * progress)
                    * gather,
                )
            else:
                out[idx] = max(out[idx], nn * 0.008 * bias * gather)

    out = np.where(mask_a, np.maximum(out, 0.0), 0.0)
    if float(out.sum()) > 0.0:
        return out / float(out.sum())
    total = float(prior_a.sum())
    if total > 0.0:
        return prior_a / total
    n = int(mask_a.sum())
    if n <= 0:
        return prior_a
    return mask_a.astype(np.float64) / float(n)


apply_early_expand_prior = apply_pre_contact_prior


def newly_revealed_cells(obs, dest_r: int, dest_c: int) -> int:
    """How many currently invisible cells become visible if we own ``dest``."""
    _types, owners, _armies = _as_grids(obs)
    owned = owners == 1
    if bool(owned[dest_r, dest_c]):
        return 0
    new_owned = owned.copy()
    new_owned[dest_r, dest_c] = True
    before = visibility_mask(owned)
    after = visibility_mask(new_owned)
    return int((after & ~before).sum())


def explore_cost(dest_army: int) -> int:
    """Armies spent to take a cell (leave 1 behind on a winning move)."""
    return 1 + max(int(dest_army), 0)


def explore_efficiency(obs, dest_r: int, dest_c: int, dest_army: int) -> float:
    """New vision per army spent — cheap fog scouting ranks high."""
    return float(newly_revealed_cells(obs, dest_r, dest_c)) / float(
        explore_cost(dest_army)
    )


def enemy_general_visible(obs, memory: VisibleMemory) -> bool:
    """True when the enemy general is in view or already latched in memory."""
    types, owners, _ = _as_grids(obs)
    if np.any(((types == TYPE_GENERAL) & (owners == OWNER_ENEMY))):
        return True
    return bool(np.any(memory.known_enemy_general))


def best_fog_explore_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Expand into fog/neutral with a formed wave; urgency rises until contact.

    Prefers high reveal × wave size. One-man tips lose to a real explore wave.
    """
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    mask = play_mask(obs, memory)
    urgency = fog_urgency(
        int(getattr(obs, "turn", 0)),
        enemy_seen=enemy_is_visible(obs, memory),
    )
    fog_goals = seek_goals(obs, memory)
    fog_dist = path_distance_field(obs, fog_goals) if fog_goals else None
    fog_target = enemy_seek_target(obs, memory)
    best: Optional[tuple[float, float, int, int, Action5]] = None
    best_weak: Optional[tuple[float, float, int, int, Action5]] = None
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        action = decode_action(idx)
        if int(action[0]) != 0 or int(action[4]) != 0:
            continue
        move: Action5 = tuple(int(x) for x in action)  # type: ignore[assignment]
        if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
            continue
        sr, sc, d = int(action[1]), int(action[2]), int(action[3])
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        if not _is_passable_type(int(types[tr, tc])):
            continue
        dest_owner = int(owners[tr, tc])
        if dest_owner == 1 or dest_owner == OWNER_ENEMY:
            continue
        src_army = int(armies[sr, sc])
        dest_army = int(armies[tr, tc])
        if dest_owner == OWNER_NEUTRAL and int(types[tr, tc]) != TYPE_FOG:
            if not _is_visible_neutral(dest_owner, int(types[tr, tc])):
                continue
            if src_army <= dest_army + 1:
                continue
        elif int(types[tr, tc]) == TYPE_FOG:
            if src_army < 2:
                continue
        else:
            continue
        reveal = newly_revealed_cells(obs, tr, tc)
        if reveal <= 0:
            continue
        ew = explore_wave_weight(src_army)
        progress = path_progress(
            sr, sc, tr, tc, fog_dist, fallback_target=fog_target
        )
        bias = direction_bias(progress, OWNER_NEUTRAL)
        score = float(reveal) * urgency * ew * bias * (1.0 + 0.8 * max(progress, 0.0))
        key = (score, ew, reveal, -idx)
        if src_army >= EXPLORE_WAVE_MIN:
            if best is None or key > best[:4]:
                best = (score, ew, reveal, -idx, move)
        else:
            if best_weak is None or key > best_weak[:4]:
                best_weak = (score, ew, reveal, -idx, move)
    if best is not None:
        return best[4]
    if best_weak is not None:
        return best_weak[4]
    return None


def best_fog_approach_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """March a formed wave onto a thin frontier tip when no expand is ready."""
    _types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    mask = play_mask(obs, memory)
    urgency = fog_urgency(
        int(getattr(obs, "turn", 0)),
        enemy_seen=enemy_is_visible(obs, memory),
    )
    fog_goals = seek_goals(obs, memory)
    fog_dist = path_distance_field(obs, fog_goals) if fog_goals else None
    fog_target = enemy_seek_target(obs, memory)
    frontier = frontier_expand_indices(obs, memory, mask)
    tips: set[tuple[int, int]] = set()
    for idx in frontier:
        action = decode_action(idx)
        tips.add((int(action[1]), int(action[2])))
    if not tips:
        return None
    best: Optional[tuple[float, Action5]] = None
    for r, c in np.argwhere(owners == 1):
        r, c = int(r), int(c)
        army = int(armies[r, c])
        if army < EXPLORE_WAVE_MIN:
            continue
        for d in range(4):
            tr = r + int(DIRECTIONS[d, 0])
            tc = c + int(DIRECTIONS[d, 1])
            if not (0 <= tr < H and 0 <= tc < W):
                continue
            if (tr, tc) not in tips:
                continue
            if int(owners[tr, tc]) != 1:
                continue
            if int(armies[tr, tc]) > COMMIT_DEST_ARMY_MAX:
                continue
            move: Action5 = (0, r, c, d, 0)
            idx = encode_action(move)
            if not mask[idx]:
                continue
            if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
                continue
            tip_reveal = 0
            for d2 in range(4):
                nr = tr + int(DIRECTIONS[d2, 0])
                nc = tc + int(DIRECTIONS[d2, 1])
                if 0 <= nr < H and 0 <= nc < W and int(owners[nr, nc]) == OWNER_NEUTRAL:
                    tip_reveal = max(tip_reveal, newly_revealed_cells(obs, nr, nc))
            progress = path_progress(
                r, c, tr, tc, fog_dist, fallback_target=fog_target
            )
            if progress < 0.0:
                continue
            score = (
                urgency
                * explore_wave_weight(army)
                * direction_bias(progress, 1)
                * (1.0 + 4.0 * float(tip_reveal) + 3.0 * max(progress, 0.0))
                * stack_gather_factor(army, int(armies[tr, tc]), 1, progress=progress)
            )
            if best is None or score > best[0]:
                best = (score, move)
    return None if best is None else best[1]


def best_structure_evacuate_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Move a fat pile off own general/castle toward fog or the enemy.

    Prefers a direct fog/neutral expand from the structure. Otherwise marches
    onto a thin own cell, especially a frontier tip. From the general, prefer
    half-moves so a garrison stays behind against rushes.
    """
    idle = structure_idle_army(obs, memory)
    if idle < STRUCTURE_IDLE_ARMY:
        return None
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    mask = play_mask(obs, memory)
    own_struct = own_structure_mask(obs, memory)
    seen = enemy_is_visible(obs, memory)
    # After contact: only unpark a huge idle pile; small evacuations strip the
    # general and lose to rush bots like macaria. Never evacuate while threatened.
    if seen and (idle < 40 or general_is_threatened(obs, memory)):
        return None
    urgency = fog_urgency(int(getattr(obs, "turn", 0)), enemy_seen=seen)
    fog_goals = seek_goals(obs, memory)
    fog_dist = path_distance_field(obs, fog_goals) if fog_goals else None
    fog_target = enemy_seek_target(obs, memory)
    frontier = frontier_expand_indices(obs, memory, mask)
    tips: set[tuple[int, int]] = set()
    for idx in frontier:
        action = decode_action(idx)
        tips.add((int(action[1]), int(action[2])))

    best: Optional[tuple[float, Action5]] = None
    for r, c in np.argwhere(own_struct):
        r, c = int(r), int(c)
        src_army = int(armies[r, c])
        if src_army < STRUCTURE_IDLE_ARMY:
            continue
        is_gen = int(types[r, c]) == TYPE_GENERAL
        # Half-move from the general keeps a garrison; full move from castles.
        halves = (1, 0) if (is_gen and src_army > 2) else (0,)
        for half in halves:
            for d in range(4):
                tr = r + int(DIRECTIONS[d, 0])
                tc = c + int(DIRECTIONS[d, 1])
                if not (0 <= tr < H and 0 <= tc < W):
                    continue
                move: Action5 = (0, r, c, d, half)
                idx = encode_action(move)
                if not mask[idx]:
                    continue
                if blocks_oscillation(
                    move, prev_action, obs, recent_actions=recent_actions
                ):
                    continue
                dest_owner = int(owners[tr, tc])
                dest_type = int(types[tr, tc])
                dest_army = int(armies[tr, tc])
                progress = path_progress(
                    r, c, tr, tc, fog_dist, fallback_target=fog_target
                )
                if seen and progress < 0.0 and dest_owner != OWNER_ENEMY:
                    continue
                garrison_bonus = 1.4 if half == 1 and is_gen else 1.0
                if dest_owner == OWNER_ENEMY:
                    if src_army <= dest_army + 1:
                        continue
                    score = (
                        urgency
                        * garrison_bonus
                        * (200.0 + 40.0 * max(progress, 0.0) + float(src_army))
                    )
                elif dest_owner != 1 and _is_passable_type(dest_type):
                    if dest_owner == OWNER_NEUTRAL and src_army <= dest_army + 1:
                        continue
                    reveal = newly_revealed_cells(obs, tr, tc)
                    score = (
                        urgency
                        * garrison_bonus
                        * explore_wave_weight(src_army)
                        * (
                            80.0
                            + 30.0 * max(progress, 0.0)
                            + 12.0 * float(reveal)
                        )
                    )
                elif dest_owner == 1:
                    tip_bonus = 25.0 if (tr, tc) in tips else 0.0
                    score = (
                        urgency
                        * garrison_bonus
                        * (20.0 + tip_bonus + 35.0 * max(progress, 0.0))
                        * stack_gather_factor(
                            src_army, dest_army, 1, progress=progress
                        )
                    )
                else:
                    continue
                if best is None or score > best[0]:
                    best = (score, move)
    return None if best is None else best[1]


def best_castle_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Best legal castle build under the early/late timing curve."""
    del prev_action, recent_actions
    _types, _owners, armies = _as_grids(obs)
    mask = play_mask(obs, memory)
    tw = castle_timing_weight(int(getattr(obs, "turn", 0)))
    if tw < 0.45:
        return None
    best: Optional[tuple[float, Action5]] = None
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        action = decode_action(idx)
        if int(action[0]) != 2:
            continue
        r, c = int(action[1]), int(action[2])
        army = int(armies[r, c])
        move: Action5 = (2, r, c, 0, 0)
        score = tw * (10.0 + 0.25 * float(min(army, 100)))
        if best is None or score > best[0]:
            best = (score, move)
    return None if best is None else best[1]


def best_cheap_scout_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Probe after contact: prefer winning enemy takes, then fog reveals.

    Enemy tiles rank above fog/neutral. Zero-reveal mop-up stays allowed but
    ranks below any positive-reveal enemy take. Small stacks preferred on ties.
    """
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    mask = play_mask(obs, memory)
    best: Optional[tuple[int, float, int, int, int, Action5]] = None
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        action = decode_action(idx)
        if int(action[0]) != 0 or int(action[4]) != 0:
            continue
        move: Action5 = tuple(int(x) for x in action)  # type: ignore[assignment]
        if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
            continue
        sr, sc, d = int(action[1]), int(action[2]), int(action[3])
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        if not _is_passable_type(int(types[tr, tc])):
            continue
        dest_owner = int(owners[tr, tc])
        if dest_owner == 1:
            continue
        src_army = int(armies[sr, sc])
        dest_army = int(armies[tr, tc])
        dest_type = int(types[tr, tc])
        if dest_owner == OWNER_ENEMY:
            if src_army <= dest_army + 1:
                continue
            reveal = newly_revealed_cells(obs, tr, tc)
            class_rank = 2 if reveal > 0 else 1
        elif dest_type == TYPE_FOG:
            if src_army < 2:
                continue
            reveal = newly_revealed_cells(obs, tr, tc)
            if reveal <= 0:
                continue
            class_rank = 0
        elif _is_visible_neutral(dest_owner, dest_type):
            if src_army <= dest_army + 1:
                continue
            reveal = newly_revealed_cells(obs, tr, tc)
            if reveal <= 0:
                continue
            class_rank = 0
        else:
            continue
        eff = explore_efficiency(obs, tr, tc, dest_army)
        key = (class_rank, eff, reveal, -src_army, -idx)
        if best is None or key > best[:5]:
            best = (class_rank, eff, reveal, -src_army, -idx, move)
    if best is None:
        return None
    return best[5]


def smoke_like_expand_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """First visible-neutral expand in scan order (same idea as bots/smoke)."""
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    mask = play_mask(obs, memory)
    for r in range(H):
        for c in range(W):
            if int(owners[r, c]) != 1:
                continue
            src_army = int(armies[r, c])
            if src_army <= 1:
                continue
            for d in range(4):
                nr = r + int(DIRECTIONS[d, 0])
                nc = c + int(DIRECTIONS[d, 1])
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if not _is_passable_type(int(types[nr, nc])):
                    continue
                dest_owner = int(owners[nr, nc])
                dest_type = int(types[nr, nc])
                dest_army = int(armies[nr, nc])
                if not _is_visible_neutral(dest_owner, dest_type):
                    continue
                if src_army <= dest_army + 1:
                    continue
                move: Action5 = (0, r, c, d, 0)
                if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
                    continue
                idx = encode_action(move)
                if mask[idx]:
                    return move
    return None


def first_playable_move(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Any playable non-pass move in scan order (skips corridor reverse)."""
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    mask = play_mask(obs, memory)
    fallback: Optional[Action5] = None
    for r in range(H):
        for c in range(W):
            if int(owners[r, c]) != 1:
                continue
            if int(armies[r, c]) < 2:
                continue
            for d in range(4):
                nr = r + int(DIRECTIONS[d, 0])
                nc = c + int(DIRECTIONS[d, 1])
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if not _is_passable_type(int(types[nr, nc])):
                    continue
                move: Action5 = (0, r, c, d, 0)
                idx = encode_action(move)
                if not mask[idx]:
                    continue
                if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
                    if fallback is None:
                        fallback = move
                    continue
                return move
    for idx in np.flatnonzero(mask):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        move = tuple(int(x) for x in decode_action(idx))  # type: ignore[assignment]
        if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
            if fallback is None:
                fallback = move
            continue
        return move
    return fallback


def best_winning_attack(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Winning attack that prefers a real stack, not a tip thrash."""
    mask = play_mask(obs, memory)
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    target = enemy_seek_target(obs, memory)
    goals = seek_goals(obs, memory)
    dist_field = path_distance_field(obs, goals) if goals else None
    max_own = largest_own_army(obs)
    best: Optional[tuple[float, Action5]] = None
    for idx in enemy_attack_indices(obs, memory, mask):
        action = decode_action(idx)
        if int(action[4]) != 0:
            continue
        sr, sc, d = int(action[1]), int(action[2]), int(action[3])
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        src_army = int(armies[sr, sc])
        dest_army = int(armies[tr, tc])
        if src_army <= dest_army + 1:
            continue
        move: Action5 = (0, sr, sc, d, 0)
        if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
            continue
        surplus = src_army - dest_army - 1
        progress = path_progress(
            sr, sc, tr, tc, dist_field, fallback_target=target
        )
        bias = direction_bias(progress, OWNER_ENEMY)
        score = (
            35.0 * max(progress, 0.0)
            + 4.0 * float(min(surplus, 120))
            + 18.0 * attack_weight(src_army)
            + (30.0 if int(types[tr, tc]) == TYPE_GENERAL else 0.0)
        ) * tip_thrash_factor(src_army, max_own, int(armies[owners == 1].sum()) if np.any(owners == 1) else max_own) * bias
        if best is None or score > best[0]:
            best = (score, move)
    return None if best is None else best[1]


def seek_kill_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Commit large stacks into approach/attack — not tip thrash or pile merges."""
    mask = play_mask(obs, memory)
    caps = general_capture_indices(obs, memory, mask)
    if caps:
        _types, _owners, armies = _as_grids(obs)
        best_cap = max(
            caps,
            key=lambda i: int(armies[decode_action(i)[1], decode_action(i)[2]]),
        )
        return tuple(int(x) for x in decode_action(best_cap))  # type: ignore[return-value]

    gen_known = enemy_general_visible(obs, memory)
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    target = enemy_seek_target(obs, memory)
    goals = seek_goals(obs, memory)
    dist_field = path_distance_field(obs, goals) if goals else None
    max_own = largest_own_army(obs)
    king = king_cell(obs)
    rally = rally_cell(obs, memory)
    king_dist = path_distance_field(obs, [rally]) if rally is not None else None
    share, _mx, tot = army_concentration(obs)

    # Only force an immediate attack when it already commits a real stack.
    attack = best_winning_attack(
        obs, memory, prev_action=prev_action, recent_actions=recent_actions
    )
    if attack is not None:
        src_a = int(armies[attack[1], attack[2]])
        if tip_thrash_factor(src_a, max_own, tot) >= 0.55:
            return attack

    if not gen_known:
        scout = best_cheap_scout_action(
            obs, memory, prev_action=prev_action, recent_actions=recent_actions
        )
        # Prefer scout only when it is an enemy take from a real stack, or no march.
        if scout is not None:
            sr, sc = int(scout[1]), int(scout[2])
            tr = sr + int(DIRECTIONS[scout[3], 0])
            tc = sc + int(DIRECTIONS[scout[3], 1])
            if (
                0 <= tr < H
                and 0 <= tc < W
                and int(owners[tr, tc]) == OWNER_ENEMY
                and tip_thrash_factor(int(armies[sr, sc]), max_own, tot) >= 0.55
            ):
                return scout

    stacks = [
        (int(armies[r, c]), int(r), int(c))
        for r, c in np.argwhere(owners == 1)
        if int(armies[r, c]) >= 2
    ]
    stacks.sort(key=lambda t: -t[0])

    def score_move(sr: int, sc: int, d: int) -> Optional[tuple[float, Action5]]:
        tr = sr + int(DIRECTIONS[d, 0])
        tc = sc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            return None
        move: Action5 = (0, sr, sc, d, 0)
        idx = encode_action(move)
        if not mask[idx]:
            return None
        if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
            return None
        army = int(armies[sr, sc])
        dest_owner = int(owners[tr, tc])
        dest_army = int(armies[tr, tc])
        dest_type = int(types[tr, tc])
        reveal = newly_revealed_cells(obs, tr, tc)
        ww = wave_weight(army)
        aw = attack_weight(army)
        thrash = tip_thrash_factor(army, max_own, tot)
        progress = path_progress(
            sr, sc, tr, tc, dist_field, fallback_target=target
        )
        bias = direction_bias(progress, dest_owner)
        is_tip = not is_committed_army(army, max_own, tot, share)

        # Never empty the general into a fog rush — hold or fight in place.
        if (
            int(types[sr, sc]) == TYPE_GENERAL
            and general_is_threatened(obs, memory)
            and dest_owner != OWNER_ENEMY
        ):
            return None

        # Tips: only gather toward the rally (or take a winning enemy tile).
        if is_tip and dest_owner == 1:
            k_prog = path_progress(
                sr, sc, tr, tc, king_dist, fallback_target=rally
            )
            if k_prog <= 0.0:
                return None
            # After contact: do not feed the general/castle.
            if enemy_is_visible(obs, memory):
                if bool(own_structure_mask(obs, memory)[tr, tc]):
                    return None
            gather = stack_gather_factor(army, dest_army, 1, progress=k_prog)
            score = ww * (12.0 + 20.0 * k_prog) * gather
            if share < GATHER_SHARE_MIN:
                score *= 2.5
            return score, move

        # Never march away from the path to the seek goals on own/neutral land.
        if progress < 0.0 and dest_owner != OWNER_ENEMY:
            return None
        if dest_owner == OWNER_ENEMY:
            if army <= dest_army + 1:
                return None
            surplus = float(min(army - dest_army - 1, 120))
            score = (
                aw
                * thrash
                * bias
                * (80.0 + 30.0 * max(progress, 0.0) + 3.5 * surplus + 8.0 * reveal)
            )
        elif dest_type == TYPE_FOG and dest_owner == OWNER_NEUTRAL:
            if is_tip:
                return None
            if progress <= 0.0 and reveal <= 0:
                return None
            score = (
                aw
                * thrash
                * bias
                * (28.0 + 22.0 * max(progress, 0.0) + 10.0 * reveal)
            )
        elif dest_owner == OWNER_NEUTRAL and _is_visible_neutral(dest_owner, dest_type):
            if is_tip:
                return None
            score = (
                aw
                * thrash
                * bias
                * (
                    14.0
                    + 14.0 * max(progress, 0.0)
                    + 5.0 * explore_efficiency(obs, tr, tc, dest_army)
                )
            )
        elif dest_owner == 1:
            gather = stack_gather_factor(army, dest_army, dest_owner, progress=progress)
            if progress > 0.0:
                score = aw * thrash * bias * (36.0 + 42.0 * progress) * gather
            else:
                if army >= STACK_GATHER_BAN:
                    return None
                score = ww * bias * 0.2 * gather
        else:
            return None
        return score, move

    options: list[tuple[float, Action5]] = []
    for _army, sr, sc in stacks[:12]:
        for d in range(4):
            scored = score_move(sr, sc, d)
            if scored is not None:
                options.append(scored)
    if not options:
        # Fall back to any winning attack / scout if marches were blocked.
        if attack is not None:
            return attack
        return best_cheap_scout_action(
            obs, memory, prev_action=prev_action, recent_actions=recent_actions
        )
    options.sort(key=lambda t: -t[0])
    return options[0][1]


def best_defend_general_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Stop a rush: clear adjacent enemies, or march the king home to garrison."""
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    gen = np.argwhere(
        np.asarray(memory.own_general, dtype=bool)
        | ((types == TYPE_GENERAL) & (owners == 1))
    )
    if gen.size == 0:
        return None
    gr, gc = int(gen[0, 0]), int(gen[0, 1])
    gen_army = int(armies[gr, gc])
    mask = play_mask(obs, memory)
    # Enemy adjacent to the general — always clear if we can.
    best_hit: Optional[tuple[float, Action5]] = None
    enemy_adj = False
    for d in range(4):
        er = gr + int(DIRECTIONS[d, 0])
        ec = gc + int(DIRECTIONS[d, 1])
        if not (0 <= er < H and 0 <= ec < W):
            continue
        if int(owners[er, ec]) != OWNER_ENEMY:
            continue
        enemy_adj = True
        for sd in range(4):
            sr = er - int(DIRECTIONS[sd, 0])
            sc = ec - int(DIRECTIONS[sd, 1])
            if not (0 <= sr < H and 0 <= sc < W):
                continue
            if int(owners[sr, sc]) != 1:
                continue
            src = int(armies[sr, sc])
            if src <= int(armies[er, ec]) + 1:
                continue
            move: Action5 = (0, sr, sc, sd, 0)
            idx = encode_action(move)
            if not mask[idx]:
                continue
            if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
                continue
            score = float(src) + (50.0 if (sr, sc) == (gr, gc) else 0.0)
            if best_hit is None or score > best_hit[0]:
                best_hit = (score, move)
    if best_hit is not None:
        return best_hit[1]

    if not enemy_is_visible(obs, memory):
        return None

    # Threat: short BFS path to our general. Manhattan-8 + thin-gen alone
    # yanked the king home every turn and cancelled every push.
    enemy_cells = np.argwhere(owners == OWNER_ENEMY)
    if enemy_cells.size == 0:
        return None
    gen_dist = path_distance_field(obs, [(gr, gc)])
    nearest_path = 10**9
    for r, c in enemy_cells:
        d = int(gen_dist[int(r), int(c)])
        if 0 <= d < nearest_path:
            nearest_path = d
    if nearest_path >= 10**9:
        nearest_path = min(
            int(abs(int(r) - gr) + abs(int(c) - gc)) for r, c in enemy_cells
        )
    threatened = enemy_adj or nearest_path <= DEFEND_PATH_NEAR or (
        nearest_path <= DEFEND_PATH_THIN and gen_army < DEFEND_GEN_THIN
    )
    if not threatened:
        return None

    # March the king stack toward the general (not tip feed).
    king = king_cell(obs)
    if king is None:
        return None
    kr, kc = king
    king_a = int(armies[kr, kc])
    share, _mx, tot = army_concentration(obs)
    # King already on the general: only strike a winnable adjacent enemy.
    # Do not walk the garrison off the tile — that is how fog rushes win.
    if (kr, kc) == (gr, gc):
        return None

    if king_a < EXPLORE_WAVE_MIN:
        return None
    if not is_committed_army(king_a, _mx, tot, share):
        # Dispersed: gather into the king first; do not walk a tip home.
        return None
    best: Optional[tuple[float, Action5]] = None
    for d in range(4):
        tr = kr + int(DIRECTIONS[d, 0])
        tc = kc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        move: Action5 = (0, kr, kc, d, 0)
        idx = encode_action(move)
        if not mask[idx]:
            continue
        if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
            continue
        dest_o = int(owners[tr, tc])
        progress = path_progress(
            kr, kc, tr, tc, gen_dist, fallback_target=(gr, gc)
        )
        if dest_o == OWNER_ENEMY:
            if king_a <= int(armies[tr, tc]) + 1:
                continue
            score = 200.0 + 40.0 * max(progress, 0.0)
        elif dest_o == 1 and progress > 0.0:
            score = 80.0 + 50.0 * progress
        elif dest_o == OWNER_NEUTRAL and progress > 0.0:
            score = 60.0 + 30.0 * progress
        else:
            continue
        if best is None or score > best[0]:
            best = (score, move)
    return None if best is None else best[1]


def best_king_commit_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Move the largest stack along the seek path when no fog expand is ready.

    Prevents the king from idling while hinterland tips feed it in place.
    """
    types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    king = king_cell(obs)
    if king is None:
        return None
    kr, kc = king
    src = int(armies[kr, kc])
    if src < EXPLORE_WAVE_MIN:
        return None
    goals = seek_goals(obs, memory)
    if not goals:
        return None
    dist = path_distance_field(obs, goals)
    target = enemy_seek_target(obs, memory)
    mask = play_mask(obs, memory)
    best: Optional[tuple[float, Action5]] = None
    for d in range(4):
        tr = kr + int(DIRECTIONS[d, 0])
        tc = kc + int(DIRECTIONS[d, 1])
        if not (0 <= tr < H and 0 <= tc < W):
            continue
        move: Action5 = (0, kr, kc, d, 0)
        idx = encode_action(move)
        if not mask[idx]:
            continue
        if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
            continue
        dest_o = int(owners[tr, tc])
        dest_t = int(types[tr, tc])
        dest_a = int(armies[tr, tc])
        progress = path_progress(kr, kc, tr, tc, dist, fallback_target=target)
        if progress <= 0.0 and dest_o != OWNER_ENEMY:
            continue
        if dest_o == OWNER_ENEMY:
            if src <= dest_a + 1:
                continue
            score = 100.0 + 40.0 * progress + float(src - dest_a)
        elif dest_o == OWNER_NEUTRAL or dest_t == TYPE_FOG:
            if not _is_passable_type(dest_t):
                continue
            reveal = newly_revealed_cells(obs, tr, tc)
            score = 50.0 + 30.0 * progress + 10.0 * float(reveal)
        elif dest_o == 1:
            score = 20.0 + 35.0 * progress
            score *= stack_gather_factor(src, dest_a, 1, progress=progress)
        else:
            continue
        score *= attack_weight(src)
        if best is None or score > best[0]:
            best = (score, move)
    return None if best is None else best[1]


def best_land_gather_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Optional[Action5]:
    """Sweep hinterland armies into the rally stack.

    Moves own cells with army >= 2 (except the rally cell) onto a neighbor that
    is closer to the rally. After contact the rally is off the general so land
    production forms a fighting stack instead of a parked gen pile.
    """
    _types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    rally = rally_cell(obs, memory)
    if rally is None:
        return None
    kr, kc = rally
    king_army = int(armies[kr, kc])
    share, _max_a, total = army_concentration(obs)
    # Always useful while dispersed; also sweep leftover 2+ tips after a fight.
    if total < 8:
        return None
    king_dist = path_distance_field(obs, [(kr, kc)])
    mask = play_mask(obs, memory)
    own_struct = own_structure_mask(obs, memory)
    seen = enemy_is_visible(obs, memory)
    best: Optional[tuple[float, Action5]] = None
    for r, c in np.argwhere(owners == 1):
        r, c = int(r), int(c)
        if (r, c) == (kr, kc):
            continue
        src = int(armies[r, c])
        if src < 2:
            continue
        # Never strip the general/castle garrison into the king.
        if bool(own_struct[r, c]) and src <= GENERAL_EVACUATE_MIN:
            continue
        # Do not break apart a committed secondary wave near the king size.
        if src >= COMMIT_ARMY_FRAC * king_army and share >= GATHER_SHARE_MIN:
            continue
        for d in range(4):
            tr = r + int(DIRECTIONS[d, 0])
            tc = c + int(DIRECTIONS[d, 1])
            if not (0 <= tr < H and 0 <= tc < W):
                continue
            if int(owners[tr, tc]) != 1:
                continue
            # After contact: do not feed the general/castle — form a front stack.
            # Exception: when the general is threatened, refill the garrison.
            if seen and bool(own_struct[tr, tc]):
                if not (
                    int(_types[tr, tc]) == TYPE_GENERAL
                    and general_is_threatened(obs, memory)
                ):
                    continue
            move: Action5 = (0, r, c, d, 0)
            idx = encode_action(move)
            if not mask[idx]:
                continue
            if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
                continue
            prog = path_progress(r, c, tr, tc, king_dist, fallback_target=rally)
            if prog <= 0.0:
                continue
            dest_a = int(armies[tr, tc])
            # Prefer feeding the rally / denser corridor over empty hinterland.
            dest_bonus = 3.0 if (tr, tc) == (kr, kc) else (
                1.5 if dest_a >= src else 1.0
            )
            # Smaller hinterland tips first so land production consolidates.
            size_pref = 1.0 + 4.0 / float(src)
            score = (
                float(prog)
                * dest_bonus
                * size_pref
                * (2.0 if share < GATHER_SHARE_MIN else 1.0)
            )
            if best is None or score > best[0]:
                best = (score, move)
    return None if best is None else best[1]


def select_play_action(
    obs,
    memory: VisibleMemory,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
) -> Action5:
    """Heuristic path: capture, cheap scout, fog explore, then seek.

    Used by the no-NN smoke grid. Full bot prefers ``constrain_nn_action`` so
    the network still chooses most moves.
    """
    mask = play_mask(obs, memory)
    caps = general_capture_indices(obs, memory, mask)
    if caps:
        _t, _o, armies = _as_grids(obs)
        best_cap = max(
            caps,
            key=lambda i: int(armies[decode_action(i)[1], decode_action(i)[2]]),
        )
        return tuple(int(x) for x in decode_action(best_cap))  # type: ignore[return-value]

    turn = int(getattr(obs, "turn", 0))
    gen_visible = enemy_general_visible(obs, memory)
    seen = enemy_is_visible(obs, memory)
    kw = {"prev_action": prev_action, "recent_actions": recent_actions}

    if gen_visible:
        kill = seek_kill_action(obs, memory, **kw)
        if kill is not None:
            return kill
        attack = best_winning_attack(obs, memory, **kw)
        if attack is not None:
            return attack

    share, max_a, _tot = army_concentration(obs)
    king = king_cell(obs)

    def _prefer_gather_over(move: Action5) -> bool:
        """True when a tip is freestyling while the board is still dispersed.

        Never interrupts the king stack — corridor marches and fog carves from
        the largest army stay highest priority.
        """
        if share >= GATHER_SHARE_MIN:
            return False
        ends = move_dest(move)
        if ends is None:
            return True
        sr, sc, tr, tc = ends
        if king is not None and (sr, sc) == king:
            return False
        owners = np.asarray(obs.owner_grid, dtype=np.int32)
        armies = np.asarray(obs.army_grid, dtype=np.int32)
        dest_o = int(owners[tr, tc])
        src_a = int(armies[sr, sc])
        if dest_o == OWNER_ENEMY and is_committed_army(src_a, max_a, _tot, share):
            return False
        return True

    if seen:
        defend = best_defend_general_action(obs, memory, **kw)
        if defend is not None:
            return defend
        # Fat pile parked on gen/castle after contact: march it out first.
        idle = structure_idle_army(obs, memory)
        if idle >= 40 and not general_is_threatened(obs, memory):
            evacuate = best_structure_evacuate_action(obs, memory, **kw)
            if evacuate is not None:
                return evacuate
        # Threatened gen: refill garrison before tip freestyle.
        if general_is_threatened(obs, memory):
            gather_home = best_land_gather_action(obs, memory, **kw)
            # Prefer gather that feeds the general when tips can reach it.
            if gather_home is not None:
                ends = move_dest(gather_home)
                if ends is not None:
                    _tr, _tc = ends[2], ends[3]
                    types = np.asarray(obs.type_grid, dtype=np.int32)
                    if int(types[_tr, _tc]) == TYPE_GENERAL:
                        return gather_home
            defend2 = best_defend_general_action(obs, memory, **kw)
            if defend2 is not None:
                return defend2
        # Prefer a real fighting stack pressing the enemy over tip gather.
        kill = seek_kill_action(obs, memory, **kw)
        if kill is not None:
            ends = move_dest(kill)
            if ends is not None:
                sr, sc, tr, tc = ends
                armies = np.asarray(obs.army_grid, dtype=np.int32)
                owners = np.asarray(obs.owner_grid, dtype=np.int32)
                src_a = int(armies[sr, sc])
                if is_committed_army(src_a, max_a, _tot, share) or (
                    king is not None
                    and (sr, sc) == king
                    and src_a >= STACK_GATHER_BAN
                ):
                    dest_o = int(owners[tr, tc])
                    if dest_o == OWNER_ENEMY:
                        return kill
                    goals = seek_goals(obs, memory)
                    dist = path_distance_field(obs, goals) if goals else None
                    tgt = enemy_seek_target(obs, memory)
                    prog = path_progress(
                        sr, sc, tr, tc, dist, fallback_target=tgt
                    )
                    if prog > 0.0:
                        return kill
        gather = best_land_gather_action(obs, memory, **kw)
        if gather is not None and share < GATHER_SHARE_MIN:
            return gather
        if kill is not None:
            if _prefer_gather_over(kill):
                if gather is not None:
                    return gather
            else:
                return kill
            return kill
        attack = best_winning_attack(obs, memory, **kw)
        if attack is not None and not _prefer_gather_over(attack):
            return attack
        if gather is not None and share < 0.55:
            return gather
        if attack is not None:
            return attack
        scout = best_cheap_scout_action(obs, memory, **kw)
        if scout is not None:
            return scout
        if castle_timing_weight(turn) >= 1.3:
            castle = best_castle_action(obs, memory, **kw)
            if castle is not None:
                return castle

    # Pre-contact / post-contact fog: explore, but take early castles when timed.
    tw = castle_timing_weight(turn)
    urg = fog_urgency(turn, enemy_seen=seen)
    explore = best_fog_explore_action(obs, memory, **kw)
    castle = best_castle_action(obs, memory, **kw) if tw >= 1.0 else None
    my_land = int(getattr(obs, "my_land", 0))
    if (
        castle is not None
        and my_land >= 4
        and not gen_visible
        and tw * 2.0 >= urg
    ):
        return castle

    # Pre-contact: do not leave a huge pile idle on general/castle.
    if not seen:
        idle = structure_idle_army(obs, memory)
        evacuate = (
            best_structure_evacuate_action(obs, memory, **kw)
            if idle >= STRUCTURE_IDLE_ARMY
            else None
        )
        if evacuate is not None:
            own_struct = own_structure_mask(obs, memory)
            _t, _o, armies = _as_grids(obs)
            explore_leaves_struct = False
            if explore is not None:
                ends = move_dest(explore)
                if ends is not None:
                    sr, sc, _tr, _tc = ends
                    explore_leaves_struct = bool(own_struct[sr, sc]) and int(
                        armies[sr, sc]
                    ) >= STRUCTURE_IDLE_ARMY
            if explore_leaves_struct and explore is not None:
                return explore
            return evacuate

    # Fog expand first (king expand always; tip expand yields to gather if dispersed).
    if explore is not None:
        ends = move_dest(explore)
        tip_explore = (
            share < 0.25
            and ends is not None
            and king is not None
            and (ends[0], ends[1]) != king
        )
        if tip_explore:
            gather = best_land_gather_action(obs, memory, **kw)
            if gather is not None:
                return gather
        return explore
    if share < 0.25:
        gather = best_land_gather_action(obs, memory, **kw)
        if gather is not None:
            return gather
    king_march = best_king_commit_action(obs, memory, **kw)
    if king_march is not None:
        return king_march
    approach = best_fog_approach_action(obs, memory, **kw)
    if approach is not None:
        return approach

    if share < GATHER_SHARE_MIN:
        gather = best_land_gather_action(obs, memory, **kw)
        if gather is not None:
            return gather

    gather = best_land_gather_action(obs, memory, **kw)
    if gather is not None and share < 0.55:
        return gather
    if castle is not None and tw >= 1.0:
        return castle
    expand = smoke_like_expand_action(obs, memory, **kw)
    if expand is not None:
        return expand
    move = first_playable_move(obs, memory, **kw)
    if move is not None:
        return move
    return (1, 0, 0, 0, 0)


def best_prior_legal_action(
    prior: Array,
    obs,
    memory: VisibleMemory,
    *,
    mask: Optional[Array] = None,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
    require_progress: bool = False,
) -> Optional[Action5]:
    """Argmax of a shaped prior among legal non-pass moves.

    When ``require_progress`` is set, skip own-land retreats so a passive NN
    top is redirected without calling a full heuristic policy.
    """
    if mask is None:
        mask = play_mask(obs, memory)
    mask_a = np.asarray(mask, dtype=bool).reshape(-1)
    prior_a = np.asarray(prior, dtype=np.float64).reshape(-1)
    if prior_a.shape != mask_a.shape:
        return None
    goals = seek_goals(obs, memory)
    dist_field = path_distance_field(obs, goals) if goals else None
    target = enemy_seek_target(obs, memory)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    H, W = int(obs.H), int(obs.W)
    best_idx = -1
    best_score = -1.0
    for idx in np.flatnonzero(mask_a):
        idx = int(idx)
        if idx == PASS_INDEX:
            continue
        action = decode_action(idx)
        move: Action5 = tuple(int(x) for x in action)  # type: ignore[assignment]
        if blocks_oscillation(move, prev_action, obs, recent_actions=recent_actions):
            continue
        if require_progress:
            ends = move_dest(move)
            if ends is None:
                continue
            sr, sc, tr, tc = ends
            if not (0 <= tr < H and 0 <= tc < W):
                continue
            dest_o = int(owners[tr, tc])
            prog = path_progress(
                sr, sc, tr, tc, dist_field, fallback_target=target
            )
            if dest_o == 1 and prog <= 0.0:
                continue
        score = float(prior_a[idx])
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx < 0:
        return None
    return tuple(int(x) for x in decode_action(best_idx))  # type: ignore[return-value]


def constrain_nn_action(
    obs,
    memory: VisibleMemory,
    action: Action5,
    *,
    prev_action: Optional[Action5] = None,
    recent_actions: Sequence[Action5] = (),
    prior: Optional[Array] = None,
) -> Action5:
    """Keep NN/search choice except for hard rules + prior re-rank.

    Hard rules: general capture; never pass when another move exists; never
    own-land oscillation. Soft redirects (pass / retreat / lateral home) pick
    the best legal action from the shaped root prior — not a full heuristic
    policy. When ``prior`` is missing, only hard rules apply.
    """
    mask = play_mask(obs, memory)
    caps = general_capture_indices(obs, memory, mask)
    if caps:
        _t, _o, armies = _as_grids(obs)
        best_cap = max(
            caps,
            key=lambda i: int(armies[decode_action(i)[1], decode_action(i)[2]]),
        )
        return tuple(int(x) for x in decode_action(best_cap))  # type: ignore[return-value]

    chosen: Action5 = tuple(int(x) for x in action)  # type: ignore[assignment]
    nonpass = np.asarray(mask, dtype=bool).copy()
    nonpass[PASS_INDEX] = False
    is_pass = int(chosen[0]) == 1 or chosen == (1, 0, 0, 0, 0)

    def _redirect(*, require_progress: bool) -> Optional[Action5]:
        if prior is None:
            return None
        return best_prior_legal_action(
            prior,
            obs,
            memory,
            mask=mask,
            prev_action=prev_action,
            recent_actions=recent_actions,
            require_progress=require_progress,
        )

    if is_pass and np.any(nonpass):
        alt = _redirect(require_progress=False)
        if alt is not None:
            return alt
        # Last resort: any legal non-pass (keeps the no-pass hard rule).
        idx = int(np.flatnonzero(nonpass)[0])
        return tuple(int(x) for x in decode_action(idx))  # type: ignore[return-value]

    if blocks_oscillation(chosen, prev_action, obs, recent_actions=recent_actions):
        alt = _redirect(require_progress=False)
        if alt is not None:
            return alt

    seek_target = (
        enemy_seek_target(obs, memory) if enemy_is_visible(obs, memory) else None
    )
    goals = seek_goals(obs, memory)
    dist_field = path_distance_field(obs, goals) if goals else None
    ends = move_dest(chosen)
    if ends is not None and prior is not None:
        sr, sc, tr, tc = ends
        H, W = int(obs.H), int(obs.W)
        if 0 <= tr < H and 0 <= tc < W:
            owners = np.asarray(obs.owner_grid, dtype=np.int32)
            dest_o = int(owners[tr, tc])
            progress = path_progress(
                sr, sc, tr, tc, dist_field, fallback_target=seek_target
            )
            # Passiveive own-land shuffle / retreat → re-rank shaped prior.
            if dest_o == 1 and (
                progress < 0.0
                or (
                    progress <= 0.0
                    and newly_revealed_cells(obs, tr, tc) == 0
                )
            ):
                alt = _redirect(require_progress=True)
                if alt is not None and alt != chosen:
                    return alt

    # Pre-contact: fat pile on gen while NN retreats — evacuate via prior if
    # an expand/leave-struct move ranks highest among progressive options.
    if prior is not None and not enemy_is_visible(obs, memory):
        idle = structure_idle_army(obs, memory)
        if idle >= STRUCTURE_IDLE_ARMY:
            leaves = False
            own_struct = own_structure_mask(obs, memory)
            if ends is not None:
                sr, sc, _tr, _tc = ends
                leaves = bool(own_struct[sr, sc]) and int(
                    np.asarray(obs.army_grid)[sr, sc]
                ) >= STRUCTURE_IDLE_ARMY
            if not leaves:
                alt = _redirect(require_progress=True)
                if alt is not None and alt != chosen:
                    return alt

    return chosen


def mandatory_action_indices(
    obs,
    memory: VisibleMemory,
    *,
    mask: Optional[Array] = None,
    cost_grid: Optional[Array] = None,
) -> list[int]:
    """Captures + attacks + frontier expands (pass only if alone)."""
    if mask is None:
        mask = play_mask(obs, memory, cost_grid=cost_grid)
    out: list[int] = []
    seen: set[int] = set()
    extras: list[int] = (
        general_capture_indices(obs, memory, mask)
        + visible_enemy_source_interaction_indices(obs, memory, mask)
        + enemy_attack_indices(obs, memory, mask)
        + frontier_expand_indices(obs, memory, mask)
    )
    nonpass = np.asarray(mask, dtype=bool).copy()
    nonpass[PASS_INDEX] = False
    prefix = [PASS_INDEX] if (bool(mask[PASS_INDEX]) and not np.any(nonpass)) else []
    for idx in prefix + extras:
        if idx in seen:
            continue
        if not mask[idx]:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def policy_ordered_candidates(
    prior: Array,
    mask: Array,
    *,
    mandatory: Sequence[int],
    limit: int,
) -> list[int]:
    """Mandatory actions first (stable), then remaining by descending prior."""
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    out: list[int] = []
    seen: set[int] = set()
    for idx in mandatory:
        idx = int(idx)
        if idx in seen or not mask[idx]:
            continue
        seen.add(idx)
        out.append(idx)
        if len(out) >= limit:
            return out
    legal_idx = np.flatnonzero(mask)
    if legal_idx.size == 0:
        return out
    remaining = np.asarray(
        [int(i) for i in legal_idx if int(i) not in seen], dtype=np.int64
    )
    if remaining.size == 0:
        return out
    order = np.argsort(-prior[remaining], kind="stable")
    for idx in remaining[order]:
        idx = int(idx)
        out.append(idx)
        if len(out) >= limit:
            break
    return out


def load_tactical_suite(path: Path | str) -> list[dict[str, Any]]:
    """Load ``tactical-suite.json`` cases."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases", data)
    if not isinstance(cases, list):
        raise ValueError("tactical suite must contain a list of cases")
    return cases


def suite_path() -> Path:
    return Path(__file__).resolve().parent / "tests" / "fixtures" / "tactical-suite.json"
