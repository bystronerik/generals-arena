"""Tactical helpers for Morpheus search: play mask, prior shaping, hard rules.

Pass is omitted whenever any other playable action exists. Until an enemy is
visible, moves onto the own general or castle are banned, and idle piles on
those structures are pushed out toward fog. Direction toward the seek target
is rewarded on own land; retreats are punished; enemy takes rank highest.
Land lead is not a hunt gate.

The standalone heuristic play cascade (``select_play_action`` and its
sub-policies) lives in ``scripts/morpheus_heuristic_play.py`` — it is probe
tooling, not part of the bot's play path.
"""
from __future__ import annotations

import json
import math
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

# Fog-hunt urgency keeps rising until first enemy sight.
FOG_URGENCY_TURN_SCALE = 40.0
# Tip armies below this are slow explorers — prefer a formed wave.
EXPLORE_WAVE_MIN = 3
# Remember this many recent army moves; reverse along any of those edges is banned.
OSCILLATION_HISTORY = 8

# Prior-shaping blend (Part 17). ``lam`` is the trust knob, ``log_clip`` bounds
# how far one heuristic may move an action, ``floor_frac`` keeps a network zero
# from being resurrected. The LEGACY_* values reproduce the unbounded geometric
# reshape that shipped before Part 17 and exist for the parity test.
LEGACY_SHAPING_LAMBDA = 1.0
LEGACY_SHAPING_LOG_CLIP = math.inf
LEGACY_SHAPING_FLOOR_FRAC = 0.0
LEGACY_SHAPING_FLOOR_ABS = 1e-6

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


def king_cell(obs) -> Optional[tuple[int, int]]:
    """Cell holding the largest own army (ties: first in row-major order)."""
    _types, owners, armies = _as_grids(obs)
    own = owners == 1
    if not np.any(own):
        return None
    max_a = int(armies[own].max())
    locs = np.argwhere(own & (armies == max_a))
    return int(locs[0, 0]), int(locs[0, 1])


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


def heuristic_action_scores(
    obs,
    memory: VisibleMemory,
    mask: Optional[Array] = None,
    belief: Optional[Any] = None,
) -> Array:
    """Per-action tactical score over the play mask, independent of the network.

    Expand before contact, seek after it. The scores are a *relative* ranking
    signal only: ``blend_prior`` decides how much of it reaches the root prior,
    so the absolute magnitudes here carry no meaning beyond their ratios.

    General captures are deliberately not scored. They are already guaranteed by
    ``mandatory_action_indices`` (root candidate inclusion) and forced by
    ``constrain_nn_action`` (hard rule), so a score term would only duplicate a
    rule that cannot be outvoted anyway.
    """
    if mask is None:
        mask = play_mask(obs, memory)
    mask_a = np.asarray(mask, dtype=bool).reshape(-1)

    _types, owners, armies = _as_grids(obs)
    H, W = int(obs.H), int(obs.W)
    out = np.zeros(mask_a.shape, dtype=np.float64)
    seen_enemy = enemy_is_visible(obs, memory)

    turn = int(getattr(obs, "turn", 0))
    castle_w = castle_timing_weight(turn)
    for idx in np.flatnonzero(mask_a):
        idx = int(idx)
        action = decode_action(idx)
        if int(action[0]) != 2:
            continue
        r, c = int(action[1]), int(action[2])
        army = float(max(int(armies[r, c]), 1))
        # Prefer cells that can still hold a useful remnant after the spend.
        out[idx] = max(
            out[idx],
            castle_w * (10.0 + 0.2 * min(army, 100.0)),
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
            thrash = tip_thrash_factor(army_i, max_own, tot)
            gather = stack_gather_factor(
                army_i, dest_army, dest_owner, progress=progress
            )

            if dest_owner == OWNER_ENEMY and surplus > 0.0:
                # Strong attack reward: raw NN rarely proposes takes (probe ~0.8%).
                score = (
                    atk_w
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
                    atk_w
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
                        army_w
                        * (14.0 + 22.0 * k_prog)
                        * stack_gather_factor(
                            army_i, dest_army, 1, progress=k_prog
                        )
                    )
                elif progress > 0.0:
                    score = (
                        atk_w
                        * thrash
                        * bias
                        * (18.0 + 32.0 * progress)
                        * gather
                    )
                else:
                    # Raw NN often picks retreat; keep mass near zero here.
                    score = 0.008 * army_w * bias * gather
            else:
                score = 0.01 * army_w * gather
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
            progress = path_progress(
                sr, sc, tr, tc, fog_dist, fallback_target=fog_target
            )
            bias = direction_bias(progress, OWNER_NEUTRAL)
            score = (
                ew
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
                    out[idx] = max(out[idx], 0.01)
                    continue
                out[idx] = max(
                    out[idx],
                    urgency
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
                    ew
                    * urgency
                    * tip_bonus
                    * bias
                    * (0.8 + 4.0 * progress)
                    * gather,
                )
            else:
                out[idx] = max(out[idx], 0.008 * bias * gather)

    return np.where(mask_a, np.maximum(out, 0.0), 0.0)


def blend_prior(
    nn_prior: Array,
    scores: Array,
    mask: Array,
    *,
    lam: float,
    log_clip: float,
    floor_frac: float,
    floor_abs: float = 0.0,
) -> Array:
    """Blend a network prior with heuristic scores under a bounded nudge.

    ``shaped = softmax( log p + lam * clip(log(h / gmean(h)), +-log_clip) )``

    - ``lam`` is the single trust knob. ``lam = 0`` returns the network prior
      renormalized over ``mask``; ``lam = 1`` with ``log_clip = inf`` is the
      plain geometric blend ``p * h``.
    - ``log_clip`` caps how far one heuristic may move an action, in log space.
      ``ln 10`` means "at most a 10x nudge either way", so no heuristic can
      override the network by orders of magnitude.
    - Centering by the geometric mean is load-bearing: raw scores sit around
      ~120, so without centering every action would saturate the clip and the
      heuristic term would collapse to a constant.
    - ``floor_frac`` lifts the prior to a fraction of its own maximum before the
      blend, so an action the network zeroed cannot be resurrected past the clip
      bound. ``floor_abs`` is the legacy absolute floor, kept for parity.

    Actions with zero heuristic score sit at the bottom of the clip range rather
    than at zero, unless ``log_clip`` is infinite (then they keep the legacy
    hard zero).
    """
    mask_a = np.asarray(mask, dtype=bool).reshape(-1)
    prior_a = np.asarray(nn_prior, dtype=np.float64).reshape(-1)
    prior_a = np.where(mask_a, np.maximum(prior_a, 0.0), 0.0)
    h = np.asarray(scores, dtype=np.float64).reshape(-1)
    h = np.where(mask_a, np.maximum(h, 0.0), 0.0)

    def _renormalized_prior() -> Array:
        total = float(prior_a.sum())
        if total > 0.0:
            return prior_a / total
        n = int(mask_a.sum())
        if n <= 0:
            return prior_a
        return mask_a.astype(np.float64) / float(n)

    lam = float(lam)
    if lam == 0.0 or float(log_clip) == 0.0:
        return _renormalized_prior()

    scored = mask_a & (h > 0.0)
    if not np.any(scored):
        return _renormalized_prior()

    floor = max(float(floor_abs), float(floor_frac) * float(prior_a.max()))
    p = np.where(mask_a, np.maximum(prior_a, floor), 0.0)
    if not np.any(p > 0.0):
        return _renormalized_prior()

    # Center before clipping so the clip measures a ratio, not a magnitude.
    log_gmean = float(np.mean(np.log(h[scored])))
    log_h = np.full(mask_a.shape, -np.inf, dtype=np.float64)
    log_h[scored] = np.log(h[scored]) - log_gmean
    clip = float(log_clip)
    log_h = np.clip(log_h, -clip, clip)

    logits = np.full(mask_a.shape, -np.inf, dtype=np.float64)
    live = mask_a & (p > 0.0)
    logits[live] = np.log(p[live]) + lam * log_h[live]

    finite = np.isfinite(logits)
    if not np.any(finite):
        return _renormalized_prior()
    top = float(logits[finite].max())
    weights = np.zeros(mask_a.shape, dtype=np.float64)
    weights[finite] = np.exp(logits[finite] - top)
    total = float(weights.sum())
    if total <= 0.0:
        return _renormalized_prior()
    return weights / total


def apply_pre_contact_prior(
    prior: Array,
    obs,
    memory: VisibleMemory,
    *,
    mask: Optional[Array] = None,
    belief: Optional[Any] = None,
    lam: float = LEGACY_SHAPING_LAMBDA,
    log_clip: float = LEGACY_SHAPING_LOG_CLIP,
    floor_frac: float = LEGACY_SHAPING_FLOOR_FRAC,
    floor_abs: float = LEGACY_SHAPING_FLOOR_ABS,
) -> Array:
    """Reshape root prior: expand pre-contact, seek after contact."""
    if mask is None:
        mask = play_mask(obs, memory)
    mask_a = np.asarray(mask, dtype=bool).reshape(-1)
    scores = heuristic_action_scores(obs, memory, mask_a, belief)
    return blend_prior(
        prior,
        scores,
        mask_a,
        lam=lam,
        log_clip=log_clip,
        floor_frac=floor_frac,
        floor_abs=floor_abs,
    )


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


def enemy_general_visible(obs, memory: VisibleMemory) -> bool:
    """True when the enemy general is in view or already latched in memory."""
    types, owners, _ = _as_grids(obs)
    if np.any(((types == TYPE_GENERAL) & (owners == OWNER_ENEMY))):
        return True
    return bool(np.any(memory.known_enemy_general))


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
