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
from transition import DEATHTOUCH_TURN, DIRECTIONS

Array = np.ndarray
Action5 = tuple[int, int, int, int, int]

# Fog-hunt urgency keeps rising until first enemy sight.
FOG_URGENCY_TURN_SCALE = 40.0
# Tip armies below this are slow explorers — prefer a formed wave.
EXPLORE_WAVE_MIN = 3
# Remember this many recent army moves; reverse along any of those edges is banned.
OSCILLATION_HISTORY = 8
# Commitment hysteresis: shaped-prior bonus for moving the stack we moved last
# turn (source == previous destination). Measured without it, the chosen source
# tile jumped >=3 Manhattan on 27% of consecutive move turns — plans died to
# tie-break jitter. The bonus only amplifies already-positive scores, so a
# banned or retreating continuation stays dead.
CONTINUATION_BONUS = 1.5
# Once the enemy general is latched, enemy takes that do not shorten the path
# to it are farming, not killing. Measured: a 1200-turn draw in which the
# border was chewed for 850 turns while the general sat at 1-9 army, unseen.
GENERAL_CHEW_DAMP = 0.3
# Multiplier per step of progress toward the known/believed general, applied
# on top of the normal take/carve/march scores. A bonus, not a goal swap: an
# earlier version replaced the enemy-land goal set with the believed cell and
# promptly lost its own general — incursions near home stopped counting as
# progress, so defense collapsed along with broad border pressure.
HUNT_PROGRESS_BONUS = 0.75
# Post-DEATHTOUCH_TURN, any executed move onto the general wins outright, so a
# touch outranks every other action. The blend clip saturates this to the top
# of the shaped prior; the adjacency hard rule makes the touch itself forced.
DEATHTOUCH_SCORE = 1.0e6

# Prior-shaping blend (Part 17). ``lam`` is the trust knob, ``log_clip`` bounds
# how far one heuristic may move an action, ``floor_frac`` keeps a network zero
# from being resurrected. Semantics: docs/bots/morpheus/prior-shaping.md.
#
# lambda stays at 1 until the Part 17 C2 sweep picks a value; the bound comes
# from the clip, not from distrusting the heuristic ranking outright.
DEFAULT_SHAPING_LAMBDA = 1.0
DEFAULT_SHAPING_LOG_CLIP = math.log(10.0)  # at most a 10x nudge either way
DEFAULT_SHAPING_FLOOR_FRAC = 1e-3

# The LEGACY_* values reproduce the unbounded geometric reshape that shipped
# before Part 17. They exist for the A1 parity test, not for play.
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


# Soft size bonus for tip-feed / explore (not for commit attacks). Raised from
# 20: measured waves hit enemy territory with a median 6 army (0.9% of total),
# so the size signal saturated far below a useful attack mass.
WAVE_ARMY_SOFT_CAP = 60
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
# Raised from 0.35: gathering stopped while the king held barely a third of
# the army, which is how 17-of-219 first-contact waves happened.
GATHER_SHARE_MIN = 0.5
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


def believed_enemy_general(belief: Optional[Any]) -> Optional[tuple[int, int]]:
    """Mode of the particle posterior over the enemy general's cell.

    The particle filter maintains exactly the signal the general hunt needs;
    before this, ``enemy_seek_target`` deleted its ``belief`` argument and the
    seek goal degenerated to "nearest border cell" — measured as a 1200-turn
    game in which the enemy general was never even seen. Duck-typed so tests
    can pass a light stub. Returns ``None`` when no usable posterior exists.
    """
    if belief is None or int(getattr(belief, "n", 0)) <= 0:
        return None
    enemy = int(belief.enemy_seat)
    mass: dict[tuple[int, int], tuple[float, int]] = {}
    for p in belief.particles:
        g = p.state.general_positions[enemy]
        gr, gc = int(g[0]), int(g[1])
        if gr < 0 or gc < 0:
            continue
        w, n = mass.get((gr, gc), (0.0, 0))
        mass[(gr, gc)] = (w + max(float(p.weight), 0.0), n + 1)
    if not mass:
        return None
    # Weighted mode; particle count breaks ties when all weights are zero.
    cell, _ = max(mass.items(), key=lambda kv: kv[1])
    return cell


def seek_goals(
    obs, memory: VisibleMemory, belief: Optional[Any] = None
) -> list[tuple[int, int]]:
    """Cells to drive toward: enemy general, else land, else believed general.

    Enemy land stays ahead of the belief posterior on purpose: it is what
    keeps incursions near home scored as progress (defense) and the border
    under broad pressure. The directed hunt is a separate *bonus* field in
    ``heuristic_action_scores``, not a goal replacement. The believed cell
    takes over only when no enemy land is visible — lost contact, or the
    pre-contact fog hunt (better beacon than the opposite corner).
    """
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
    believed = believed_enemy_general(belief)
    if believed is not None:
        return [believed]
    # Pre-contact fallback: opposite corner from own general (fog beacon).
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
    """Own-land friction: near-free forward merges, hard on retreat dumps.

    Forward merges into a big stack ARE the wave-building move, so the forward
    penalty only bites past ~80 army (was: visibly punished from ~25, which
    kept attack waves at tip size). Lateral merges stay dampened but no longer
    crushed — the retreat branch is the one that stays brutal.
    """
    if int(dest_owner) != 1:
        return 1.0
    src = max(int(src_army), 1)
    dest = max(int(dest_army), 0)
    p = float(progress)
    if p > 0.0:
        # Toward enemy: corridor gathers / front consolidation are near-free.
        return float(1.0 / (1.0 + (dest / 80.0) ** 2))
    if p < 0.0:
        # Away from enemy: crush pile dumps and retreat shuffles.
        return float(0.03 / (1.0 + dest / 4.0))
    # Lateral: dampen, don't forbid — merging a tip into a nearby pile can
    # still be the right wave-forming move.
    feed = 1.0 / (1.0 + (dest / 18.0) ** 2)
    if src >= STACK_GATHER_BAN and dest > COMMIT_DEST_ARMY_MAX:
        return float(feed * 0.25)
    return float(feed)


def _as_grids(obs) -> tuple[Array, Array, Array]:
    return (
        np.asarray(obs.type_grid, dtype=np.int32),
        np.asarray(obs.owner_grid, dtype=np.int32),
        np.asarray(obs.army_grid, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Vectorized scoring machinery. ``heuristic_action_scores`` ran a Python loop
# with two full visibility dilations per candidate action (~15 ms per call
# mid-game); these tables and array helpers replace that with one dilation and
# pure numpy. The scalar helpers above stay the semantic reference — the
# parity test in tests/test_heuristic_scores_parity.py holds the two together.
# ---------------------------------------------------------------------------

_DECODE_TABLES: Optional[tuple[Array, Array, Array, Array, Array]] = None


def _decode_tables() -> tuple[Array, Array, Array, Array, Array]:
    """Static ``(kind, sr, sc, tr, tc)`` per non-pass action index.

    ``tr``/``tc`` are move destinations (unclipped; may fall outside a small
    board — callers bounds-check against the live ``H``/``W``). For builds the
    destination equals the source.
    """
    global _DECODE_TABLES
    if _DECODE_TABLES is None:
        n = PASS_INDEX
        kind = np.zeros(n, dtype=np.int16)
        sr = np.zeros(n, dtype=np.int16)
        sc = np.zeros(n, dtype=np.int16)
        tr = np.zeros(n, dtype=np.int16)
        tc = np.zeros(n, dtype=np.int16)
        for i in range(n):
            k, r, c, d, _s = decode_action(i)
            kind[i] = k
            sr[i] = r
            sc[i] = c
            if k == 0:
                tr[i] = r + int(DIRECTIONS[d, 0])
                tc[i] = c + int(DIRECTIONS[d, 1])
            else:
                tr[i] = r
                tc[i] = c
        _DECODE_TABLES = (kind, sr, sc, tr, tc)
    return _DECODE_TABLES


def reveal_count_grid(obs) -> Array:
    """Per-cell ``newly_revealed_cells`` computed once for the whole board.

    Visibility is a 3×3 dilation of ownership, so owning one new cell reveals
    exactly the currently-invisible cells inside that cell's 3×3 box. Cells we
    already own reveal nothing (matches ``newly_revealed_cells``).
    """
    _types, owners, _armies = _as_grids(obs)
    owned = owners == 1
    before = visibility_mask(owned)
    invisible = (~before).astype(np.int32)
    H, W = invisible.shape
    padded = np.pad(invisible, 1)
    box = np.zeros((H, W), dtype=np.int32)
    for dr in range(3):
        for dc in range(3):
            box += padded[dr : dr + H, dc : dc + W]
    return np.where(owned, 0, box)


def _wave_weight_v(army: Array) -> Array:
    a = np.clip(army, 1, WAVE_ARMY_SOFT_CAP)
    return 1.0 + 0.25 * np.log1p(a)


def _attack_weight_v(army: Array) -> Array:
    a = np.clip(army, 1, ATTACK_ARMY_CAP)
    return 1.0 + 0.85 * np.log1p(a)


def _explore_wave_weight_v(army: Array) -> Array:
    a = np.maximum(army, 1)
    formed = 1.0 + 0.9 * np.log1p(np.minimum(a, 80))
    return np.where(a < EXPLORE_WAVE_MIN, 0.12, formed)


def _committed_v(army: Array, max_army: int, total: int, share: float) -> Array:
    a = np.asarray(army)
    tot = max(int(total), 1)
    mx = max(int(max_army), 1)
    if share < GATHER_SHARE_MIN:
        floor = max(STACK_GATHER_BAN, int(COMMIT_TOTAL_FRAC * tot))
        return (a >= mx) & (a >= floor)
    return (a >= COMMIT_ARMY_FRAC * mx) | (a >= COMMIT_TOTAL_FRAC * tot)


def _tip_thrash_v(src_army: Array, max_army: int, total: int) -> Array:
    src = np.maximum(src_army, 1)
    big = max(int(max_army), 1)
    tot = max(int(total), 1)
    share = float(big) / float(tot)
    committed = _committed_v(src, big, tot, share)
    out = 0.08 + 0.6 * (src / float(big))
    if big < STACK_GATHER_BAN and share >= GATHER_SHARE_MIN:
        return np.ones_like(out)
    return np.where(committed, 1.0, out)


def _gather_v(src_army: Array, dest_army: Array, progress: Array) -> Array:
    """Vector ``stack_gather_factor`` for own-land destinations."""
    src = np.maximum(src_army, 1)
    dest = np.maximum(dest_army, 0).astype(np.float64)
    forward = 1.0 / (1.0 + (dest / 80.0) ** 2)
    retreat = 0.03 / (1.0 + dest / 4.0)
    feed = 1.0 / (1.0 + (dest / 18.0) ** 2)
    lateral = np.where(
        (src >= STACK_GATHER_BAN) & (dest > COMMIT_DEST_ARMY_MAX),
        feed * 0.25,
        feed,
    )
    return np.where(progress > 0.0, forward, np.where(progress < 0.0, retreat, lateral))


def _bias_v(progress: Array, dest_owner: Array) -> Array:
    """Vector ``direction_bias``."""
    p = np.asarray(progress, dtype=np.float64)
    own = np.asarray(dest_owner)
    enemy = 4.0 + 2.0 * np.maximum(p, 0.0)
    toward = np.where(own == 1, 1.5 + 1.2 * p, 1.8 + 1.4 * p)
    flat = np.where(own == 1, 0.25, 0.55)
    non_enemy = np.where(p > 0.0, toward, np.where(p < 0.0, 0.06, flat))
    return np.where(own == OWNER_ENEMY, enemy, non_enemy)


def _path_progress_v(
    sr: Array,
    sc: Array,
    tr: Array,
    tc: Array,
    dist_field: Optional[Array],
    fallback_target: Optional[tuple[int, int]],
) -> Array:
    """Vector ``path_progress`` (all coordinates already in bounds)."""
    if fallback_target is None:
        manhattan = np.zeros(len(sr), dtype=np.float64)
    else:
        gr, gc = fallback_target
        before_m = np.abs(sr - gr) + np.abs(sc - gc)
        after_m = np.abs(tr - gr) + np.abs(tc - gc)
        manhattan = (before_m - after_m).astype(np.float64)
    if dist_field is None:
        return manhattan
    before = dist_field[sr, sc].astype(np.float64)
    after = dist_field[tr, tc].astype(np.float64)
    both_unreachable = (before < 0) & (after < 0)
    enter = (before < 0) & (after >= 0)
    leave = (before >= 0) & (after < 0)
    out = before - after
    out = np.where(enter, 2.0, out)
    out = np.where(leave, -2.0, out)
    return np.where(both_unreachable, manhattan, out)


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
    """Drive toward known/believed enemy general, else nearest enemy land.

    Prefers the enemy cell closest (Manhattan) to the largest own stack so the
    king stack does not aim at a centroid behind mountains.
    """
    goals = seek_goals(obs, memory, belief)
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
    *,
    prev_action: Optional[Action5] = None,
) -> Array:
    """Per-action tactical score over the play mask, independent of the network.

    Expand before contact, seek after it. The scores are a *relative* ranking
    signal only: ``blend_prior`` decides how much of it reaches the root prior,
    so the absolute magnitudes here carry no meaning beyond their ratios.

    General captures are deliberately not scored. They are already guaranteed by
    ``mandatory_action_indices`` (root candidate inclusion) and forced by
    ``constrain_nn_action`` (hard rule), so a score term would only duplicate a
    rule that cannot be outvoted anyway.

    ``prev_action`` enables commitment hysteresis: moves continuing the stack
    that moved last turn get ``CONTINUATION_BONUS`` if they already score > 0.
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
    kind_t, sr_t, sc_t, tr_t, tc_t = _decode_tables()
    legal = np.flatnonzero(mask_a[:PASS_INDEX])
    build_idx = legal[kind_t[legal] == 2]
    if build_idx.size:
        b_army = np.maximum(
            armies[sr_t[build_idx], sc_t[build_idx]], 1
        ).astype(np.float64)
        # Prefer cells that can still hold a useful remnant after the spend.
        out[build_idx] = castle_w * (10.0 + 0.2 * np.minimum(b_army, 100.0))

    move_all = legal[kind_t[legal] == 0]
    sr = sr_t[move_all].astype(np.int64)
    sc = sc_t[move_all].astype(np.int64)
    tr = tr_t[move_all].astype(np.int64)
    tc = tc_t[move_all].astype(np.int64)
    inb = (tr >= 0) & (tr < H) & (tc >= 0) & (tc < W)
    move_idx = move_all[inb]
    sr, sc, tr, tc = sr[inb], sc[inb], tr[inb], tc[inb]

    if seen_enemy:
        target = enemy_seek_target(obs, memory, belief=belief)
        goals = seek_goals(obs, memory, belief)
        dist_field = path_distance_field(obs, goals) if goals else None
        king = king_cell(obs)
        king_dist = path_distance_field(obs, [king]) if king is not None else None
        share, max_own, tot = army_concentration(obs)
        gen_known = enemy_general_visible(obs, memory)
        if move_idx.size:
            army_i = armies[sr, sc].astype(np.int64)
            dest_owner = owners[tr, tc]
            dest_army = armies[tr, tc].astype(np.int64)
            reveal = reveal_count_grid(obs)[tr, tc].astype(np.float64)
            efficiency = reveal / (1.0 + np.maximum(dest_army, 0))
            progress = _path_progress_v(sr, sc, tr, tc, dist_field, target)
            bias = _bias_v(progress, dest_owner)
            surplus = np.maximum(army_i - dest_army - 1, 0).astype(np.float64)
            thrash = _tip_thrash_v(army_i, max_own, tot)
            army_w = _wave_weight_v(army_i)
            atk_w = _attack_weight_v(army_i)
            gather = np.where(
                dest_owner == 1, _gather_v(army_i, dest_army, progress), 1.0
            )

            # Strong attack reward: raw NN rarely proposes takes (probe ~0.8%).
            enemy_score = (
                atk_w
                * thrash
                * bias
                * (
                    120.0
                    + 40.0 * np.maximum(progress, 0.0)
                    + 8.0 * reveal
                    + 3.5 * np.minimum(surplus, 200.0)
                )
            )
            gen_cell = known_enemy_general_cell(obs, memory)
            hunt_cell = gen_cell or believed_enemy_general(belief)
            if gen_cell is not None:
                is_gen_touch = (tr == gen_cell[0]) & (tc == gen_cell[1])
            else:
                is_gen_touch = np.zeros(len(move_idx), dtype=bool)
            if hunt_cell is not None:
                hunt_dist = path_distance_field(obs, [hunt_cell])
                hunt_prog = _path_progress_v(sr, sc, tr, tc, hunt_dist, hunt_cell)
            else:
                hunt_prog = np.zeros(len(move_idx), dtype=np.float64)
            hunt_factor = 1.0 + HUNT_PROGRESS_BONUS * np.maximum(hunt_prog, 0.0)
            if gen_known:
                # Kill, don't farm: takes that shorten the path to the known
                # general keep the boost; sideways border chew is damped.
                enemy_score = enemy_score * np.where(
                    is_gen_touch | (progress > 0.0), 1.35, GENERAL_CHEW_DAMP
                )
            enemy_score = enemy_score * hunt_factor
            # Fog carve that shortens the path to enemy is a real attack prep.
            neutral_score = (
                atk_w
                * thrash
                * bias
                * (
                    45.0
                    + 30.0 * np.maximum(progress, 0.0)
                    + 12.0 * efficiency
                    + 8.0 * reveal
                )
            ) * hunt_factor
            # Own corridor: reward path-toward-enemy; crush retreat.
            # Also reward hinterland tips gathering into the king stack.
            committed = _committed_v(army_i, max_own, tot, share)
            if king_dist is not None:
                k_prog = np.where(
                    committed,
                    0.0,
                    _path_progress_v(sr, sc, tr, tc, king_dist, king),
                )
            else:
                k_prog = np.zeros(len(move_idx), dtype=np.float64)
            # Army-weighted: a 100-army hinterland stack's gather move must
            # outrank a 3-army tip shuffle, so weight by attack_weight (caps
            # at 200) rather than the wave cap.
            king_gather = (
                atk_w
                * (14.0 + 22.0 * k_prog)
                * _gather_v(army_i, dest_army, k_prog)
            )
            own_forward = (
                atk_w * thrash * bias * (18.0 + 32.0 * progress) * gather
            ) * hunt_factor
            # Raw NN often picks retreat; keep mass near zero here.
            own_idle = 0.008 * army_w * bias * gather
            own_score = np.where(
                (k_prog > 0.0) & (share < GATHER_SHARE_MIN),
                king_gather,
                np.where(progress > 0.0, own_forward, own_idle),
            )
            score = np.where(
                (dest_owner == OWNER_ENEMY) & (surplus > 0.0),
                enemy_score,
                np.where(
                    dest_owner == OWNER_NEUTRAL,
                    neutral_score,
                    np.where(dest_owner == 1, own_score, 0.01 * army_w * gather),
                ),
            )
            # Deathtouch: any executed touch wins outright, so the surplus
            # gate must not suppress it. Pre-800 an underpowered touch stays
            # in the near-zero branch (feeding a defended general is a loss).
            if turn >= DEATHTOUCH_TURN:
                score = np.where(is_gen_touch, DEATHTOUCH_SCORE, score)
            out[move_idx] = np.maximum(out[move_idx], score)
    elif move_idx.size:
        # No enemy yet: hunt fog with a formed wave; urgency rises with turn.
        # Idle piles on general/castle are heavily discouraged.
        urgency = fog_urgency(turn, enemy_seen=False)
        own_struct = np.asarray(own_structure_mask(obs, memory), dtype=bool)
        fog_target = enemy_seek_target(obs, memory, belief=belief)
        fog_goals = seek_goals(obs, memory, belief)
        fog_dist = path_distance_field(obs, fog_goals) if fog_goals else None

        src_army = armies[sr, sc].astype(np.int64)
        dest_owner = owners[tr, tc]
        dest_army = armies[tr, tc].astype(np.int64)
        progress = _path_progress_v(sr, sc, tr, tc, fog_dist, fog_target)
        struct_src = own_struct[sr, sc] & (src_army >= STRUCTURE_IDLE_ARMY)

        frontier = dest_owner == OWNER_NEUTRAL
        if np.any(frontier):
            reveal = reveal_count_grid(obs)[tr, tc].astype(np.float64)
            eff = reveal / (1.0 + np.maximum(dest_army, 0))
            ew = _explore_wave_weight_v(src_army)
            bias_f = _bias_v(progress, np.full(len(move_idx), OWNER_NEUTRAL))
            f_score = (
                ew
                * urgency
                * bias_f
                * (
                    4.0
                    + 12.0 * reveal
                    + 7.0 * eff
                    + 6.0 * np.maximum(progress, 0.0)
                )
            )
            # Leaving a fat structure into fog is the right pre-contact move.
            f_score = np.where(
                struct_src,
                f_score * (2.5 + 0.04 * np.minimum(src_army, 80)),
                f_score,
            )
            fi = move_idx[frontier]
            out[fi] = np.maximum(out[fi], f_score[frontier])

        # Own corridor: reward toward fog target; punish retreat / circles.
        corridor = dest_owner == 1
        if np.any(corridor):
            frontier_src_grid = np.zeros((H, W), dtype=bool)
            frontier_src_grid[sr[frontier], sc[frontier]] = True
            bias_c = _bias_v(progress, np.ones(len(move_idx), dtype=np.int64))
            gather = _gather_v(src_army, dest_army, progress)
            tip_bonus = np.where(frontier_src_grid[tr, tc], 2.0, 1.0)
            struct_score = np.where(
                progress < 0.0,
                0.01,
                urgency
                * tip_bonus
                * bias_c
                * (
                    5.0
                    + 0.1 * np.minimum(src_army, 100)
                    + 8.0 * np.maximum(progress, 0.0)
                )
                * gather,
            )
            ew_c = _explore_wave_weight_v(src_army)
            plain_score = np.where(
                progress > 0.0,
                ew_c * urgency * tip_bonus * bias_c * (0.8 + 4.0 * progress) * gather,
                0.008 * bias_c * gather,
            )
            c_score = np.where(struct_src, struct_score, plain_score)
            ci = move_idx[corridor]
            out[ci] = np.maximum(out[ci], c_score[corridor])

    if prev_action is not None:
        ends = move_dest(tuple(int(x) for x in prev_action))  # type: ignore[arg-type]
        if ends is not None:
            _pr, _pc, p_tr, p_tc = ends
            if 0 <= p_tr < H and 0 <= p_tc < W:
                cont = np.flatnonzero(
                    (kind_t == 0) & (sr_t == p_tr) & (sc_t == p_tc)
                )
                out[cont] = np.where(
                    out[cont] > 0.0, out[cont] * CONTINUATION_BONUS, out[cont]
                )

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
    lam: Optional[float] = None,
    log_clip: float = DEFAULT_SHAPING_LOG_CLIP,
    floor_frac: float = DEFAULT_SHAPING_FLOOR_FRAC,
    floor_abs: float = 0.0,
    lam_pre_contact: float = DEFAULT_SHAPING_LAMBDA,
    lam_post_contact: float = DEFAULT_SHAPING_LAMBDA,
    prev_action: Optional[Action5] = None,
) -> Array:
    """Reshape root prior: expand pre-contact, seek after contact.

    ``lam`` pins one trust level for both phases. Leave it ``None`` to select
    ``lam_pre_contact`` / ``lam_post_contact`` by ``enemy_is_visible``.
    """
    if mask is None:
        mask = play_mask(obs, memory)
    mask_a = np.asarray(mask, dtype=bool).reshape(-1)
    if lam is None:
        lam = (
            lam_post_contact
            if enemy_is_visible(obs, memory)
            else lam_pre_contact
        )
    scores = heuristic_action_scores(
        obs, memory, mask_a, belief, prev_action=prev_action
    )
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


def known_enemy_general_cell(
    obs, memory: VisibleMemory
) -> Optional[tuple[int, int]]:
    """Latched or currently visible enemy general cell, else ``None``."""
    types, owners, _ = _as_grids(obs)
    latched = np.argwhere(memory.known_enemy_general)
    if latched.size:
        return (int(latched[0, 0]), int(latched[0, 1]))
    gen = np.argwhere((types == TYPE_GENERAL) & (owners == OWNER_ENEMY))
    if gen.size:
        return (int(gen[0, 0]), int(gen[0, 1]))
    return None


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


def _capture_moved_army(idx: int, armies: Array) -> int:
    """Army that actually moves for a capture action (full or half split)."""
    action = decode_action(idx)
    src = int(armies[int(action[1]), int(action[2])])
    return src // 2 if int(action[4]) == 1 else src - 1


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

    Hard rules: winning general capture (any touch from ``DEATHTOUCH_TURN``);
    never pass when another move exists; never own-land oscillation. Soft
    redirects (pass / retreat / lateral home) pick the best legal action from
    the shaped root prior — not a full heuristic policy. When ``prior`` is
    missing, only hard rules apply.
    """
    mask = play_mask(obs, memory)
    caps = general_capture_indices(obs, memory, mask)
    if caps:
        _t, _o, armies = _as_grids(obs)
        turn = int(getattr(obs, "turn", 0))
        if turn < DEATHTOUCH_TURN:
            # Kill calculus: forcing an underpowered capture feeds the
            # defense. Only a touch that actually wins the clash is forced;
            # otherwise fall through and let scoring gather next door.
            def _defender(idx: int) -> int:
                a = decode_action(idx)
                d_r = int(a[1]) + int(DIRECTIONS[int(a[3]), 0])
                d_c = int(a[2]) + int(DIRECTIONS[int(a[3]), 1])
                return int(armies[d_r, d_c])

            caps = [
                i for i in caps if _capture_moved_army(i, armies) > _defender(i)
            ]
        if caps:
            best_cap = max(caps, key=lambda i: _capture_moved_army(i, armies))
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
