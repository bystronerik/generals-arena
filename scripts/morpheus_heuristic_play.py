"""Standalone Morpheus heuristic play policy for measurement scripts.

Moved out of ``bots/morpheus/tactics.py``: the deployed bot never calls this
cascade (its play path is ``constrain_nn_action`` + the shaped root prior).
Only the no-NN probe scripts use it (``morpheus_vs_smoke_grid.py``,
``morpheus_seed1_probe.py``, ``morpheus_vs_opp_probe.py``) as a fast
network-free baseline. Not part of the bot closure or content hash.

Callers must have ``bots/`` and ``bots/morpheus/`` on ``sys.path`` before
importing this module (the probe scripts already do).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from action import PASS_INDEX, decode_action, encode_action
from memory import (
    OWNER_ENEMY,
    OWNER_NEUTRAL,
    TYPE_FOG,
    TYPE_GENERAL,
    VisibleMemory,
)
from tactics import (
    COMMIT_ARMY_FRAC,
    COMMIT_DEST_ARMY_MAX,
    EXPLORE_WAVE_MIN,
    GATHER_SHARE_MIN,
    STACK_GATHER_BAN,
    STRUCTURE_IDLE_ARMY,
    _as_grids,
    _is_passable_type,
    _is_visible_neutral,
    army_concentration,
    attack_weight,
    blocks_oscillation,
    castle_timing_weight,
    direction_bias,
    enemy_attack_indices,
    enemy_general_visible,
    enemy_is_visible,
    enemy_seek_target,
    explore_cost,
    explore_wave_weight,
    fog_urgency,
    frontier_expand_indices,
    general_capture_indices,
    is_committed_army,
    king_cell,
    move_dest,
    newly_revealed_cells,
    own_structure_mask,
    path_distance_field,
    path_progress,
    play_mask,
    seek_goals,
    stack_gather_factor,
    structure_idle_army,
    tip_thrash_factor,
    wave_weight,
)
from transition import DIRECTIONS

Array = np.ndarray
Action5 = tuple[int, int, int, int, int]

# Path distance (BFS) at which an enemy threatens the general enough to
# yank the king home. Manhattan-8 was too soft and stalled every push.
DEFEND_PATH_NEAR = 4
DEFEND_PATH_THIN = 6
# Garrison that still needs reinforcement when an enemy is within THIN range.
DEFEND_GEN_THIN = 20
# Minimum army left on the general when evacuating a fat pile is impossible
# in one step (moves leave 1); instead delay evacuate until above this.
GENERAL_EVACUATE_MIN = 18


def largest_own_army(obs) -> int:
    _types, owners, armies = _as_grids(obs)
    own = armies[owners == 1]
    if own.size == 0:
        return 0
    return int(own.max())


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


def explore_efficiency(obs, dest_r: int, dest_c: int, dest_army: int) -> float:
    """New vision per army spent — cheap fog scouting ranks high."""
    return float(newly_revealed_cells(obs, dest_r, dest_c)) / float(
        explore_cost(dest_army)
    )


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

    No-NN baseline for measurement scripts. The full bot uses
    ``tactics.constrain_nn_action`` so the network still chooses most moves.
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
