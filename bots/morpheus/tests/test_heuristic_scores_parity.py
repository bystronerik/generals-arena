"""Vectorized ``heuristic_action_scores`` matches the scalar reference.

The production path is pure numpy (one visibility dilation per call). This
file keeps the original per-action scalar implementation as the oracle; both
call the same scalar helpers (``wave_weight``, ``stack_gather_factor``, …), so
constant tuning propagates to both sides and only a structural divergence
fails the test.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import PASS_INDEX, decode_action
from memory import OWNER_ENEMY, OWNER_NEUTRAL
from tactics import (
    DEATHTOUCH_SCORE,
    ENEMY_TAKE_BASE,
    GATHER_SHARE_MIN,
    GENERAL_CHEW_DAMP,
    HUNT_PROGRESS_BONUS,
    NEUTRAL_CARVE_BASE,
    PRE_EFFICIENCY_WEIGHT,
    PRE_PROGRESS_WEIGHT,
    PRE_REVEAL_WEIGHT,
    STRUCTURE_IDLE_ARMY,
    army_concentration,
    attack_weight,
    castle_timing_weight,
    direction_bias,
    enemy_general_visible,
    enemy_is_visible,
    enemy_seek_target,
    explore_cost,
    explore_wave_weight,
    fog_urgency,
    frontier_expand_indices,
    heuristic_action_scores,
    is_committed_army,
    king_cell,
    known_enemy_general_cell,
    newly_revealed_cells,
    own_structure_mask,
    path_distance_field,
    path_progress,
    play_mask,
    seek_goals,
    stack_gather_factor,
    tip_thrash_factor,
    wave_weight,
)
from tactics import _as_grids  # noqa: F401  (shared grid casting)
from transition import DEATHTOUCH_TURN, DIRECTIONS

from shaping_boards import BOARDS


def reference_scores(obs, memory, mask):
    """Pre-vectorization scalar implementation (verbatim semantics)."""
    mask_a = np.asarray(mask, dtype=bool).reshape(-1)
    types = np.asarray(obs.type_grid, dtype=np.int32)
    owners = np.asarray(obs.owner_grid, dtype=np.int32)
    armies = np.asarray(obs.army_grid, dtype=np.int32)
    del types
    H, W = int(obs.H), int(obs.W)
    out = np.zeros(mask_a.shape, dtype=np.float64)
    seen_enemy = enemy_is_visible(obs, memory)

    turn = int(getattr(obs, "turn", 0))
    castle_w = castle_timing_weight(turn)
    from action import BASE_COST, live_build_cost

    cost_g = np.asarray(live_build_cost(obs, memory))
    for idx in np.flatnonzero(mask_a):
        idx = int(idx)
        action = decode_action(idx)
        if int(action[0]) != 2:
            continue
        r, c = int(action[1]), int(action[2])
        if int(cost_g[r, c]) != BASE_COST:
            continue  # surcharged builds are never rewarded
        army = float(max(int(armies[r, c]), 1))
        out[idx] = max(out[idx], castle_w * (10.0 + 0.2 * min(army, 100.0)))

    if seen_enemy:
        target = enemy_seek_target(obs, memory)
        goals = seek_goals(obs, memory)
        dist_field = path_distance_field(obs, goals) if goals else None
        from tactics import _movable_exclude_cell, wave_assembly_cell

        movable_exclude = _movable_exclude_cell(obs, memory)
        king = wave_assembly_cell(obs, memory, exclude=movable_exclude)
        king_dist = path_distance_field(obs, [king]) if king is not None else None
        share, max_own, tot = army_concentration(obs, exclude=movable_exclude)
        gen_known = enemy_general_visible(obs, memory)
        gen_cell = known_enemy_general_cell(obs, memory)
        hunt_dist = (
            path_distance_field(obs, [gen_cell]) if gen_cell is not None else None
        )
        for idx in np.flatnonzero(mask_a):
            idx = int(idx)
            if idx == PASS_INDEX:
                continue
            action = decode_action(idx)
            if int(action[0]) == 2:
                continue
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
            progress = path_progress(sr, sc, tr, tc, dist_field, fallback_target=target)
            bias = direction_bias(progress, dest_owner)
            surplus = float(max(army_i - dest_army - 1, 0))
            thrash = tip_thrash_factor(army_i, max_own, tot)
            gather = stack_gather_factor(army_i, dest_army, dest_owner, progress=progress)

            is_touch = gen_cell is not None and (tr, tc) == gen_cell
            if hunt_dist is not None:
                hunt_prog = path_progress(
                    sr, sc, tr, tc, hunt_dist, fallback_target=gen_cell
                )
            else:
                hunt_prog = 0.0
            hunt_factor = 1.0 + HUNT_PROGRESS_BONUS * max(hunt_prog, 0.0)
            if dest_owner == OWNER_ENEMY and surplus > 0.0:
                score = atk_w * thrash * bias * (
                    ENEMY_TAKE_BASE
                    + 40.0 * max(progress, 0.0)
                    + 8.0 * float(reveal)
                    + 3.5 * min(surplus, 200.0)
                )
                if gen_known:
                    score *= 1.35 if (is_touch or progress > 0.0) else GENERAL_CHEW_DAMP
                score *= hunt_factor
            elif dest_owner == OWNER_NEUTRAL:
                score = atk_w * thrash * bias * (
                    NEUTRAL_CARVE_BASE
                    + 30.0 * max(progress, 0.0)
                    + 12.0 * efficiency
                    + 8.0 * float(reveal)
                ) * hunt_factor
            elif dest_owner == 1:
                from tactics import CASTLE_CATCHMENT, castle_build_site

                k_prog = 0.0
                if king_dist is not None and not is_committed_army(
                    army_i, max_own, tot, share
                ):
                    k_prog = path_progress(
                        sr, sc, tr, tc, king_dist, fallback_target=king
                    )
                site = castle_build_site(obs, memory)
                if site is not None:
                    b_dist = path_distance_field(obs, [site])
                    if 0 <= int(b_dist[sr, sc]) <= CASTLE_CATCHMENT:
                        if is_committed_army(army_i, max_own, tot, share):
                            k_prog = 0.0
                        else:
                            k_prog = path_progress(
                                sr, sc, tr, tc, b_dist, fallback_target=site
                            )
                if k_prog > 0.0 and share < GATHER_SHARE_MIN:
                    score = (
                        attack_weight(army_i)
                        * (14.0 + 22.0 * k_prog)
                        * stack_gather_factor(army_i, dest_army, 1, progress=k_prog)
                    )
                elif progress > 0.0:
                    score = (
                        atk_w * thrash * bias * (18.0 + 32.0 * progress) * gather
                    ) * hunt_factor
                else:
                    score = 0.008 * army_w * bias * gather
            else:
                score = 0.01 * army_w * gather
            if turn >= DEATHTOUCH_TURN and is_touch:
                score = DEATHTOUCH_SCORE
            out[idx] = max(out[idx], score)
    else:
        urgency = fog_urgency(turn, enemy_seen=False)
        own_struct = own_structure_mask(obs, memory)
        fog_target = enemy_seek_target(obs, memory)
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
            reveal = (
                newly_revealed_cells(obs, tr, tc) if 0 <= tr < H and 0 <= tc < W else 0
            )
            dest_army = int(armies[tr, tc]) if 0 <= tr < H and 0 <= tc < W else 0
            eff = float(reveal) / float(explore_cost(dest_army))
            src_army = int(armies[sr, sc])
            ew = explore_wave_weight(src_army)
            progress = path_progress(sr, sc, tr, tc, fog_dist, fallback_target=fog_target)
            bias = direction_bias(progress, OWNER_NEUTRAL)
            score = ew * urgency * bias * (
                4.0
                + PRE_REVEAL_WEIGHT * float(reveal)
                + PRE_EFFICIENCY_WEIGHT * eff
                + PRE_PROGRESS_WEIGHT * max(progress, 0.0)
            )
            if bool(own_struct[sr, sc]) and src_army >= STRUCTURE_IDLE_ARMY:
                score *= 2.5 + 0.04 * float(min(src_army, 80))
            out[int(idx)] = max(out[int(idx)], score)
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
            progress = path_progress(sr, sc, tr, tc, fog_dist, fallback_target=fog_target)
            bias = direction_bias(progress, 1)
            gather = stack_gather_factor(src_army, dest_army, 1, progress=progress)
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
                    ew * urgency * tip_bonus * bias * (0.8 + 4.0 * progress) * gather,
                )
            else:
                out[idx] = max(out[idx], 0.008 * bias * gather)

    return np.where(mask_a, np.maximum(out, 0.0), 0.0)


@pytest.mark.parametrize("board", sorted(BOARDS))
def test_vectorized_scores_match_scalar_reference(board):
    obs, mem = BOARDS[board]()
    mask = play_mask(obs, mem)
    got = heuristic_action_scores(obs, mem, mask)
    want = reference_scores(obs, mem, mask)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_vectorized_scores_match_scalar_reference_fuzz(seed):
    from memory import empty_memory, update_memory
    from observe import emit_observation
    from state import create_initial_state

    rng = np.random.default_rng(seed)
    H = W = 10
    grid = np.zeros((H, W), dtype=np.int32)
    grid[H - 1, 0] = 1
    grid[0, W - 1] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    ownership = np.asarray(state.ownership, dtype=bool).copy()
    neut = np.asarray(state.ownership_neutral, dtype=bool).copy()
    # Random own blob near our corner; on odd seeds an enemy blob too
    # (post-contact path), even seeds stay pre-contact.
    for _ in range(14):
        r, c = int(rng.integers(4, H)), int(rng.integers(0, 6))
        ownership[0, r, c] = True
        neut[r, c] = False
        armies[r, c] = int(rng.integers(1, 60))
    if seed % 2 == 1:
        for _ in range(4):
            r, c = int(rng.integers(3, 7)), int(rng.integers(4, W))
            if not ownership[0, r, c]:
                ownership[1, r, c] = True
                neut[r, c] = False
                armies[r, c] = int(rng.integers(1, 20))
    state = state._replace(
        armies=armies, ownership=ownership, ownership_neutral=neut,
        time=int(rng.integers(0, 300)),
    )
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(H, W), obs)
    mask = play_mask(obs, mem)
    got = heuristic_action_scores(obs, mem, mask)
    want = reference_scores(obs, mem, mask)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


def test_reveal_grid_matches_per_cell_scalar():
    obs, _mem = BOARDS["contact"]()
    from tactics import reveal_count_grid

    grid = reveal_count_grid(obs)
    H, W = int(obs.H), int(obs.W)
    for r in range(H):
        for c in range(W):
            assert int(grid[r, c]) == newly_revealed_cells(obs, r, c), (r, c)
