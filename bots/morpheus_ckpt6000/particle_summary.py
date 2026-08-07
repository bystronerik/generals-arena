"""Aggregate weighted particles into tensor BeliefSummary planes."""
from __future__ import annotations

import numpy as np

from belief import BeliefState, ess_fraction
from observe import visibility_mask
from tensor import BeliefSummary

Array = np.ndarray


def summarize_belief(belief: BeliefState) -> BeliefSummary:
    """Weighted particle aggregates in raw army units (Part 03 contract)."""
    if belief.n == 0:
        H = W = 1
        z = np.zeros((H, W), dtype=np.float32)
        return BeliefSummary(
            enemy_owner=z,
            enemy_army_mean=z,
            enemy_army_std=z,
            enemy_general=z,
            enemy_castle_owner=z,
            enemy_visibility=z,
            ess_fraction=0.0,
        )

    H, W = belief.particles[0].state.armies.shape
    enemy = belief.enemy_seat
    weights = np.asarray([max(p.weight, 0.0) for p in belief.particles], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        weights = np.full(belief.n, 1.0 / belief.n, dtype=np.float64)
    else:
        weights = weights / total

    owner = np.zeros((H, W), dtype=np.float64)
    army_mean = np.zeros((H, W), dtype=np.float64)
    army_sq = np.zeros((H, W), dtype=np.float64)
    general = np.zeros((H, W), dtype=np.float64)
    castle_owner = np.zeros((H, W), dtype=np.float64)
    visibility = np.zeros((H, W), dtype=np.float64)

    for w, particle in zip(weights, belief.particles):
        state = particle.state
        e_own = state.ownership[enemy]
        owner += w * e_own.astype(np.float64)
        e_army = (state.armies * e_own).astype(np.float64)
        army_mean += w * e_army
        army_sq += w * (e_army * e_army)
        g = state.general_positions[enemy]
        gr, gc = int(g[0]), int(g[1])
        if 0 <= gr < H and 0 <= gc < W:
            general[gr, gc] += w
        castle_owner += w * (state.castles & e_own).astype(np.float64)
        visibility += w * visibility_mask(e_own).astype(np.float64)

    var = np.maximum(army_sq - army_mean * army_mean, 0.0)
    army_std = np.sqrt(var)

    return BeliefSummary(
        enemy_owner=owner.astype(np.float32),
        enemy_army_mean=army_mean.astype(np.float32),
        enemy_army_std=army_std.astype(np.float32),
        enemy_general=general.astype(np.float32),
        enemy_castle_owner=castle_owner.astype(np.float32),
        enemy_visibility=visibility.astype(np.float32),
        ess_fraction=float(ess_fraction(belief)),
    )
