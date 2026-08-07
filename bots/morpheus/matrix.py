"""Pure simultaneous-matrix math for Morpheus search.

No tree storage. All values use the root-player perspective. Regret matching
plus updates never alternate signs.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

Array = np.ndarray

# Defaults — initial guesses; measurements may replace them.
SELF_WIDENING_CAP = 16
SELF_WIDENING_COEFF = 2.0
SELF_WIDENING_FLOOR = 8
ENEMY_WIDENING_CAP = 12
ENEMY_WIDENING_COEFF = 1.5
EXPLORATION_FLOOR = 0.05
EXPLORATION_NUMERATOR = 0.5


def widening_limit(n: int, *, coeff: float, cap: int) -> int:
    """``K(N) = min(cap, 1 + floor(coeff * sqrt(N)))``."""
    n = max(int(n), 0)
    return int(min(cap, 1 + math.floor(coeff * math.sqrt(n))))


def self_widening_limit(n: int) -> int:
    # Floor keeps root from spending early sims on pass-only before policy
    # expands enter (K(0)=1 would otherwise lock average strategy onto pass).
    return max(
        SELF_WIDENING_FLOOR,
        widening_limit(n, coeff=SELF_WIDENING_COEFF, cap=SELF_WIDENING_CAP),
    )


def enemy_widening_limit(n: int) -> int:
    return widening_limit(n, coeff=ENEMY_WIDENING_COEFF, cap=ENEMY_WIDENING_CAP)


def exploration_epsilon(
    n: int,
    *,
    floor: float = EXPLORATION_FLOOR,
    numerator: float = EXPLORATION_NUMERATOR,
) -> float:
    """``epsilon(N) = max(floor, numerator / sqrt(1 + N))``."""
    return float(max(floor, numerator / math.sqrt(1.0 + max(int(n), 0))))


def regret_matching_strategy(regrets: Array, prior: Array) -> Array:
    """Normalize positive regrets; if all zero, return the prior."""
    regrets = np.asarray(regrets, dtype=np.float64).reshape(-1)
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    if regrets.shape != prior.shape:
        raise ValueError("regrets and prior shape mismatch")
    positive = np.maximum(regrets, 0.0)
    total = float(positive.sum())
    if total <= 0.0:
        p = np.maximum(prior, 0.0)
        s = float(p.sum())
        if s <= 0.0:
            return np.full_like(prior, 1.0 / max(len(prior), 1))
        return p / s
    return positive / total


def mixed_strategy(
    regrets: Array,
    prior: Array,
    n: int,
    *,
    floor: float = EXPLORATION_FLOOR,
    numerator: float = EXPLORATION_NUMERATOR,
) -> Array:
    """``sigma = (1 - epsilon) * regret_strategy + epsilon * prior``."""
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    p = np.maximum(prior, 0.0)
    s = float(p.sum())
    if s <= 0.0:
        p = np.full_like(prior, 1.0 / max(len(prior), 1))
    else:
        p = p / s
    regret_sigma = regret_matching_strategy(regrets, p)
    eps = exploration_epsilon(n, floor=floor, numerator=numerator)
    return (1.0 - eps) * regret_sigma + eps * p


def effective_q(
    visits: Array,
    q: Array,
    first_play: float,
) -> Array:
    """Visited entries use Q; unvisited entries use first-play urgency."""
    visits = np.asarray(visits, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return np.where(visits > 0.0, q, float(first_play))


def matrix_utilities(
    sigma_self: Array,
    sigma_enemy: Array,
    q_eff: Array,
) -> tuple[Array, Array, float]:
    """Return ``(u_self, u_enemy, v)`` for one enemy-hash submatrix."""
    sigma_self = np.asarray(sigma_self, dtype=np.float64).reshape(-1)
    sigma_enemy = np.asarray(sigma_enemy, dtype=np.float64).reshape(-1)
    q_eff = np.asarray(q_eff, dtype=np.float64)
    if q_eff.shape != (len(sigma_self), len(sigma_enemy)):
        raise ValueError(
            f"Q shape {q_eff.shape} != ({len(sigma_self)}, {len(sigma_enemy)})"
        )
    u_self = q_eff @ sigma_enemy
    u_enemy = sigma_self @ q_eff
    v = float(sigma_self @ u_self)
    return u_self, u_enemy, v


def aggregate_self_utilities(
    sigma_self: Array,
    enemy_weights: Array,
    enemy_sigmas: Sequence[Array],
    q_eff_list: Sequence[Array],
) -> tuple[Array, float]:
    """Particle-weighted ``u_self`` across enemy hashes; ``v = sigma_A · u_self``."""
    sigma_self = np.asarray(sigma_self, dtype=np.float64).reshape(-1)
    enemy_weights = np.asarray(enemy_weights, dtype=np.float64).reshape(-1)
    if len(enemy_weights) != len(enemy_sigmas) or len(enemy_weights) != len(q_eff_list):
        raise ValueError("enemy weight / sigma / Q list length mismatch")
    u_self = np.zeros(len(sigma_self), dtype=np.float64)
    w = np.maximum(enemy_weights, 0.0)
    total = float(w.sum())
    if total <= 0.0 and len(w) > 0:
        w = np.full(len(w), 1.0 / len(w))
    elif total > 0.0:
        w = w / total
    for weight, sigma_b, q_eff in zip(w, enemy_sigmas, q_eff_list):
        if weight <= 0.0:
            continue
        u_h, _, _ = matrix_utilities(sigma_self, sigma_b, q_eff)
        u_self = u_self + weight * u_h
    v = float(sigma_self @ u_self)
    return u_self, v


def regret_plus_update(
    regrets: Array,
    utilities: Array,
    value: float,
    *,
    maximizing: bool,
) -> Array:
    """Regret matching plus: clip cumulative regrets at zero."""
    regrets = np.asarray(regrets, dtype=np.float64).reshape(-1).copy()
    utilities = np.asarray(utilities, dtype=np.float64).reshape(-1)
    if maximizing:
        regrets = regrets + utilities - float(value)
    else:
        regrets = regrets + float(value) - utilities
    return np.maximum(regrets, 0.0)


def accumulate_average_strategy(avg: Array, sigma: Array) -> Array:
    """Add the current mixed strategy into the cumulative average."""
    avg = np.asarray(avg, dtype=np.float64).reshape(-1).copy()
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    return avg + sigma


def normalize_average_strategy(avg: Array) -> Array:
    avg = np.asarray(avg, dtype=np.float64).reshape(-1)
    s = float(np.maximum(avg, 0.0).sum())
    if s <= 0.0:
        n = max(len(avg), 1)
        return np.full(n, 1.0 / n, dtype=np.float64)
    return np.maximum(avg, 0.0) / s


def sample_index(probs: Array, rng: np.random.Generator) -> int:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    p = np.maximum(probs, 0.0)
    total = float(p.sum())
    if total <= 0.0:
        return int(rng.integers(0, len(p)))
    return int(rng.choice(len(p), p=p / total))


def apply_joint_backup(
    *,
    visits: Array,
    value_sum: Array,
    q: Array,
    a_idx: int,
    b_idx: int,
    leaf_value: float,
) -> tuple[Array, Array, Array]:
    """Update ``N``, ``W``, ``Q`` for one joint entry. Returns new arrays."""
    visits = np.asarray(visits, dtype=np.float64).copy()
    value_sum = np.asarray(value_sum, dtype=np.float64).copy()
    q = np.asarray(q, dtype=np.float64).copy()
    visits[a_idx, b_idx] += 1.0
    value_sum[a_idx, b_idx] += float(leaf_value)
    q[a_idx, b_idx] = value_sum[a_idx, b_idx] / visits[a_idx, b_idx]
    return visits, value_sum, q


def select_root_action(
    avg_strategy: Array,
    marginal_visits: Array,
    prior: Array,
    legal_mask: Optional[Array] = None,
) -> int:
    """Largest normalized ``S_A``; ties by marginal visits, then prior.

    Near-zero prior entries are ignored so early progressive-widening mass on
    pass cannot win after the live network prior has moved elsewhere.
    """
    avg = normalize_average_strategy(avg_strategy)
    visits = np.asarray(marginal_visits, dtype=np.float64).reshape(-1)
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    if legal_mask is not None:
        mask = np.asarray(legal_mask, dtype=bool).reshape(-1)
        avg = np.where(mask, avg, -1.0)
        visits = np.where(mask, visits, -1.0)
        prior = np.where(mask, prior, -1.0)
    # Drop stale average/visit mass on actions the live prior has zeroed out.
    max_prior = float(np.max(prior)) if prior.size else 0.0
    if max_prior > 0.0:
        active = prior > max(1e-12, 1e-6 * max_prior)
        if bool(active.any()) and not bool(active.all()):
            avg = np.where(active, avg, -1.0)
            visits = np.where(active, visits, -1.0)
    # Lexicographic: strategy, visits, prior.
    order = np.lexsort((-prior, -visits, -avg))
    return int(order[0])
