"""Part 06 — hand-computed matrix backup and regret-matching-plus fixtures."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from matrix import (
    aggregate_self_utilities,
    apply_joint_backup,
    effective_q,
    enemy_widening_limit,
    exploration_epsilon,
    matrix_utilities,
    mixed_strategy,
    normalize_average_strategy,
    regret_matching_strategy,
    regret_plus_update,
    select_root_action,
    self_widening_limit,
)


def test_widening_and_epsilon_defaults():
    assert self_widening_limit(0) == 1
    assert self_widening_limit(1) == 3  # 1 + floor(2*1) = 3
    assert self_widening_limit(100) == 16  # capped
    assert enemy_widening_limit(0) == 1
    assert enemy_widening_limit(4) == 4  # 1 + floor(1.5*2) = 4
    assert enemy_widening_limit(100) == 12
    assert abs(exploration_epsilon(0) - 0.5) < 1e-12
    assert abs(exploration_epsilon(99) - 0.05) < 1e-12  # floor


def test_regret_matching_falls_back_to_prior():
    prior = np.array([0.25, 0.75])
    sigma = regret_matching_strategy(np.zeros(2), prior)
    np.testing.assert_allclose(sigma, prior)


def test_hand_computed_2x2_backup():
    """Fixture: Q = [[0, -1], [1, 0]], uniform sigma, first-play unused."""
    visits = np.ones((2, 2), dtype=np.float64)
    q = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    sigma_a = np.array([0.5, 0.5])
    sigma_b = np.array([0.5, 0.5])
    q_eff = effective_q(visits, q, first_play=0.25)
    u_self, u_enemy, v = matrix_utilities(sigma_a, sigma_b, q_eff)
    # u_self[0] = 0.5*0 + 0.5*(-1) = -0.5
    # u_self[1] = 0.5*1 + 0.5*0 = 0.5
    # v = 0.5*(-0.5) + 0.5*0.5 = 0
    np.testing.assert_allclose(u_self, [-0.5, 0.5])
    np.testing.assert_allclose(u_enemy, [0.5, -0.5])
    assert abs(v - 0.0) < 1e-12

    r_a = regret_plus_update(np.zeros(2), u_self, v, maximizing=True)
    # R[0] = max(0, -0.5 - 0) = 0; R[1] = max(0, 0.5 - 0) = 0.5
    np.testing.assert_allclose(r_a, [0.0, 0.5])
    r_b = regret_plus_update(np.zeros(2), u_enemy, v, maximizing=False)
    # R_B[b] += v - u_enemy(b) → [0-0.5, 0-(-0.5)] clipped → [0, 0.5]
    np.testing.assert_allclose(r_b, [0.0, 0.5])


def test_first_play_urgency_in_utilities():
    visits = np.array([[1.0, 0.0], [0.0, 0.0]])
    q = np.array([[0.5, 0.0], [0.0, 0.0]])
    q_eff = effective_q(visits, q, first_play=0.2)
    np.testing.assert_allclose(q_eff, [[0.5, 0.2], [0.2, 0.2]])


def test_particle_weighted_self_utility():
    sigma_a = np.array([1.0])
    sig_b0 = np.array([1.0])
    sig_b1 = np.array([1.0])
    q0 = np.array([[1.0]])
    q1 = np.array([[-1.0]])
    u_self, v = aggregate_self_utilities(
        sigma_a,
        np.array([0.5, 0.5]),
        [sig_b0, sig_b1],
        [q0, q1],
    )
    np.testing.assert_allclose(u_self, [0.0])
    assert abs(v - 0.0) < 1e-12


def test_joint_visit_updates_q():
    visits = np.zeros((1, 1))
    w = np.zeros((1, 1))
    q = np.zeros((1, 1))
    visits, w, q = apply_joint_backup(
        visits=visits, value_sum=w, q=q, a_idx=0, b_idx=0, leaf_value=1.0
    )
    assert visits[0, 0] == 1.0
    assert q[0, 0] == 1.0
    visits, w, q = apply_joint_backup(
        visits=visits, value_sum=w, q=q, a_idx=0, b_idx=0, leaf_value=-1.0
    )
    assert visits[0, 0] == 2.0
    assert abs(q[0, 0] - 0.0) < 1e-12


def test_root_selection_tie_breaks():
    avg = np.array([1.0, 1.0])
    visits = np.array([2.0, 5.0])
    prior = np.array([0.9, 0.1])
    assert select_root_action(avg, visits, prior) == 1
    assert select_root_action(avg, np.array([3.0, 3.0]), prior) == 0


def test_mixed_strategy_mixes_prior():
    regrets = np.array([1.0, 0.0])
    prior = np.array([0.5, 0.5])
    sigma = mixed_strategy(regrets, prior, n=10_000)
    np.testing.assert_allclose(sigma, [0.95 + 0.025, 0.025], atol=1e-9)


def test_normalize_average_strategy():
    np.testing.assert_allclose(
        normalize_average_strategy(np.array([2.0, 6.0])), [0.25, 0.75]
    )
