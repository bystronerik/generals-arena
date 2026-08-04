"""Part 12 — hand-computed losses and weighted objective."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.morpheus

from training.morpheus.objective.config import (
    LossWeights,
    ObjectiveConfig,
    ObjectiveConfigError,
    ExplorationConfig,
    ScalarNormalization,
)
from training.morpheus.objective.losses import (
    combine_losses,
    compute_objective_losses,
    hand_policy_ce,
    hand_wdl_ce,
    huber_regression,
    policy_cross_entropy,
    wdl_cross_entropy,
)


def _weights(**overrides: float) -> LossWeights:
    base = {
        "policy": 1.0,
        "wdl": 1.0,
        "hidden_owner": 0.0,
        "enemy_army_bins": 0.0,
        "enemy_general": 0.0,
        "hidden_castle": 0.0,
        "land_margin": 0.0,
        "army_margin": 0.0,
        "castle_margin": 0.0,
        "turns_to_termination": 0.0,
    }
    base.update(overrides)
    return LossWeights(**base)


def _config(weights: LossWeights | None = None) -> ObjectiveConfig:
    return ObjectiveConfig(
        name="test",
        mode="training",
        loss_weights=weights or _weights(),
        exploration=ExplorationConfig(
            root_noise_epsilon=0.0,
            root_noise_alpha=0.3,
            action_temperature=1.0,
            deterministic_turn=0,
        ),
        scalar_normalization=ScalarNormalization(
            land_margin_scale=1.0,
            army_margin_scale=1.0,
            castle_margin_scale=1.0,
            turns_to_termination_scale=1200.0,
        ),
    )


def test_policy_ce_matches_uniform_hand_value():
    n = 8
    logits = np.zeros(n, dtype=np.float64)
    target = np.zeros(n, dtype=np.float64)
    target[3] = 1.0
    expected = float(-np.log(1.0 / n))
    assert abs(hand_policy_ce(logits, target) - expected) < 1e-6


def test_wdl_ce_matches_uniform_hand_value():
    logits = np.zeros(3, dtype=np.float64)
    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    expected = float(-np.log(1.0 / 3.0))
    assert abs(hand_wdl_ce(logits, target) - expected) < 1e-6


def test_combine_losses_requires_every_weight_key():
    terms = {
        "policy": torch.tensor(1.0),
        "wdl": torch.tensor(2.0),
    }
    with pytest.raises(ObjectiveConfigError, match="missing"):
        combine_losses(terms, {"policy": 1.0, "wdl": 1.0})


def test_weighted_total_is_explicit_sum():
    terms = {
        "policy": torch.tensor(2.0),
        "wdl": torch.tensor(4.0),
        "hidden_owner": torch.tensor(1.0),
        "enemy_army_bins": torch.tensor(1.0),
        "enemy_general": torch.tensor(1.0),
        "hidden_castle": torch.tensor(1.0),
        "land_margin": torch.tensor(1.0),
        "army_margin": torch.tensor(1.0),
        "castle_margin": torch.tensor(1.0),
        "turns_to_termination": torch.tensor(1.0),
    }
    weights = _weights(policy=0.5, wdl=0.25, hidden_owner=1.0)
    total, packed = combine_losses(terms, weights)
    assert abs(float(total) - (0.5 * 2.0 + 0.25 * 4.0 + 1.0)) < 1e-6
    assert packed.total == pytest.approx(float(total))


def test_compute_objective_losses_finite_on_aligned_batch():
    cfg = _config(_weights(hidden_owner=0.1, enemy_army_bins=0.1, enemy_general=0.1))
    n_actions = 16
    pad = 4
    n_bins = 16
    policy_logits = torch.zeros(1, n_actions)
    policy_target = torch.zeros(1, n_actions)
    policy_target[0, 0] = 1.0
    wdl_logits = torch.tensor([[3.0, 0.0, -3.0]])
    wdl_target = torch.tensor([[1.0, 0.0, 0.0]])
    hidden_owner = torch.zeros(1, 1, pad, pad)
    hidden_owner[0, 0, 1, 1] = 1.0
    army_logits = torch.zeros(1, n_bins, pad, pad)
    army_logits[0, 3, 1, 1] = 5.0
    army_bins = torch.full((1, pad, pad), -1, dtype=torch.long)
    army_bins[0, 1, 1] = 3
    enemy_general = torch.zeros(1, 1, pad, pad)
    enemy_general[0, 0, 1, 1] = 1.0
    hidden_castle = torch.zeros(1, 1, pad, pad)
    board = torch.ones(1, 1, pad, pad)
    total, terms = compute_objective_losses(
        config=cfg,
        policy_logits=policy_logits,
        wdl_logits=wdl_logits,
        hidden_owner_logits=hidden_owner * 5,
        enemy_army_logits=army_logits,
        enemy_general_logits=enemy_general * 5,
        hidden_castle_logits=hidden_castle,
        land_margin_pred=torch.tensor([[0.1]]),
        army_margin_pred=torch.tensor([[0.2]]),
        castle_margin_pred=torch.tensor([[0.0]]),
        turns_pred=torch.tensor([[0.01]]),
        policy_target=policy_target,
        wdl_target=wdl_target,
        hidden_owner_target=hidden_owner,
        enemy_army_bin_target=army_bins,
        enemy_general_target=enemy_general,
        hidden_castle_target=hidden_castle,
        land_margin_target=torch.tensor([[0.1]]),
        army_margin_target=torch.tensor([[0.2]]),
        castle_margin_target=torch.tensor([[0.0]]),
        turns_target=torch.tensor([[0.01]]),
        board_mask=board,
    )
    assert torch.isfinite(total)
    assert terms.total == pytest.approx(float(total))


def test_huber_zero_when_equal():
    pred = torch.tensor([1.0, -0.5])
    target = torch.tensor([1.0, -0.5])
    assert float(huber_regression(pred, target)) == pytest.approx(0.0)


def test_policy_mask_zeros_illegal_mass():
    logits = torch.tensor([[0.0, 0.0, 0.0, 5.0]])
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss = policy_cross_entropy(logits, target, mask)
    # Softmax only over first two → equal probs → CE = -log(0.5)
    assert float(loss) == pytest.approx(-np.log(0.5), rel=1e-5)


def test_wdl_perfect_prediction_near_zero():
    logits = torch.tensor([[20.0, -20.0, -20.0]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    assert float(wdl_cross_entropy(logits, target)) < 1e-6
