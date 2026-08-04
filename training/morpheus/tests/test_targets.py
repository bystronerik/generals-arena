"""Part 12 — hand-computed training targets and terminal reward."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from training.morpheus.objective.config import (
    ExplorationConfig,
    ObjectiveConfig,
    ObjectiveConfigError,
    ScalarNormalization,
    assert_rated_disables_exploration,
)
from training.morpheus.objective.reward import (
    DISCOUNT,
    assert_no_shaping,
    reward_contract,
    terminal_reward,
    wdl_one_hot,
)
from training.morpheus.objective.targets import (
    apply_root_noise,
    army_bin_index,
    build_seat_targets,
    normalize_margin,
    sample_action_from_strategy,
    sparse_policy_to_dense,
)
from training.morpheus.self_play.schema import SparsePolicy


def _board():
    H = W = 3
    ownership = np.zeros((2, H, W), dtype=bool)
    ownership[0, 0, 0] = True
    ownership[1, 2, 2] = True
    ownership[1, 2, 1] = True
    armies = np.zeros((H, W), dtype=np.int32)
    armies[0, 0] = 4
    armies[2, 2] = 16
    armies[2, 1] = 2
    generals = np.zeros((H, W), dtype=bool)
    generals[0, 0] = True
    generals[2, 2] = True
    castles = np.zeros((H, W), dtype=bool)
    castles[2, 1] = True
    return ownership, armies, generals, castles


def test_terminal_wdl_hand_values():
    assert terminal_reward("a", seat=0) == 1.0
    assert terminal_reward("a", seat=1) == -1.0
    assert terminal_reward("b", seat=0) == -1.0
    assert terminal_reward("draw", seat=0) == 0.0
    assert wdl_one_hot("a", seat=0) == (1.0, 0.0, 0.0)
    assert wdl_one_hot("draw", seat=1) == (0.0, 1.0, 0.0)
    assert wdl_one_hot("b", seat=0) == (0.0, 0.0, 1.0)
    assert DISCOUNT == 1.0
    contract = reward_contract()
    assert contract["win"] == 1.0
    assert contract["loss"] == -1.0
    assert contract["nonterminal"] == 0.0


def test_reward_rejects_shaping_terms():
    with pytest.raises(ValueError, match="shaping"):
        assert_no_shaping({"win": 1.0, "land": 0.01})
    with pytest.raises(ValueError, match="shaping"):
        assert_no_shaping({"army": 1.0})
    assert_no_shaping(reward_contract())


def test_sparse_root_average_strategy_to_dense():
    policy = SparsePolicy(indices=(0, 5, 3969), probs=(0.2, 0.3, 0.5))
    dense = sparse_policy_to_dense(policy)
    assert dense.shape == (3970,)
    assert abs(dense.sum() - 1.0) < 1e-9
    assert dense[0] == 0.2
    assert dense[5] == 0.3
    assert dense[3969] == 0.5
    with pytest.raises(ValueError):
        SparsePolicy(indices=(0, 1), probs=(0.5, 0.6))


def test_hidden_state_and_margin_targets_from_engine_truth():
    ownership, armies, generals, castles = _board()
    policy = SparsePolicy(indices=(3969,), probs=(1.0,))
    t0 = build_seat_targets(
        winner="a",
        seat=0,
        policy=policy,
        ownership=ownership,
        armies=armies,
        generals=generals,
        castles=castles,
        final_land=(1, 2),
        final_army=(4, 18),
        final_castles=(0, 1),
        current_turn=5,
        terminal_turn=25,
    )
    assert t0.value == 1.0
    assert t0.turns_to_termination == 20.0
    assert abs(t0.land_margin - normalize_margin(1, 2)) < 1e-12
    assert abs(t0.army_margin - normalize_margin(4, 18)) < 1e-12
    assert abs(t0.castle_margin - normalize_margin(0, 1)) < 1e-12
    # Enemy cells from seat 0
    assert t0.hidden_owner[2, 2] == 1.0
    assert t0.hidden_owner[2, 1] == 1.0
    assert t0.hidden_owner[0, 0] == 0.0
    assert t0.enemy_general[2, 2] == 1.0
    assert t0.hidden_castle[2, 1] == 1.0
    assert t0.enemy_army_bin[2, 2] == army_bin_index(16)
    assert t0.enemy_army_bin[0, 0] == -1


def test_army_bin_uses_part04_edges():
    import sys
    from pathlib import Path

    bot = Path(__file__).resolve().parents[3] / "bots" / "morpheus"
    if str(bot) not in sys.path:
        sys.path.insert(0, str(bot))
    from schema import ARMY_BIN_EDGES as EDGES, N_ARMY_BINS as NB

    assert len(EDGES) == NB + 1
    assert army_bin_index(0.0) == 0
    assert army_bin_index(EDGES[-1]) == NB - 1
    assert army_bin_index(1.0) >= 0
    # Monotone in army
    assert army_bin_index(2.0) <= army_bin_index(100.0)


def test_omitted_objective_fields_fail_closed():
    with pytest.raises(ObjectiveConfigError, match="missing"):
        ObjectiveConfig.from_dict({"name": "x"})
    with pytest.raises(ObjectiveConfigError, match="loss_weights"):
        ObjectiveConfig.from_dict(
            {
                "name": "x",
                "loss_weights": {"policy": 1.0},
                "exploration": ExplorationConfig.rated().to_dict(),
                "scalar_normalization": {
                    "land_margin_scale": 1.0,
                    "army_margin_scale": 1.0,
                    "castle_margin_scale": 1.0,
                    "turns_to_termination_scale": 1200.0,
                },
            }
        )


def test_rated_mode_disables_exploration():
    from training.morpheus.objective.config import LossWeights

    weights = LossWeights(
        policy=1.0,
        wdl=1.0,
        hidden_owner=0.0,
        enemy_army_bins=0.0,
        enemy_general=0.0,
        hidden_castle=0.0,
        land_margin=0.0,
        army_margin=0.0,
        castle_margin=0.0,
        turns_to_termination=0.0,
    )
    norm = ScalarNormalization(
        land_margin_scale=1.0,
        army_margin_scale=1.0,
        castle_margin_scale=1.0,
        turns_to_termination_scale=1200.0,
    )
    rated = ObjectiveConfig(
        name="rated",
        mode="rated",
        loss_weights=weights,
        exploration=ExplorationConfig.rated(),
        scalar_normalization=norm,
    )
    assert_rated_disables_exploration(rated)
    with pytest.raises(ObjectiveConfigError, match="rated mode forbids"):
        ObjectiveConfig(
            name="bad",
            mode="rated",
            loss_weights=weights,
            exploration=ExplorationConfig(
                root_noise_epsilon=0.2,
                root_noise_alpha=0.3,
                action_temperature=1.0,
                deterministic_turn=0,
            ),
            scalar_normalization=norm,
        )


def test_root_noise_and_temperature_sampling():
    prior = np.array([0.7, 0.2, 0.1], dtype=np.float64)
    rng = np.random.default_rng(0)
    quiet = apply_root_noise(prior, epsilon=0.0, alpha=0.3, rng=rng)
    np.testing.assert_allclose(quiet, prior)
    noisy = apply_root_noise(prior, epsilon=0.5, alpha=0.3, rng=rng)
    assert abs(noisy.sum() - 1.0) < 1e-9
    # After deterministic_turn, always argmax
    idx = sample_action_from_strategy(
        prior,
        temperature=2.0,
        turn=10,
        deterministic_turn=5,
        rng=rng,
    )
    assert idx == 0
