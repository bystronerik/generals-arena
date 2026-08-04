"""Part 12: training objective — targets, losses, augmentation, config."""

from training.morpheus.objective.augmentation import (
    AugmentableSample,
    apply_symmetry,
    invert_symmetry,
    round_trip_ok,
)
from training.morpheus.objective.config import (
    ExplorationConfig,
    LossWeights,
    ObjectiveConfig,
    ObjectiveConfigError,
    ScalarNormalization,
    assert_rated_disables_exploration,
    load_ablation_candidates,
    load_objective_config,
)
from training.morpheus.objective.losses import (
    LossTerms,
    combine_losses,
    compute_objective_losses,
    hand_policy_ce,
    hand_wdl_ce,
)
from training.morpheus.objective.reward import (
    DISCOUNT,
    DRAW,
    LOSS,
    WIN,
    assert_no_shaping,
    reward_contract,
    terminal_reward,
    wdl_one_hot,
)
from training.morpheus.objective.targets import (
    SeatTargets,
    apply_root_noise,
    army_bin_index,
    build_seat_targets,
    sample_action_from_strategy,
    sparse_policy_to_dense,
)

__all__ = [
    "AugmentableSample",
    "DISCOUNT",
    "DRAW",
    "ExplorationConfig",
    "LOSS",
    "LossTerms",
    "LossWeights",
    "ObjectiveConfig",
    "ObjectiveConfigError",
    "ScalarNormalization",
    "SeatTargets",
    "WIN",
    "apply_root_noise",
    "apply_symmetry",
    "army_bin_index",
    "assert_no_shaping",
    "assert_rated_disables_exploration",
    "build_seat_targets",
    "combine_losses",
    "compute_objective_losses",
    "hand_policy_ce",
    "hand_wdl_ce",
    "invert_symmetry",
    "load_ablation_candidates",
    "load_objective_config",
    "reward_contract",
    "round_trip_ok",
    "sample_action_from_strategy",
    "sparse_policy_to_dense",
    "terminal_reward",
    "wdl_one_hot",
]
