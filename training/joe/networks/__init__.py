"""Network port of AverageJoe networks/ for competition rules.

39 observation channels (paper's 38 + own build cost) and a 10-action
policy head (4 full moves, 4 half moves, pass, build). Plan section 4.
"""

from training.joe.networks.common import (  # noqa: F401
    AugmentedObsState,
    N_CHANNELS,
    augment_obs,
    build_cost_from_obs,
    compute_build_mask,
    decode_action,
    encode_action,
    init_obs_state,
    normalize_observations,
    obs_to_array,
    prepare_action_mask,
    reset_obs_state,
)
from training.joe.networks.transformer import (  # noqa: F401
    HistoryTransformer,
    greedy_action_transformer,
)
