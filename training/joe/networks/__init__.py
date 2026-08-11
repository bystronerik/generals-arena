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
    reset_done_envs,
    reset_obs_state,
)
from training.joe.networks.transformer import (  # noqa: F401
    HistoryTransformer,
    greedy_action_transformer,
)

NETWORK_REGISTRY = {
    "history_transformer": {
        "cls": HistoryTransformer,
        "init_obs_state": init_obs_state,
        "augment_obs": augment_obs,
        "reset_obs_state": reset_obs_state,
        "greedy_action": greedy_action_transformer,
    },
}


def get_network_bundle(name: str) -> dict:
    """Look up a network bundle (cls + obs state functions) by name."""
    if name not in NETWORK_REGISTRY:
        available = ", ".join(NETWORK_REGISTRY.keys())
        raise ValueError(f"Unknown network '{name}'. Available: {available}")
    return NETWORK_REGISTRY[name]


# Config fields forwarded to a network constructor when present in its signature.
_NET_CFG_FIELDS = [
    "depth", "embed_dim", "n_head", "ff_factor", "patch_size",
    "use_bf16", "value_loss", "num_bins", "v_min", "v_max",
]


def build_network(cfg, key):
    """Instantiate the configured network, forwarding only the config fields
    that match its constructor signature."""
    import inspect

    cls = get_network_bundle(cfg.network)["cls"]
    params = inspect.signature(cls).parameters
    kwargs = {"grid_size": cfg.pad_to, "pad_to": cfg.pad_to, "key": key}
    kwargs.update({f: getattr(cfg, f) for f in _NET_CFG_FIELDS if f in params})
    return cls(**kwargs)
