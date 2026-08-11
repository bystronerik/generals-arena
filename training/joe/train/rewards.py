"""Sparse terminal reward — the only reward in this pipeline (D5).

The paper's ablation shows shaping destabilizes at high throughput; the
released code hardcodes win/loss over every YAML ``reward_fn`` key.
"""

import jax.numpy as jnp


def win_lose_reward(winners):
    """+1 if we won (winner == 0), -1 if we lost (winner == 1), else 0.

    ``winners`` is already from the acting player's perspective; the rollout
    flips it for the p1 seat before calling this.
    """
    return jnp.where(winners == 0, 1.0, jnp.where(winners == 1, -1.0, 0.0))
