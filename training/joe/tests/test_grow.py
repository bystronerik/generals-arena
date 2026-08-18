"""Depth-growth surgery — parity, splice order, zeroed leaves, EMA path.

Growth plan section 2 (docs/research/strategies/joe-depth7-growth-plan.md)
at tiny dims: embed 32, depth 2 -> 3 (plus the layout rule at 5 -> 7).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.joe

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from training.joe.config import Config
from training.joe.grow import assert_forward_parity, grow_depth, splice_layout
from training.joe.networks import build_network


def tiny_cfg(depth, **overrides):
    kwargs = dict(pad_to=9, min_grid_size=9, max_grid_size=9, depth=depth,
                  embed_dim=32, n_head=4, ff_factor=1, patch_size=3,
                  num_bins=16)
    kwargs.update(overrides)
    return Config(**kwargs)


def _grow_pair(base_depth=2, new_depth=3):
    base = build_network(tiny_cfg(base_depth), jrandom.PRNGKey(0))
    grown = grow_depth(base, tiny_cfg(new_depth), jrandom.PRNGKey(100))
    return base, grown


def _count(tree):
    return sum(x.size for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array)))


def test_splice_layout_matches_plan():
    # Plan section 2: [L0, L1, NEW, L2, L3, NEW, L4] at 5 -> 7.
    assert splice_layout(5, 7) == [0, 1, None, 2, 3, None, 4]
    assert splice_layout(2, 3) == [0, None, 1]
    with pytest.raises(ValueError):
        splice_layout(3, 3)


def test_forward_parity_exact_under_bf16():
    # use_bf16 stays at the production default True: x + 0 == x bitwise.
    base, grown = _grow_pair()
    assert base.use_bf16 and grown.use_bf16
    assert_forward_parity(base, grown, n_samples=4)


def test_splice_preserves_base_layers():
    base, grown = _grow_pair()
    assert len(grown.transformer_layers) == 3
    for old_i, new_i in [(0, 0), (1, 2)]:
        a = base.transformer_layers[old_i]
        b = grown.transformer_layers[new_i]
        for x, y in zip(jax.tree.leaves(eqx.filter(a, eqx.is_array)),
                        jax.tree.leaves(eqx.filter(b, eqx.is_array))):
            assert bool(jnp.array_equal(x, y))


def test_new_block_zeroed_leaves():
    _, grown = _grow_pair()
    blk = grown.transformer_layers[1]
    for leaf in (blk.attn.out_proj.weight, blk.attn.out_proj.bias,
                 blk.ff_linear2.weight, blk.ff_linear2.bias):
        assert bool(jnp.all(leaf == 0.0))
    # The rest of the block keeps fresh init (it must receive gradients).
    for leaf in (blk.attn.q_proj.weight, blk.attn.k_proj.weight,
                 blk.attn.v_proj.weight, blk.ff_linear1.weight):
        assert not bool(jnp.all(leaf == 0.0))


def test_non_layer_fields_copied_from_base():
    base, grown = _grow_pair()
    for name in ("embedder", "value_token", "pos_encoding", "norm_out",
                 "policy_head", "value_head", "temporal_encoder",
                 "temporal_type_embed", "bin_centers"):
        a = jax.tree.leaves(eqx.filter(getattr(base, name), eqx.is_array))
        b = jax.tree.leaves(eqx.filter(getattr(grown, name), eqx.is_array))
        assert len(a) == len(b) and len(a) > 0
        for x, y in zip(a, b):
            assert bool(jnp.array_equal(x, y))


def test_param_count_grows_by_the_inserted_blocks():
    base, grown = _grow_pair()
    assert _count(grown) - _count(base) == _count(grown.transformer_layers[1])


def test_ema_lineage_parity_and_shared_fresh_init():
    # Both lineages are grown with the same key (plan section 2), so the
    # train net and the EMA net share fresh-block init, and each preserves
    # its own function.
    cfg3 = tiny_cfg(3)
    key = jrandom.PRNGKey(7)
    ema_net = build_network(tiny_cfg(2), jrandom.PRNGKey(1))
    train_net = build_network(tiny_cfg(2), jrandom.PRNGKey(0))
    g_train = grow_depth(train_net, cfg3, key)
    g_ema = grow_depth(ema_net, cfg3, key)
    assert_forward_parity(ema_net, g_ema, n_samples=4)
    assert bool(jnp.array_equal(g_train.transformer_layers[1].ff_linear1.weight,
                                g_ema.transformer_layers[1].ff_linear1.weight))


def test_grow_rejects_anything_but_deeper():
    base = build_network(tiny_cfg(2), jrandom.PRNGKey(0))
    with pytest.raises(ValueError):
        grow_depth(base, tiny_cfg(2), jrandom.PRNGKey(1))
    with pytest.raises(ValueError):
        grow_depth(base, tiny_cfg(3, embed_dim=64), jrandom.PRNGKey(1))
    with pytest.raises(ValueError):
        grow_depth(base, tiny_cfg(3, n_head=8), jrandom.PRNGKey(1))
