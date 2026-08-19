"""Growth surgery — parity, splice order, zeroed leaves, EMA path.

Depth growth: docs/research/strategies/joe-depth7-growth-plan.md section 2,
at tiny dims — embed 32, depth 2 -> 3 (plus the layout rule at 5 -> 7).

FF growth: docs/research/strategies/joe-ff4-growth-plan.md section 2, at the
same tiny dims — ff_factor 1 -> 3, every block edited.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.joe

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from training.joe.config import Config
from training.joe.grow import (
    assert_forward_parity, grow_depth, grow_ff, splice_layout)
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


# ---- FF growth (ff4 plan section 2) ----

FF_ATOL = 1e-5  # ff4 plan section 2: f32-epsilon scale


def _grow_ff_pair(base_ff=1, new_ff=3, depth=2):
    base = build_network(tiny_cfg(depth, ff_factor=base_ff), jrandom.PRNGKey(0))
    grown = grow_ff(base, tiny_cfg(depth, ff_factor=new_ff), jrandom.PRNGKey(100))
    return base, grown


def test_ff_forward_parity_within_tolerance_and_argmax():
    # The FFN GEMM shapes change, so the check is bitwise-first with the
    # tolerance + argmax fallback; either outcome must agree on argmax.
    base, grown = _grow_ff_pair()
    assert base.use_bf16 and grown.use_bf16
    report = assert_forward_parity(base, grown, n_samples=4, atol=FF_ATOL)
    assert report["argmax_agreement"] == 1.0
    assert all(d <= FF_ATOL for d in report["max_abs_diff"].values())


def test_ff_zeroed_columns_appended_in_every_layer():
    base, grown = _grow_ff_pair()
    assert len(grown.transformer_layers) == len(base.transformer_layers)
    for b, g in zip(base.transformer_layers, grown.transformer_layers):
        old_hidden = b.ff_linear2.weight.shape[1]
        assert g.ff_linear2.weight.shape == (b.ff_linear2.weight.shape[0],
                                             3 * old_hidden)
        assert bool(jnp.array_equal(g.ff_linear2.weight[:, :old_hidden],
                                    b.ff_linear2.weight))
        assert bool(jnp.all(g.ff_linear2.weight[:, old_hidden:] == 0.0))
        # The second-layer bias is untouched: the new units add exact zero.
        assert bool(jnp.array_equal(g.ff_linear2.bias, b.ff_linear2.bias))


def test_ff_fresh_rows_appended_in_every_layer():
    base, grown = _grow_ff_pair()
    tails = []
    for b, g in zip(base.transformer_layers, grown.transformer_layers):
        old_hidden = b.ff_linear1.weight.shape[0]
        assert bool(jnp.array_equal(g.ff_linear1.weight[:old_hidden],
                                    b.ff_linear1.weight))
        assert bool(jnp.array_equal(g.ff_linear1.bias[:old_hidden],
                                    b.ff_linear1.bias))
        tail = g.ff_linear1.weight[old_hidden:]
        # Fresh random init, not zeros and not a copy of the base rows.
        assert not bool(jnp.all(tail == 0.0))
        assert not bool(jnp.array_equal(tail[:old_hidden], b.ff_linear1.weight))
        tails.append(tail)
    # One independent draw per block, so no two blocks share new units.
    assert not bool(jnp.array_equal(tails[0], tails[1]))


def test_ff_leaves_outside_the_ffn_are_copied_from_base():
    base, grown = _grow_ff_pair()
    for name in ("embedder", "value_token", "pos_encoding", "norm_out",
                 "policy_head", "value_head", "temporal_encoder",
                 "temporal_type_embed", "bin_centers"):
        a = jax.tree.leaves(eqx.filter(getattr(base, name), eqx.is_array))
        b = jax.tree.leaves(eqx.filter(getattr(grown, name), eqx.is_array))
        assert len(a) == len(b) and len(a) > 0
        for x, y in zip(a, b):
            assert bool(jnp.array_equal(x, y))
    for b, g in zip(base.transformer_layers, grown.transformer_layers):
        for part in ("norm1", "attn", "norm2"):
            for x, y in zip(
                    jax.tree.leaves(eqx.filter(getattr(b, part), eqx.is_array)),
                    jax.tree.leaves(eqx.filter(getattr(g, part), eqx.is_array))):
                assert bool(jnp.array_equal(x, y))


def test_ff_param_count_and_leaf_count():
    base, grown = _grow_ff_pair()
    d_model = base.transformer_layers[0].ff_linear2.weight.shape[0]
    n_new = 2 * base.transformer_layers[0].ff_linear1.weight.shape[0]
    depth = len(base.transformer_layers)
    # Per block: new ff_linear1 rows + their bias, new zeroed ff_linear2
    # columns. ff_linear2's bias does not grow.
    assert _count(grown) - _count(base) == depth * (2 * n_new * d_model + n_new)
    # Leaf count is unchanged: the graft reshapes leaves, it does not add any.
    assert len(jax.tree.leaves(eqx.filter(grown, eqx.is_array))) == \
        len(jax.tree.leaves(eqx.filter(base, eqx.is_array)))


def test_ff_ema_lineage_parity_and_shared_fresh_init():
    # Both lineages are grown with the same key (ff4 plan section 2), so the
    # train net and the EMA net share fresh-unit init, and each preserves
    # its own function.
    cfg_new = tiny_cfg(2, ff_factor=3)
    key = jrandom.PRNGKey(7)
    train_net = build_network(tiny_cfg(2, ff_factor=1), jrandom.PRNGKey(0))
    ema_net = build_network(tiny_cfg(2, ff_factor=1), jrandom.PRNGKey(1))
    g_train = grow_ff(train_net, cfg_new, key)
    g_ema = grow_ff(ema_net, cfg_new, key)
    assert_forward_parity(ema_net, g_ema, n_samples=4, atol=FF_ATOL)
    old_hidden = train_net.transformer_layers[0].ff_linear1.weight.shape[0]
    assert bool(jnp.array_equal(
        g_train.transformer_layers[0].ff_linear1.weight[old_hidden:],
        g_ema.transformer_layers[0].ff_linear1.weight[old_hidden:]))


def test_grow_ff_rejects_anything_but_a_wider_ffn():
    base = build_network(tiny_cfg(2, ff_factor=2), jrandom.PRNGKey(0))
    for bad in (tiny_cfg(2, ff_factor=2),          # not wider
                tiny_cfg(2, ff_factor=1),          # narrower
                tiny_cfg(3, ff_factor=3),          # depth changed too
                tiny_cfg(2, ff_factor=3, embed_dim=64),
                tiny_cfg(2, ff_factor=3, n_head=8)):
        with pytest.raises(ValueError):
            grow_ff(base, bad, jrandom.PRNGKey(1))


# ---- Parity gates ----

def _bump_pass_logit(net):
    """Raise the pass-action logit of every patch's (0, 0) offset. Pass is
    action channel 8 and is never masked, so the argmax must move."""
    idx = 8 * net.patch_size * net.patch_size
    return eqx.tree_at(
        lambda n: n.policy_head.bias, net,
        net.policy_head.bias.at[idx].add(1e6))


def test_parity_argmax_gate_rejects_independently_of_the_tolerance():
    # atol wide open, so only the argmax gate can reject this pair.
    base, grown = _grow_ff_pair()
    with pytest.raises(AssertionError, match="argmax agreement"):
        assert_forward_parity(base, _bump_pass_logit(grown), n_samples=4,
                              atol=1e9)


def test_parity_without_atol_demands_bitwise_equality():
    base, grown = _grow_ff_pair()
    with pytest.raises(AssertionError, match="bitwise equality is required"):
        assert_forward_parity(base, _bump_pass_logit(grown), n_samples=4)


def test_parity_reports_bitwise_on_an_identical_pair():
    base, _ = _grow_ff_pair()
    report = assert_forward_parity(base, base, n_samples=2, atol=FF_ATOL)
    assert report["bitwise"] and report["argmax_agreement"] == 1.0
    assert set(report["max_abs_diff"]) == {"logits", "value", "value_aux"}
