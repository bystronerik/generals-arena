"""Function-preserving depth growth for the HistoryTransformer.

docs/research/strategies/joe-depth7-growth-plan.md, section 2: the blocks
are pre-norm residual, so a block whose ``attn.out_proj`` and
``ff_linear2`` weights and biases are zero is an exact identity — the
zeroed sublayer outputs are exact zeros even under bf16 (``x + 0 == x``
bitwise), so parity holds with the production ``use_bf16`` setting.

``grow_depth`` splices fresh identity blocks into a trained net;
``assert_forward_parity`` verifies the bit-identical claim instead of
assuming it. No training-loop imports: the surgery is checkpoint-in,
checkpoint-out.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from training.joe.networks import build_network


def splice_layout(old_depth: int, new_depth: int) -> list:
    """Layout of the grown stack: base layer index, or None for a fresh block.

    The k fresh blocks split the base stack into k + 1 groups, as even as
    possible with the larger groups first — depth 5 -> 7 gives
    ``[L0, L1, NEW, L2, L3, NEW, L4]`` (plan section 2).
    """
    k = new_depth - old_depth
    if old_depth <= 0 or k <= 0:
        raise ValueError(
            f"cannot grow depth {old_depth} -> {new_depth}: the target "
            "depth must exceed the base depth")
    group = old_depth // (k + 1)
    rem = old_depth % (k + 1)
    sizes = [group + (1 if g < rem else 0) for g in range(k + 1)]
    layout, next_old = [], 0
    for g, size in enumerate(sizes):
        layout.extend(range(next_old, next_old + size))
        next_old += size
        if g < k:
            layout.append(None)
    return layout


def _identity_block(block):
    """Zero ``attn.out_proj`` and ``ff_linear2`` (weights and biases) so the
    pre-norm block is an exact identity; ``norm1/norm2/q/k/v/ff_linear1``
    keep their fresh init and receive gradients as soon as the zeroed
    layers move."""
    return eqx.tree_at(
        lambda b: (b.attn.out_proj.weight, b.attn.out_proj.bias,
                   b.ff_linear2.weight, b.ff_linear2.bias),
        block,
        replace_fn=jnp.zeros_like)


def _shared_field_names(cls) -> list:
    """Non-static fields copied from the base net (everything outside the
    transformer stack: embedder, value token, positional encoding,
    temporal encoder + type embed, norm_out, both heads, bin centers)."""
    return [f.name for f in dataclasses.fields(cls)
            if not f.metadata.get("static", False)
            and f.name != "transformer_layers"]


def _leaf_shapes(tree):
    return [(x.shape, x.dtype)
            for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def _check_compatible(base_net, template):
    """The target config may differ from the base in depth only."""
    for f in dataclasses.fields(type(base_net)):
        if not f.metadata.get("static", False):
            continue
        a, b = getattr(base_net, f.name), getattr(template, f.name)
        if a != b:
            raise ValueError(
                f"static field {f.name} differs: base {a!r}, target {b!r}; "
                "growth changes depth only")
    shared = _shared_field_names(type(base_net))
    base_shared = [getattr(base_net, n) for n in shared]
    tmpl_shared = [getattr(template, n) for n in shared]
    if _leaf_shapes(base_shared) != _leaf_shapes(tmpl_shared):
        raise ValueError(
            "base and target nets disagree outside transformer_layers; "
            "growth changes depth only")
    base_l0, tmpl_l0 = base_net.transformer_layers[0], template.transformer_layers[0]
    if base_l0.attn.n_head != tmpl_l0.attn.n_head:
        raise ValueError(
            f"n_head differs: base {base_l0.attn.n_head}, target "
            f"{tmpl_l0.attn.n_head}; growth changes depth only")
    if _leaf_shapes(base_l0) != _leaf_shapes(tmpl_l0):
        raise ValueError(
            "per-block shapes differ (embed_dim or ff_factor changed); "
            "growth changes depth only")


def grow_depth(base_net, cfg_new, key):
    """Splice identity blocks into ``base_net`` to reach ``cfg_new.depth``.

    Fresh blocks come from a ``build_network(cfg_new, key)`` template and
    are zeroed into exact identities; everything outside
    ``transformer_layers`` is copied from ``base_net`` unchanged (token
    count does not change, so ``pos_encoding`` needs no surgery). The same
    ``key`` grows both lineages so the train net and the EMA net share
    fresh-block init.
    """
    template = build_network(cfg_new, key)
    old_layers = list(base_net.transformer_layers)
    new_layers = list(template.transformer_layers)
    layout = splice_layout(len(old_layers), len(new_layers))
    _check_compatible(base_net, template)

    spliced = [old_layers[i] if i is not None
               else _identity_block(new_layers[pos])
               for pos, i in enumerate(layout)]

    shared = _shared_field_names(type(base_net))
    grown = eqx.tree_at(
        lambda n: [getattr(n, name) for name in shared],
        template,
        [getattr(base_net, name) for name in shared])
    return eqx.tree_at(lambda n: n.transformer_layers, grown, spliced)


def _sample_inputs(net, key, n):
    kobs, kmove, kbuild, ktemp = jax.random.split(key, 4)
    p = net.pad_to
    obs = jax.random.uniform(kobs, (n, net.n_channels, p, p))
    move = (jax.random.uniform(kmove, (n, p, p, 4)) > 0.5).astype(jnp.float32)
    build = jax.random.uniform(kbuild, (n, p, p)) > 0.5
    temporal = jax.random.uniform(ktemp, (n, 2, net.temporal_window)) * 100.0
    return obs, move, build, temporal


def _forward_batch(net, obs, move, build, temporal):
    # Deliberately NOT jitted: XLA fuses different-depth programs
    # differently, which can shift bf16 rounding in the shared layers by an
    # ulp even though the surgery is exact. Per-op (eager, vmapped)
    # execution runs the identical primitive sequence for the shared layers
    # in both nets, so the comparison isolates the surgery itself.
    return jax.vmap(
        lambda o, m, b, t: net._forward(o, m, b, t))(obs, move, build, temporal)


def assert_forward_parity(old, new, n_samples=64, *, key=None):
    """Exact-equality forward check on random masked inputs (plan section 6).

    Raises AssertionError unless logits, value, and value_aux agree exactly
    on every sample. Runs with whatever ``use_bf16`` the nets carry.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    inputs = _sample_inputs(old, key, n_samples)
    old_out = _forward_batch(old, *inputs)
    new_out = _forward_batch(new, *inputs)
    for name, a, b in zip(("logits", "value", "value_aux"), old_out, new_out):
        if not bool(jnp.array_equal(a, b)):
            diff = float(jnp.max(jnp.abs(
                a.astype(jnp.float32) - b.astype(jnp.float32))))
            raise AssertionError(
                f"forward parity failed on {name}: max abs diff {diff:.3g} "
                f"over {n_samples} samples")
