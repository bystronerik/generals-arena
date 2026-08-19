"""Function-preserving growth surgery for the HistoryTransformer.

Two operations, both checkpoint-in / checkpoint-out (no training-loop
imports).

``grow_depth`` — docs/research/strategies/joe-depth7-growth-plan.md,
section 2. The blocks are pre-norm residual, so a block whose
``attn.out_proj`` and ``ff_linear2`` weights and biases are zero is an
exact identity: the zeroed sublayer outputs are exact zeros even under
bf16 (``x + 0 == x`` bitwise), so parity holds with the production
``use_bf16`` setting.

``grow_ff`` — docs/research/strategies/joe-ff4-growth-plan.md, section 2.
Every block gains fresh ``ff_linear1`` rows (the new hidden units) and an
equal count of **zeroed** ``ff_linear2`` columns, so the new units add
exactly zero to the block output while their gradients are nonzero from
the first step.

``assert_forward_parity`` verifies the preservation claim instead of
assuming it. It demands bitwise equality by default; the FF graft passes
an ``atol`` because it changes the FFN GEMM shapes (ff4 plan section 2).
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


def _check_trunk_compatible(base_net, template, changes):
    """Static fields, everything outside ``transformer_layers``, and the
    head count must agree: a growth op edits the stack only, and each op
    edits one dimension of it (``changes`` names which)."""
    for f in dataclasses.fields(type(base_net)):
        if not f.metadata.get("static", False):
            continue
        a, b = getattr(base_net, f.name), getattr(template, f.name)
        if a != b:
            raise ValueError(
                f"static field {f.name} differs: base {a!r}, target {b!r}; "
                f"growth changes {changes} only")
    shared = _shared_field_names(type(base_net))
    base_shared = [getattr(base_net, n) for n in shared]
    tmpl_shared = [getattr(template, n) for n in shared]
    if _leaf_shapes(base_shared) != _leaf_shapes(tmpl_shared):
        raise ValueError(
            "base and target nets disagree outside transformer_layers; "
            f"growth changes {changes} only")
    base_l0, tmpl_l0 = base_net.transformer_layers[0], template.transformer_layers[0]
    if base_l0.attn.n_head != tmpl_l0.attn.n_head:
        raise ValueError(
            f"n_head differs: base {base_l0.attn.n_head}, target "
            f"{tmpl_l0.attn.n_head}; growth changes {changes} only")


def _check_compatible(base_net, template):
    """The target config may differ from the base in depth only."""
    _check_trunk_compatible(base_net, template, "depth")
    base_l0, tmpl_l0 = base_net.transformer_layers[0], template.transformer_layers[0]
    if _leaf_shapes(base_l0) != _leaf_shapes(tmpl_l0):
        raise ValueError(
            "per-block shapes differ (embed_dim or ff_factor changed); "
            "growth changes depth only")


def _check_ff_compatible(base_net, template):
    """The target config may differ from the base in ``ff_factor`` only:
    same depth, same embed_dim, same attention and norms, wider FFN."""
    _check_trunk_compatible(base_net, template, "the FFN width")
    old_depth = len(base_net.transformer_layers)
    new_depth = len(template.transformer_layers)
    if old_depth != new_depth:
        raise ValueError(
            f"depth differs: base {old_depth}, target {new_depth}; growth "
            "changes the FFN width only")
    for i, (b, t) in enumerate(zip(base_net.transformer_layers,
                                   template.transformer_layers)):
        if _leaf_shapes((b.norm1, b.attn, b.norm2)) != \
                _leaf_shapes((t.norm1, t.attn, t.norm2)):
            raise ValueError(
                f"block {i}: attention or norm shapes differ; growth "
                "changes the FFN width only")
        old_hidden, d_model = b.ff_linear1.weight.shape
        new_hidden, tmpl_d_model = t.ff_linear1.weight.shape
        if (d_model, b.ff_linear2.weight.shape[0]) != \
                (tmpl_d_model, t.ff_linear2.weight.shape[0]):
            raise ValueError(
                f"block {i}: embed_dim differs (base {d_model}, target "
                f"{tmpl_d_model}); growth changes the FFN width only")
        if new_hidden <= old_hidden:
            raise ValueError(
                f"block {i}: cannot grow the FFN width {old_hidden} -> "
                f"{new_hidden}: the target width must exceed the base width")


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


def _grow_block_ff(base_block, tmpl_block):
    """Widen one block's FFN, keeping its function exactly.

    ``ff_linear1`` (weight rows + bias entries) gains the template's extra
    rows — fresh random init, so the new hidden units carry no
    duplicate-neuron symmetry. ``ff_linear2`` gains the same count of
    zeroed weight columns and keeps its bias, so the new units add exactly
    zero to the block output. Both new Linears are rebuilt from the
    template modules, so their static ``in_features``/``out_features``
    describe the grown shapes.
    """
    old_hidden = base_block.ff_linear1.weight.shape[0]
    new_hidden = tmpl_block.ff_linear1.weight.shape[0]
    d_model = base_block.ff_linear2.weight.shape[0]
    w1 = jnp.concatenate(
        [base_block.ff_linear1.weight,
         tmpl_block.ff_linear1.weight[old_hidden:]], axis=0)
    b1 = jnp.concatenate(
        [base_block.ff_linear1.bias,
         tmpl_block.ff_linear1.bias[old_hidden:]], axis=0)
    w2 = jnp.concatenate(
        [base_block.ff_linear2.weight,
         jnp.zeros((d_model, new_hidden - old_hidden),
                   base_block.ff_linear2.weight.dtype)], axis=1)
    ff1 = eqx.tree_at(lambda l: (l.weight, l.bias), tmpl_block.ff_linear1,
                      (w1, b1))
    ff2 = eqx.tree_at(lambda l: (l.weight, l.bias), tmpl_block.ff_linear2,
                      (w2, base_block.ff_linear2.bias))
    return eqx.tree_at(lambda b: (b.ff_linear1, b.ff_linear2), base_block,
                       (ff1, ff2))


def grow_ff(base_net, cfg_new, key):
    """Widen every block's FFN in ``base_net`` to ``cfg_new.ff_factor``.

    Unlike ``grow_depth``, which adds self-contained blocks, this edits
    **every existing layer** (ff4 plan section 2). The fresh hidden units
    come from a ``build_network(cfg_new, key)`` template, one independent
    draw per block; everything outside the two FFN Linears — attention,
    norms, embedder, heads, temporal encoder, ``pos_encoding`` — is copied
    from ``base_net`` unchanged. Token count and embed width do not
    change. The same ``key`` grows both lineages so the train net and the
    EMA net share fresh-unit init.
    """
    template = build_network(cfg_new, key)
    _check_ff_compatible(base_net, template)
    grown_layers = [_grow_block_ff(b, t)
                    for b, t in zip(base_net.transformer_layers,
                                    template.transformer_layers)]

    shared = _shared_field_names(type(base_net))
    grown = eqx.tree_at(
        lambda n: [getattr(n, name) for name in shared],
        template,
        [getattr(base_net, name) for name in shared])
    return eqx.tree_at(lambda n: n.transformer_layers, grown, grown_layers)


def _sample_inputs(net, key, n):
    kobs, kmove, kbuild, ktemp = jax.random.split(key, 4)
    p = net.pad_to
    obs = jax.random.uniform(kobs, (n, net.n_channels, p, p))
    move = (jax.random.uniform(kmove, (n, p, p, 4)) > 0.5).astype(jnp.float32)
    build = jax.random.uniform(kbuild, (n, p, p)) > 0.5
    temporal = jax.random.uniform(ktemp, (n, 2, net.temporal_window)) * 100.0
    return obs, move, build, temporal


def _forward_batch(net, obs, move, build, temporal):
    # Deliberately NOT jitted: XLA fuses different-shape programs
    # differently, which can shift bf16 rounding in the shared layers by an
    # ulp even though the surgery is exact. Per-op (eager, vmapped)
    # execution runs the identical primitive sequence for the shared parts
    # in both nets, so the comparison isolates the surgery itself.
    return jax.vmap(
        lambda o, m, b, t: net._forward(o, m, b, t))(obs, move, build, temporal)


_OUTPUT_NAMES = ("logits", "value", "value_aux")


def assert_forward_parity(old, new, n_samples=64, *, key=None, atol=None):
    """Forward-equality check on random masked inputs.

    ``atol=None`` (the depth graft, depth plan section 6) demands bitwise
    equality of logits, value, and value_aux on every sample.

    A float ``atol`` (the FF graft, ff4 plan section 2) still tries bitwise
    first and, only on failure, accepts a max |Δ| at or below ``atol`` on
    every output **plus** an identical argmax action on every sample. The
    FF graft changes the FFN GEMM shapes, so a different tiling of the
    *nonzero* partial sums can move a result by an ulp even eager; a real
    wiring bug still fails loudly, on argmax or on a Δ far above epsilon.

    Runs with whatever ``use_bf16`` the nets carry. Returns the report the
    plan's run log wants: the bitwise flag, per-output max |Δ|, and argmax
    agreement.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    inputs = _sample_inputs(old, key, n_samples)
    old_out = _forward_batch(old, *inputs)
    new_out = _forward_batch(new, *inputs)

    bitwise = all(bool(jnp.array_equal(a, b))
                  for a, b in zip(old_out, new_out))
    diffs = {name: float(jnp.max(jnp.abs(
                 a.astype(jnp.float32) - b.astype(jnp.float32))))
             for name, a, b in zip(_OUTPUT_NAMES, old_out, new_out)}
    agreement = float(jnp.mean(jnp.argmax(old_out[0], axis=-1)
                               == jnp.argmax(new_out[0], axis=-1)))
    report = {"n_samples": n_samples, "bitwise": bitwise,
              "max_abs_diff": diffs, "argmax_agreement": agreement}
    if bitwise:
        return report

    summary = ", ".join(f"{name} {diffs[name]:.3g}" for name in _OUTPUT_NAMES)
    if atol is None:
        raise AssertionError(
            f"forward parity failed over {n_samples} samples: max abs diff "
            f"{summary}; bitwise equality is required for this graft")
    over = [name for name in _OUTPUT_NAMES if not diffs[name] <= atol]
    if over or agreement < 1.0:
        raise AssertionError(
            f"forward parity failed over {n_samples} samples: max abs diff "
            f"{summary} (atol {atol:g}, over on {over or 'none'}), argmax "
            f"agreement {agreement:.4f}")
    return report
