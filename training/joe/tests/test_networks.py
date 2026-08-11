"""Phase 2 network port — action codec, masks, augmentation, tier sizes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.joe

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from training.joe.networks import (
    N_CHANNELS,
    HistoryTransformer,
    augment_obs,
    build_cost_from_obs,
    compute_build_mask,
    decode_action,
    encode_action,
    init_obs_state,
    prepare_action_mask,
)

P = 21


def test_action_round_trip_all_indices():
    idx = jnp.arange(10 * P * P)
    actions = jax.vmap(lambda i: decode_action(i, P))(idx)
    back = jax.vmap(lambda a: encode_action(a, P))(actions)
    assert bool(jnp.all(back == idx))


def test_decode_channel_semantics():
    gc = P * P
    pos = 4 * P + 7  # r=4, c=7
    full = decode_action(jnp.int32(2 * gc + pos), P)
    assert full.tolist() == [0, 4, 7, 2, 0]
    half = decode_action(jnp.int32((4 + 2) * gc + pos), P)
    assert half.tolist() == [0, 4, 7, 2, 1]
    pass_a = decode_action(jnp.int32(8 * gc + pos), P)
    assert pass_a.tolist() == [1, 4, 7, 0, 0]
    build = decode_action(jnp.int32(9 * gc + pos), P)
    # pass-field 2 is BUILD in generals/modifiers/build_castles.py
    assert build.tolist() == [2, 4, 7, 0, 0]


def test_prepare_action_mask_channels():
    key = jrandom.PRNGKey(0)
    k1, k2 = jrandom.split(key)
    move = (jrandom.uniform(k1, (P, P, 4)) > 0.5).astype(jnp.float32)
    build = jrandom.uniform(k2, (P, P)) > 0.5
    pen = prepare_action_mask(move, build, P)
    assert pen.shape == (10, P, P)
    # full and half channels share the move mask
    assert bool(jnp.all(pen[:4] == pen[4:8]))
    assert bool(jnp.all((pen[:4] == 0) == (jnp.transpose(move, (2, 0, 1)) == 1)))
    assert bool(jnp.all(pen[8] == 0.0))  # pass always allowed by default
    assert bool(jnp.all((pen[9] == 0) == build))
    pen_np = prepare_action_mask(move, build, P, allow_pass=False)
    assert bool(jnp.all(pen_np[8] == -1e9))


def test_prepare_action_mask_pads_invalid():
    h, w = 18, 20
    move = jnp.ones((h, w, 4))
    build = jnp.ones((h, w), dtype=bool)
    pen = prepare_action_mask(move, build, P)
    # Move and build channels are invalid in the padding; the pass channel
    # stays open everywhere (any pass index is the same pass).
    assert bool(jnp.all(pen[:8, h:, :] == -1e9))
    assert bool(jnp.all(pen[:8, :, w:] == -1e9))
    assert bool(jnp.all(pen[9, h:, :] == -1e9))
    assert bool(jnp.all(pen[9, :, w:] == -1e9))
    assert bool(jnp.all(pen[8] == 0))
    assert bool(jnp.all(pen[:4, :h, :w] == 0))


def test_augment_obs_shape_and_build_cost_channel():
    h, w = 18, 20  # true competition board, smaller than pad_to
    raw = jnp.zeros((14, h, w)).at[5].set(1.0)  # own everything, keep it simple
    cost = jnp.full((h, w), 41.0)
    st = init_obs_state(P)
    aug, st2 = augment_obs(raw, cost, st)
    assert aug.shape == (N_CHANNELS, P, P)
    assert N_CHANNELS == 39
    # Channel 24 carries cost / 50, zero in the padding
    assert bool(jnp.allclose(aug[24, :h, :w], 41.0 / 50.0))
    assert bool(jnp.all(aug[24, h:, :] == 0.0))
    # Padding beyond the true board reads as structure (mountain or in-fog)
    assert bool(jnp.all((aug[8, h:, :] + aug[13, h:, :]) == 1.0))


def test_build_cost_from_obs_surcharge_kernel():
    z = jnp.zeros((P, P), dtype=bool)
    obs = SimpleNamespace(
        armies=jnp.zeros((P, P)),
        generals=z.at[10, 10].set(True),
        castles=z,
        owned_cells=jnp.ones((P, P), dtype=bool),
    )
    cost = build_cost_from_obs(obs)
    assert int(cost[10, 10]) == 35 + 14      # d=0
    assert int(cost[10, 11]) == 35 + 12      # d=1
    assert int(cost[10, 14]) == 35 + 14 - 2 * 4
    assert int(cost[10, 17]) == 35           # d=7, beyond the kernel
    assert int(cost[0, 0]) == 35


def test_compute_build_mask_engine_validity():
    z = jnp.zeros((3, 3), dtype=bool)
    obs = SimpleNamespace(
        armies=jnp.array([[35., 34., 0.], [50., 50., 50.], [0., 0., 0.]]),
        generals=z.at[1, 0].set(True),
        castles=z.at[1, 1].set(True),
        owned_cells=jnp.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
    )
    cost = jnp.full((3, 3), 35)
    mask = compute_build_mask(obs, cost)
    # armies >= cost on a plain own cell
    assert bool(mask[0, 0]) and not bool(mask[0, 1]) and not bool(mask[0, 2])
    # generals and castles are not plain
    assert not bool(mask[1, 0]) and not bool(mask[1, 1]) and bool(mask[1, 2])
    # unowned cells never build
    assert not bool(mask[2, 0])


def test_tier_parameter_counts():
    # Plan section 2: S ~5M, M ~8M at 39 channels / 10 actions / patch 3.
    tiers = {
        "S": (dict(depth=4, embed_dim=352, n_head=8, ff_factor=2), 5_087_930),
        "M": (dict(depth=5, embed_dim=384, n_head=8, ff_factor=3), 8_556_250),
    }
    for _tier, (spec, expected) in tiers.items():
        net = HistoryTransformer(grid_size=P, pad_to=P, patch_size=3, **spec,
                                 key=jrandom.PRNGKey(0))
        n = sum(x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_array)))
        assert n == expected


def test_forward_respects_masks():
    net = HistoryTransformer(grid_size=P, pad_to=P, patch_size=3,
                             depth=1, embed_dim=64, n_head=4, ff_factor=1,
                             key=jrandom.PRNGKey(0))
    obs = jrandom.uniform(jrandom.PRNGKey(1), (N_CHANNELS, P, P))
    td = jnp.zeros((2, 512))
    # Only one valid action in the whole space: build at (3, 4)
    move = jnp.zeros((P, P, 4))
    build = jnp.zeros((P, P), dtype=bool).at[3, 4].set(True)
    action, value, logprob, entropy, value_aux, p_dist = net(
        obs, move, build, td, jrandom.PRNGKey(2), allow_pass=False)
    assert action.tolist() == [2, 3, 4, 0, 0]
    assert p_dist.shape == (10 * P * P,)
    assert float(p_dist[9 * P * P + 3 * P + 4]) == pytest.approx(1.0)
    assert value_aux.shape == (128,)
    assert -1.0 <= float(value) <= 1.0
