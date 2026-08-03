"""Part 04 — Morpheus network heads, perspective sign, and symmetry utilities."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.morpheus

from _common.wire import Observation
from action import N_ACTIONS, PASS_INDEX, encode_action, legal_mask
from memory import empty_memory
from network import (
    BOARD,
    IN_CHANNELS,
    N_ARMY_BINS,
    N_BLOCKS,
    backup_value,
    legal_normalized_policy,
    make_model,
    parameter_count,
    policy_symmetry_equivariant,
    transform_policy_spatial,
    transform_spatial_head,
    wdl_logits_swap_seats,
    wdl_value,
)
from schema import ARMY_BIN_EDGES, N_ARMY_BINS as SCHEMA_BINS
from symmetry import all_symmetry_names, get_symmetry, inverse_symmetry
from tensor import N_PLANES, observation_tensor


def _masked_random_tensor(H: int, W: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tensor = rng.standard_normal((N_PLANES, BOARD, BOARD), dtype=np.float32)
    mask = np.zeros((BOARD, BOARD), dtype=np.float32)
    mask[:H, :W] = 1.0
    tensor[0] = mask
    tensor *= mask
    return tensor


def _obs_hw(H: int, W: int, turn: int = 10) -> Observation:
    types = np.ones((H, W), dtype=np.int32)
    owners = np.zeros((H, W), dtype=np.int32)
    armies = np.zeros((H, W), dtype=np.int32)
    types[0, 0] = 4
    owners[0, 0] = 1
    armies[0, 0] = 10
    types[H - 1, W - 1] = 1
    owners[H - 1, W - 1] = 1
    armies[H - 1, W - 1] = 5
    return Observation(
        H=H,
        W=W,
        turn=turn,
        my_land=2,
        my_army=15,
        opp_land=1,
        opp_army=5,
        type_grid=types.tolist(),
        owner_grid=owners.tolist(),
        army_grid=armies.tolist(),
    )


@pytest.mark.parametrize("hw", [(18, 18), (19, 20), (20, 19), (21, 21)])
def test_head_shapes_on_padded_rectangles(hw: tuple[int, int]):
    H, W = hw
    model = make_model(seed=0, n_blocks=3)
    x = torch.from_numpy(_masked_random_tensor(H, W, seed=H * 100 + W)).unsqueeze(0)
    out = model(x)
    assert out.policy.shape == (1, 9, BOARD, BOARD)
    assert out.pass_logit.shape == (1, 1)
    assert out.wdl_logits.shape == (1, 3)
    assert out.hidden_owner.shape == (1, 1, BOARD, BOARD)
    assert out.enemy_army_bins.shape == (1, N_ARMY_BINS, BOARD, BOARD)
    assert out.enemy_general.shape == (1, 1, BOARD, BOARD)
    assert out.hidden_castle.shape == (1, 1, BOARD, BOARD)
    assert out.land_margin.shape == (1, 1)
    assert out.army_margin.shape == (1, 1)
    assert out.castle_margin.shape == (1, 1)
    assert out.turns_to_termination.shape == (1, 1)


def test_parameter_budget_near_initial_guess():
    model = make_model(seed=0, n_blocks=N_BLOCKS)
    count = parameter_count(model)
    assert 200_000 <= count <= 450_000


def test_army_bin_edges_versioned():
    assert len(ARMY_BIN_EDGES) == SCHEMA_BINS + 1
    assert ARMY_BIN_EDGES[0] == 0.0
    assert ARMY_BIN_EDGES[-1] == 4096.0
    assert all(ARMY_BIN_EDGES[i] < ARMY_BIN_EDGES[i + 1] for i in range(len(ARMY_BIN_EDGES) - 1))


def test_wdl_perspective_sign_convention():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.1, -0.2]])
    flipped = wdl_logits_swap_seats(logits)
    assert torch.allclose(wdl_value(logits), -wdl_value(flipped))
    root_v = backup_value(logits, from_root=True)
    enemy_v = backup_value(logits, from_root=False)
    assert torch.allclose(root_v, -enemy_v)


def test_legal_policy_normalization():
    obs = _obs_hw(18, 18)
    mem = empty_memory(18, 18)
    tensor, _ = observation_tensor(obs, mem)
    model = make_model(seed=1, n_blocks=3)
    x = torch.from_numpy(tensor).unsqueeze(0)
    mask_np = legal_mask(obs, mem)
    mask = torch.from_numpy(mask_np)
    with torch.no_grad():
        out = model(x)
        dist = legal_normalized_policy(out.policy, out.pass_logit, mask)
    assert dist.shape == (1, N_ACTIONS)
    assert torch.allclose(dist.sum(dim=1), torch.ones(1))
    illegal = ~mask
    assert torch.all(dist[0, illegal] == 0.0)
    assert dist[0, PASS_INDEX] > 0.0


def test_policy_symmetry_round_trip_on_logits():
    obs = _obs_hw(21, 21)
    mem = empty_memory(21, 21)
    tensor, _ = observation_tensor(obs, mem, previous_action=(0, 0, 0, 3, 0))
    model = make_model(seed=2, n_blocks=3)
    with torch.no_grad():
        policy = model(torch.from_numpy(tensor).unsqueeze(0)).policy
    for name in all_symmetry_names():
        sym = get_symmetry(name)
        inv = inverse_symmetry(name)
        p1 = transform_policy_spatial(policy, sym.name)
        p2 = transform_policy_spatial(p1, inv.name)
        assert torch.allclose(p2, policy, atol=1e-5, rtol=1e-5)


def test_auxiliary_spatial_symmetry_round_trip():
    model = make_model(seed=3, n_blocks=3)
    x = torch.from_numpy(_masked_random_tensor(21, 21, 99)).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    for head in (out.hidden_owner, out.enemy_army_bins, out.enemy_general, out.hidden_castle):
        for name in all_symmetry_names():
            sym = get_symmetry(name)
            inv = inverse_symmetry(name)
            h1 = transform_spatial_head(head, sym.name)
            h2 = transform_spatial_head(h1, inv.name)
            assert torch.allclose(h2, head, atol=1e-5, rtol=1e-5)


def test_policy_spatial_matches_action_codec_transform():
    obs = _obs_hw(21, 21)
    mem = empty_memory(21, 21)
    tensor, _ = observation_tensor(obs, mem)
    model = make_model(seed=4, n_blocks=3)
    with torch.no_grad():
        policy = model(torch.from_numpy(tensor).unsqueeze(0)).policy[0]
    flat = policy.reshape(-1).cpu().numpy()
    logits = np.zeros(N_ACTIONS, dtype=np.float32)
    logits[: flat.shape[0]] = flat
    from action import decode_action
    from symmetry import transform_action_tuple

    for sym_name in all_symmetry_names():
        sym = get_symmetry(sym_name)
        remapped = np.zeros_like(logits)
        for idx in range(PASS_INDEX):
            new_action = transform_action_tuple(decode_action(idx), sym)
            remapped[encode_action(new_action)] = logits[idx]
        torch_remapped = transform_policy_spatial(policy.unsqueeze(0), sym_name)[0]
        assert np.allclose(
            torch_remapped.reshape(-1).cpu().numpy(),
            remapped[: 9 * BOARD * BOARD],
        )


def test_identity_policy_equivariance():
    """Identity is the only architectural free lunch; D4 comes from training augments."""
    obs = _obs_hw(21, 21)
    mem = empty_memory(21, 21)
    tensor, _ = observation_tensor(obs, mem)
    model = make_model(seed=5, n_blocks=3)
    assert policy_symmetry_equivariant(model, tensor, "id")
