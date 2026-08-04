"""Part 12 — joint symmetry augmentation round-trips."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from training.morpheus.objective.augmentation import (
    AugmentableSample,
    all_named_symmetries,
    apply_symmetry,
    invert_symmetry,
    round_trip_ok,
)
from training.morpheus.objective.targets import SeatTargets, sparse_policy_to_dense
from training.morpheus.self_play.schema import SparsePolicy


def _sample() -> AugmentableSample:
    pad = 21
    board = np.zeros((pad, pad), dtype=np.float32)
    board[:5, :5] = 1.0
    hidden_owner = np.zeros((pad, pad), dtype=np.float32)
    hidden_owner[1, 2] = 1.0
    hidden_owner[2, 3] = 1.0
    enemy_general = np.zeros((pad, pad), dtype=np.float32)
    enemy_general[1, 2] = 1.0
    hidden_castle = np.zeros((pad, pad), dtype=np.float32)
    hidden_castle[2, 3] = 1.0
    bins = np.full((pad, pad), -1, dtype=np.int16)
    bins[1, 2] = 4
    bins[2, 3] = 7
    policy = sparse_policy_to_dense(
        SparsePolicy(
            indices=(0, 21, 42, 3969),
            probs=(0.1, 0.2, 0.3, 0.4),
        )
    )
    legal = (policy > 0).astype(np.float32)
    legal[3969] = 1.0
    tensor = np.zeros((49, pad, pad), dtype=np.float32)
    tensor[0] = board
    tensor[7, 0, 0] = 1.0  # own general mark
    tensor[22] = hidden_owner  # belief_enemy_owner plane index
    targets = SeatTargets(
        policy=policy,
        wdl=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        value=1.0,
        hidden_owner=hidden_owner,
        enemy_army_bin=bins,
        enemy_general=enemy_general,
        hidden_castle=hidden_castle,
        land_margin=0.25,
        army_margin=-0.1,
        castle_margin=0.0,
        turns_to_termination=100.0,
        board_mask=board,
    )
    # Canonicalize coordinate planes the same way transform_tensor does.
    return apply_symmetry(
        AugmentableSample(tensor=tensor, legal_mask=legal, targets=targets),
        "id",
    )


def test_eight_symmetries_named():
    names = all_named_symmetries()
    assert len(names) == 8
    assert "id" in names and "rot90" in names


def test_every_symmetry_round_trips():
    sample = _sample()
    for name in all_named_symmetries():
        assert round_trip_ok(sample, name), name


def test_scalars_and_wdl_unchanged_under_symmetry():
    sample = _sample()
    mid = apply_symmetry(sample, "rot90")
    assert mid.targets.value == sample.targets.value
    np.testing.assert_array_equal(mid.targets.wdl, sample.targets.wdl)
    assert mid.targets.land_margin == sample.targets.land_margin
    assert mid.targets.army_margin == sample.targets.army_margin
    assert mid.targets.castle_margin == sample.targets.castle_margin
    assert mid.targets.turns_to_termination == sample.targets.turns_to_termination


def test_spatial_targets_move_together():
    sample = _sample()
    mid = apply_symmetry(sample, "rot90")
    # Original enemy general at (1,2); rot90 (r,c)->(c, 20-r) => (2, 19)
    assert mid.targets.enemy_general[2, 19] == 1.0
    assert mid.targets.hidden_owner[2, 19] == 1.0
    assert mid.targets.enemy_army_bin[2, 19] == 4
    # Pass logit fixed
    assert mid.targets.policy[3969] == pytest.approx(sample.targets.policy[3969])


def test_invert_restores_policy_mass():
    sample = _sample()
    mid = apply_symmetry(sample, "flip_h")
    back = invert_symmetry(mid)
    np.testing.assert_allclose(back.targets.policy, sample.targets.policy, atol=1e-9)
    assert abs(back.targets.policy.sum() - 1.0) < 1e-9
