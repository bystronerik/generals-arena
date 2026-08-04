"""Joint board symmetry augmentation for tensors, masks, actions, and targets.

One dihedral transform remaps the observation tensor, legal mask, policy
target, memory, belief planes (inside the tensor), generals, and every spatial
auxiliary target together. Scalar margin and WDL targets do not move.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from training.morpheus.objective.targets import SeatTargets

REPO = Path(__file__).resolve().parents[3]
MORPHEUS_BOT = REPO / "bots" / "morpheus"

Array = np.ndarray


def _ensure_bot_path() -> None:
    for entry in (REPO, REPO / "bots", MORPHEUS_BOT):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


def _symmetry_api():
    _ensure_bot_path()
    from symmetry import (  # type: ignore
        all_symmetry_names,
        get_symmetry,
        inverse_symmetry,
        transform_memory,
        transform_plane,
        transform_policy,
        transform_tensor,
    )

    return (
        all_symmetry_names,
        get_symmetry,
        inverse_symmetry,
        transform_memory,
        transform_plane,
        transform_policy,
        transform_tensor,
    )


@dataclass(frozen=True)
class AugmentableSample:
    """One training row that symmetry must transform as a single unit."""

    tensor: Array  # (49, PAD, PAD)
    legal_mask: Array  # (N_ACTIONS,)
    targets: SeatTargets
    symmetry_name: str = "id"
    memory: Optional[object] = None  # VisibleMemory when present

    def with_symmetry(self, name: str) -> AugmentableSample:
        return apply_symmetry(self, name)


def transform_bin_plane(bins: Array, sym) -> Array:
    """Remap int16 army-bin indices; keep -1 ignore cells."""
    (
        _names,
        _get,
        _inv,
        _mem,
        transform_plane,
        _pol,
        _ten,
    ) = _symmetry_api()
    # transform_plane copies values cell-wise; works for int labels.
    return transform_plane(bins, sym)


def apply_symmetry(sample: AugmentableSample, name: str) -> AugmentableSample:
    """Apply one named dihedral transform to every spatial field together."""
    (
        _names,
        get_symmetry,
        _inv,
        transform_memory,
        transform_plane,
        transform_policy,
        transform_tensor,
    ) = _symmetry_api()
    sym = get_symmetry(name)
    t = sample.targets
    new_targets = SeatTargets(
        policy=transform_policy(t.policy.astype(np.float64), sym).astype(t.policy.dtype),
        wdl=np.asarray(t.wdl, dtype=t.wdl.dtype).copy(),
        value=float(t.value),
        hidden_owner=transform_plane(t.hidden_owner, sym),
        enemy_army_bin=transform_bin_plane(t.enemy_army_bin, sym),
        enemy_general=transform_plane(t.enemy_general, sym),
        hidden_castle=transform_plane(t.hidden_castle, sym),
        land_margin=float(t.land_margin),
        army_margin=float(t.army_margin),
        castle_margin=float(t.castle_margin),
        turns_to_termination=float(t.turns_to_termination),
        board_mask=transform_plane(t.board_mask, sym),
    )
    new_memory = None
    if sample.memory is not None:
        new_memory = transform_memory(sample.memory, sym)
    return AugmentableSample(
        tensor=transform_tensor(sample.tensor, sym),
        legal_mask=transform_policy(
            sample.legal_mask.astype(np.float64), sym
        ).astype(sample.legal_mask.dtype),
        targets=new_targets,
        symmetry_name=name if sample.symmetry_name == "id" else f"{sample.symmetry_name}+{name}",
        memory=new_memory,
    )


def invert_symmetry(sample: AugmentableSample) -> AugmentableSample:
    """Apply the inverse of ``sample.symmetry_name`` (single named transform)."""
    (_names, _get, inverse_symmetry, *_rest) = _symmetry_api()
    # Only invert a single applied name stored on the sample.
    name = sample.symmetry_name
    if "+" in name:
        raise ValueError(
            "invert_symmetry supports a single applied transform; "
            f"got composed name {name!r}"
        )
    inv = inverse_symmetry(name)
    restored = apply_symmetry(
        AugmentableSample(
            tensor=sample.tensor,
            legal_mask=sample.legal_mask,
            targets=sample.targets,
            symmetry_name="id",
            memory=sample.memory,
        ),
        inv.name,
    )
    return replace(restored, symmetry_name="id")


def all_named_symmetries() -> tuple[str, ...]:
    names, *_ = _symmetry_api()
    return names()


def round_trip_ok(sample: AugmentableSample, name: str, *, atol: float = 1e-5) -> bool:
    """True when transform then inverse restores tensor, mask, and targets."""
    mid = apply_symmetry(sample, name)
    back = invert_symmetry(mid)
    if not np.allclose(back.tensor, sample.tensor, atol=atol):
        return False
    if not np.allclose(back.legal_mask, sample.legal_mask, atol=atol):
        return False
    t0, t1 = sample.targets, back.targets
    if not np.allclose(t0.policy, t1.policy, atol=atol):
        return False
    if not np.allclose(t0.hidden_owner, t1.hidden_owner, atol=atol):
        return False
    if not np.array_equal(t0.enemy_army_bin, t1.enemy_army_bin):
        return False
    if not np.allclose(t0.enemy_general, t1.enemy_general, atol=atol):
        return False
    if not np.allclose(t0.hidden_castle, t1.hidden_castle, atol=atol):
        return False
    if not np.allclose(t0.board_mask, t1.board_mask, atol=atol):
        return False
    if abs(t0.value - t1.value) > atol:
        return False
    if not np.allclose(t0.wdl, t1.wdl, atol=atol):
        return False
    return True
