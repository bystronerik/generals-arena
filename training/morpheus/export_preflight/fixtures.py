"""Fixed random tensors for root / leaf / enemy-proposal batch shapes."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from training.morpheus.export_preflight.model import BOARD, IN_CHANNELS

# From docs/bots/morpheus/network.md and runtime.md.
BATCH_ROOT = 1
BATCH_LEAF = 4
BATCH_ENEMY_PROPOSAL = 64
BATCH_SHAPES = (BATCH_ROOT, BATCH_LEAF, BATCH_ENEMY_PROPOSAL)


@dataclass(frozen=True)
class TensorFixture:
    name: str
    batch: int
    tensor: torch.Tensor


def make_input(batch: int, *, seed: int) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(batch, IN_CHANNELS, BOARD, BOARD, generator=gen)


def all_batch_fixtures(*, seed: int = 0) -> list[TensorFixture]:
    names = {
        BATCH_ROOT: "root",
        BATCH_LEAF: "leaf",
        BATCH_ENEMY_PROPOSAL: "enemy_proposal",
    }
    fixtures: list[TensorFixture] = []
    for i, batch in enumerate(BATCH_SHAPES):
        fixtures.append(
            TensorFixture(
                name=names[batch],
                batch=batch,
                tensor=make_input(batch, seed=seed + 1000 + i),
            )
        )
    return fixtures
