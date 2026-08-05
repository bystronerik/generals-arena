"""Thin pilot trainer — one backward step and checkpoint reload."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.morpheus

from training.morpheus.curriculum.definitions import CLASS_TACTICAL
from training.morpheus.curriculum.schema import CurriculumItem
from training.morpheus.objective.config import load_pilot_objective_bundle
from training.morpheus.objective.targets import build_seat_targets
from training.morpheus.pilot.batch import PilotSample
from training.morpheus.pilot.checkpoint import load_pilot_checkpoint, save_pilot_checkpoint
from training.morpheus.pilot.step import pilot_train_step
from training.morpheus.self_play.schema import SparsePolicy

REPO = Path(__file__).resolve().parents[3]
OBJECTIVE = REPO / "training/morpheus/configs/pilot-objective.json"


def _tiny_sample() -> PilotSample:
    H = W = 4
    ownership = np.zeros((2, H, W), dtype=bool)
    ownership[0, 0, 0] = True
    ownership[1, 3, 3] = True
    armies = np.zeros((H, W), dtype=np.int32)
    armies[0, 0] = 5
    armies[3, 3] = 8
    generals = np.zeros((H, W), dtype=bool)
    generals[0, 0] = True
    generals[3, 3] = True
    castles = np.zeros((H, W), dtype=bool)
    targets = build_seat_targets(
        winner="a",
        seat=0,
        policy=SparsePolicy(indices=(3969,), probs=(1.0,)),
        ownership=ownership,
        armies=armies,
        generals=generals,
        castles=castles,
        final_land=(2, 1),
        final_army=(5, 8),
        final_castles=(0, 0),
        current_turn=10,
        terminal_turn=40,
    )
    pad = targets.board_mask.shape[0]
    tensor = np.zeros((49, pad, pad), dtype=np.float32)
    tensor[0] = targets.board_mask
    legal = (targets.policy > 0).astype(bool)
    # Pass is always legal for the mask helper; ensure PASS is on.
    legal[3969] = True
    return PilotSample(
        item_id="fake",
        sample_seat=0,
        tensor=tensor,
        legal_mask=legal,
        targets=targets,
        source_label="ResBot_reconstructions",
        outcome="a",
    )


def test_pilot_one_backward_and_checkpoint(tmp_path: Path):
    import sys

    bot = REPO / "bots" / "morpheus"
    for entry in (REPO, REPO / "bots", bot):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)
    from network import make_model

    bundle = load_pilot_objective_bundle(OBJECTIVE)
    model = make_model(seed=0, n_blocks=2)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sample = _tiny_sample()
    loss0, _ = pilot_train_step(
        model=model,
        optimizer=opt,
        samples=[sample],
        objective=bundle["training"],
    )
    loss1, _ = pilot_train_step(
        model=model,
        optimizer=opt,
        samples=[sample],
        objective=bundle["training"],
    )
    assert np.isfinite(loss0) and np.isfinite(loss1)

    ckpt = tmp_path / "ckpt"
    save_pilot_checkpoint(model, ckpt, meta={"seed": 0, "n_blocks": 2})
    reloaded, meta = load_pilot_checkpoint(ckpt)
    assert meta["n_blocks"] == 2
    reloaded.eval()
    with torch.no_grad():
        x = torch.as_tensor(sample.tensor).unsqueeze(0)
        out = reloaded(x)
    assert out.wdl_logits.shape[-1] == 3


def test_curriculum_item_sample_seat_for_pilot():
    item = CurriculumItem.build(
        class_id=CLASS_TACTICAL,
        engine_version="e",
        map_seed=1,
        source_label="erik.bystron_reconstructions",
        prefix_len=12,
        sample_seat=1,
        decisive=True,
        outcome="b",
    )
    assert item.sample_seat == 1
