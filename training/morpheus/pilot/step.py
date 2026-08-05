"""One AdamW training step on a small pilot batch."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from training.morpheus.objective.config import ObjectiveConfig
from training.morpheus.objective.losses import LossTerms, compute_objective_losses
from training.morpheus.pilot.batch import PilotSample

REPO = Path(__file__).resolve().parents[3]
MORPHEUS_BOT = REPO / "bots" / "morpheus"


def _ensure_bot_path() -> None:
    for entry in (REPO, REPO / "bots", MORPHEUS_BOT):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


def _stack_batch(samples: Sequence[PilotSample], device: torch.device):
    x = torch.as_tensor(
        np.stack([s.tensor for s in samples], axis=0),
        dtype=torch.float32,
        device=device,
    )
    legal = torch.as_tensor(
        np.stack([s.legal_mask for s in samples], axis=0),
        dtype=torch.bool,
        device=device,
    )
    policy_t = torch.as_tensor(
        np.stack([s.targets.policy for s in samples], axis=0),
        dtype=torch.float32,
        device=device,
    )
    wdl_t = torch.as_tensor(
        np.stack([s.targets.wdl for s in samples], axis=0),
        dtype=torch.float32,
        device=device,
    )
    hidden_owner = torch.as_tensor(
        np.stack([s.targets.hidden_owner for s in samples], axis=0),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    enemy_bins = torch.as_tensor(
        np.stack([s.targets.enemy_army_bin for s in samples], axis=0),
        dtype=torch.int64,
        device=device,
    )
    enemy_general = torch.as_tensor(
        np.stack([s.targets.enemy_general for s in samples], axis=0),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    hidden_castle = torch.as_tensor(
        np.stack([s.targets.hidden_castle for s in samples], axis=0),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    board_mask = torch.as_tensor(
        np.stack([s.targets.board_mask for s in samples], axis=0),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    return {
        "x": x,
        "legal": legal,
        "policy_t": policy_t,
        "wdl_t": wdl_t,
        "hidden_owner": hidden_owner,
        "enemy_bins": enemy_bins,
        "enemy_general": enemy_general,
        "hidden_castle": hidden_castle,
        "board_mask": board_mask,
        "samples": samples,
    }


def pilot_train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[PilotSample],
    objective: ObjectiveConfig,
    device: torch.device | None = None,
) -> tuple[float, LossTerms]:
    """Forward + backward + AdamW step. Returns (total_loss, terms)."""
    if not samples:
        raise ValueError("pilot_train_step requires at least one sample")
    _ensure_bot_path()
    from network import flatten_policy_logits

    dev = device or next(model.parameters()).device
    batch = _stack_batch(samples, dev)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(batch["x"])
    policy_logits = flatten_policy_logits(out.policy, out.pass_logit)

    scaled = [s.targets.scaled_scalars(objective.scalar_normalization) for s in samples]
    land_t = torch.as_tensor(
        [row["land_margin"] for row in scaled], dtype=torch.float32, device=dev
    )
    army_t = torch.as_tensor(
        [row["army_margin"] for row in scaled], dtype=torch.float32, device=dev
    )
    castle_t = torch.as_tensor(
        [row["castle_margin"] for row in scaled], dtype=torch.float32, device=dev
    )
    turns_t = torch.as_tensor(
        [row["turns_to_termination"] for row in scaled],
        dtype=torch.float32,
        device=dev,
    )

    total, terms = compute_objective_losses(
        config=objective,
        policy_logits=policy_logits,
        wdl_logits=out.wdl_logits,
        hidden_owner_logits=out.hidden_owner,
        enemy_army_logits=out.enemy_army_bins,
        enemy_general_logits=out.enemy_general,
        hidden_castle_logits=out.hidden_castle,
        land_margin_pred=out.land_margin.reshape(-1),
        army_margin_pred=out.army_margin.reshape(-1),
        castle_margin_pred=out.castle_margin.reshape(-1),
        turns_pred=out.turns_to_termination.reshape(-1),
        policy_target=batch["policy_t"],
        wdl_target=batch["wdl_t"],
        hidden_owner_target=batch["hidden_owner"],
        enemy_army_bin_target=batch["enemy_bins"],
        enemy_general_target=batch["enemy_general"],
        hidden_castle_target=batch["hidden_castle"],
        land_margin_target=land_t,
        army_margin_target=army_t,
        castle_margin_target=castle_t,
        turns_target=turns_t,
        legal_mask=batch["legal"],
        board_mask=batch["board_mask"],
    )
    total.backward()
    optimizer.step()
    return float(terms.total), terms
