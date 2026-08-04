"""Explicit joint training losses. Weights come only from ObjectiveConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from training.morpheus.objective.config import LossWeights, ObjectiveConfig

Array = np.ndarray
Tensor = torch.Tensor


@dataclass(frozen=True)
class LossTerms:
    """Per-head losses before weighting, plus the weighted total."""

    policy: float
    wdl: float
    hidden_owner: float
    enemy_army_bins: float
    enemy_general: float
    hidden_castle: float
    land_margin: float
    army_margin: float
    castle_margin: float
    turns_to_termination: float
    total: float
    weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": {
                "policy": self.policy,
                "wdl": self.wdl,
                "hidden_owner": self.hidden_owner,
                "enemy_army_bins": self.enemy_army_bins,
                "enemy_general": self.enemy_general,
                "hidden_castle": self.hidden_castle,
                "land_margin": self.land_margin,
                "army_margin": self.army_margin,
                "castle_margin": self.castle_margin,
                "turns_to_termination": self.turns_to_termination,
            },
            "weights": dict(self.weights),
            "total": self.total,
        }


def _as_tensor(x: Array | Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype or torch.float32)


def policy_cross_entropy(
    logits: Tensor,
    target: Tensor,
    legal_mask: Optional[Tensor] = None,
) -> Tensor:
    """Cross-entropy of dense root-average-strategy target vs legal logits."""
    logits = logits.reshape(logits.shape[0], -1)
    target = target.reshape(target.shape[0], -1).to(dtype=logits.dtype)
    if legal_mask is not None:
        mask = legal_mask.reshape(logits.shape[0], -1).to(dtype=torch.bool)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        target = target * mask.to(dtype=target.dtype)
        # Renormalize target over legal mass (should already be legal).
        mass = target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        target = target / mass
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def wdl_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """Soft cross-entropy on (win, draw, loss) targets."""
    logits = logits.reshape(-1, 3)
    target = target.reshape(-1, 3).to(dtype=logits.dtype)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def masked_bce_with_logits(
    logits: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    logits = logits.reshape(logits.shape[0], -1)
    target = target.reshape(target.shape[0], -1).to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if mask is None:
        return loss.mean()
    m = mask.reshape(logits.shape[0], -1).to(dtype=loss.dtype)
    denom = m.sum().clamp_min(1.0)
    return (loss * m).sum() / denom


def enemy_army_bin_cross_entropy(
    logits: Tensor,
    bin_indices: Tensor,
    enemy_mask: Tensor,
) -> Tensor:
    """Per-cell CE over army bins, averaged on enemy-owned cells only."""
    # logits: N×C×H×W ; bin_indices: N×H×W (int, -1 ignore)
    n, c, h, w = logits.shape
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, c)
    flat_idx = bin_indices.reshape(-1).to(dtype=torch.long)
    flat_mask = enemy_mask.reshape(-1).to(dtype=torch.bool)
    valid = flat_mask & (flat_idx >= 0) & (flat_idx < c)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(flat_logits[valid], flat_idx[valid])


def masked_soft_ce(
    logits: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Soft spatial CE (e.g. enemy-general one-hot / soft map)."""
    n = logits.shape[0]
    flat_logits = logits.reshape(n, -1)
    flat_target = target.reshape(n, -1).to(dtype=logits.dtype)
    if mask is not None:
        m = mask.reshape(n, -1).to(dtype=torch.bool)
        flat_logits = flat_logits.masked_fill(~m, torch.finfo(logits.dtype).min)
        flat_target = flat_target * m.to(dtype=flat_target.dtype)
        mass = flat_target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        flat_target = flat_target / mass
    log_probs = F.log_softmax(flat_logits, dim=-1)
    return -(flat_target * log_probs).sum(dim=-1).mean()


def huber_regression(pred: Tensor, target: Tensor) -> Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1).to(dtype=pred.dtype)
    return F.smooth_l1_loss(pred, target)


def combine_losses(
    terms: Mapping[str, Tensor],
    weights: LossWeights | Mapping[str, float],
) -> tuple[Tensor, LossTerms]:
    """Weighted sum. Missing weight keys fail closed."""
    if isinstance(weights, LossWeights):
        wdict = weights.to_dict()
    else:
        wdict = LossWeights.from_dict(weights).to_dict()
    missing = [k for k in wdict if k not in terms]
    if missing:
        raise ValueError(f"loss terms missing for weights: {missing}")
    total = None
    scalars: dict[str, float] = {}
    for key, weight in wdict.items():
        term = terms[key]
        scalars[key] = float(term.detach().cpu())
        piece = float(weight) * term
        total = piece if total is None else total + piece
    assert total is not None
    return total, LossTerms(
        policy=scalars["policy"],
        wdl=scalars["wdl"],
        hidden_owner=scalars["hidden_owner"],
        enemy_army_bins=scalars["enemy_army_bins"],
        enemy_general=scalars["enemy_general"],
        hidden_castle=scalars["hidden_castle"],
        land_margin=scalars["land_margin"],
        army_margin=scalars["army_margin"],
        castle_margin=scalars["castle_margin"],
        turns_to_termination=scalars["turns_to_termination"],
        total=float(total.detach().cpu()),
        weights=wdict,
    )


def compute_objective_losses(
    *,
    config: ObjectiveConfig,
    policy_logits: Tensor,
    wdl_logits: Tensor,
    hidden_owner_logits: Tensor,
    enemy_army_logits: Tensor,
    enemy_general_logits: Tensor,
    hidden_castle_logits: Tensor,
    land_margin_pred: Tensor,
    army_margin_pred: Tensor,
    castle_margin_pred: Tensor,
    turns_pred: Tensor,
    policy_target: Tensor,
    wdl_target: Tensor,
    hidden_owner_target: Tensor,
    enemy_army_bin_target: Tensor,
    enemy_general_target: Tensor,
    hidden_castle_target: Tensor,
    land_margin_target: Tensor,
    army_margin_target: Tensor,
    castle_margin_target: Tensor,
    turns_target: Tensor,
    legal_mask: Optional[Tensor] = None,
    board_mask: Optional[Tensor] = None,
) -> tuple[Tensor, LossTerms]:
    """Compute every head loss and the weighted total from an explicit config."""
    enemy_mask = hidden_owner_target > 0.5
    if board_mask is not None:
        enemy_mask = enemy_mask & (board_mask > 0.5)

    terms = {
        "policy": policy_cross_entropy(policy_logits, policy_target, legal_mask),
        "wdl": wdl_cross_entropy(wdl_logits, wdl_target),
        "hidden_owner": masked_bce_with_logits(
            hidden_owner_logits, hidden_owner_target, board_mask
        ),
        "enemy_army_bins": enemy_army_bin_cross_entropy(
            enemy_army_logits, enemy_army_bin_target, enemy_mask
        ),
        "enemy_general": masked_soft_ce(
            enemy_general_logits, enemy_general_target, board_mask
        ),
        "hidden_castle": masked_bce_with_logits(
            hidden_castle_logits, hidden_castle_target, board_mask
        ),
        "land_margin": huber_regression(land_margin_pred, land_margin_target),
        "army_margin": huber_regression(army_margin_pred, army_margin_target),
        "castle_margin": huber_regression(castle_margin_pred, castle_margin_target),
        "turns_to_termination": huber_regression(turns_pred, turns_target),
    }
    return combine_losses(terms, config.loss_weights)


def hand_policy_ce(logits: Array, target: Array) -> float:
    """NumPy reference CE for hand-computed fixtures."""
    logits_t = _as_tensor(logits.reshape(1, -1))
    target_t = _as_tensor(target.reshape(1, -1))
    return float(policy_cross_entropy(logits_t, target_t).item())


def hand_wdl_ce(logits: Array, target: Array) -> float:
    logits_t = _as_tensor(logits.reshape(1, 3))
    target_t = _as_tensor(target.reshape(1, 3))
    return float(wdl_cross_entropy(logits_t, target_t).item())
