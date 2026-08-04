"""Network-backed search and proposal adapters for the frozen export."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from action import legal_mask
from belief import BeliefState
from inference import InferenceSession
from memory import VisibleMemory
from network import (
    backup_value,
    flatten_policy_logits,
    legal_normalized_policy,
)
from particle_summary import summarize_belief
from tensor import build_tensor

Array = np.ndarray


@dataclass
class NetworkEvaluator:
    """``SearchEvaluator`` that runs dedicated online export entry points.

    Belief proposal and enemy priors use policy-only inference. Root and leaf
    evaluation use policy plus WDL. Auxiliary heads are not computed on the
    online search path.
    """

    session: InferenceSession
    previous_action: Optional[tuple[int, int, int, int, int]] = None

    def set_previous_action(
        self, action: Optional[tuple[int, int, int, int, int]]
    ) -> None:
        self.previous_action = action

    def _tensor(
        self,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
    ) -> torch.Tensor:
        summary = summarize_belief(belief) if belief.n > 0 else None
        arr = build_tensor(
            obs,
            memory,
            belief=summary,
            previous_action=self.previous_action,
        )
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0)

    def evaluate(
        self,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
        *,
        from_root: bool,
    ) -> tuple[Array, float]:
        x = self._tensor(obs, memory, belief)
        policy, pass_logit, wdl_logits = self.session.forward_policy_wdl(x)
        mask = torch.from_numpy(np.asarray(legal_mask(obs, memory), dtype=bool)).unsqueeze(0)
        prior_t = legal_normalized_policy(policy, pass_logit, mask)
        prior = prior_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
        value_t = backup_value(wdl_logits, from_root=from_root)
        return prior, float(value_t.squeeze(0).item())

    def evaluate_many(
        self,
        items: list[tuple[object, VisibleMemory, BeliefState, bool]],
    ) -> list[tuple[Array, float]]:
        """Batched root/leaf evaluation. ``items`` are ``(obs, mem, belief, from_root)``."""
        if not items:
            return []
        if len(items) == 1:
            obs, memory, belief, from_root = items[0]
            return [self.evaluate(obs, memory, belief, from_root=from_root)]
        tensors = []
        masks = []
        from_roots = []
        for obs, memory, belief, from_root in items:
            tensors.append(self._tensor(obs, memory, belief).squeeze(0))
            masks.append(np.asarray(legal_mask(obs, memory), dtype=bool))
            from_roots.append(bool(from_root))
        x = torch.stack(tensors, dim=0)
        policy, pass_logit, wdl_logits = self.session.forward_policy_wdl(x)
        mask_t = torch.from_numpy(np.stack(masks, axis=0))
        prior_t = legal_normalized_policy(policy, pass_logit, mask_t)
        values_t = backup_value(wdl_logits, from_root=True)
        results: list[tuple[Array, float]] = []
        for i, from_root in enumerate(from_roots):
            prior = prior_t[i].detach().cpu().numpy().astype(np.float64)
            value = float(values_t[i].item())
            if not from_root:
                value = -value
            results.append((prior, value))
        return results

    def policy_logits(self, tensors: Array) -> Array:
        """Belief-proposal PolicyFn: ``(B,49,21,21) -> (B,3970)`` logits."""
        batch = np.asarray(tensors, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[None, ...]
        x = torch.from_numpy(batch)
        policy, pass_logit = self.session.forward_policy(x)
        logits = flatten_policy_logits(policy, pass_logit)
        return logits.detach().cpu().numpy().astype(np.float64)
