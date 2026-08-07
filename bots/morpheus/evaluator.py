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
from tactics import (
    DEFAULT_SHAPING_FLOOR_FRAC,
    DEFAULT_SHAPING_LAMBDA,
    DEFAULT_SHAPING_LOG_CLIP,
    apply_pre_contact_prior,
    enemy_is_visible,
    play_mask,
)
from tensor import build_tensor

Array = np.ndarray


@dataclass
class NetworkEvaluator:
    """``SearchEvaluator`` that runs dedicated online export entry points.

    Belief proposal and enemy priors use policy-only inference. Root and leaf
    evaluation use policy plus WDL. Auxiliary heads are not computed on the
    online search path.

    Root priors pass through the Part 17 shaping blend; the knobs come from
    ``deployment.json`` so every setting is a distinct content-hash entity.
    ``last_unshaped_prior`` keeps the raw legal-normalized network prior from
    the most recent root evaluation for the "who's deciding" probe.
    """

    session: InferenceSession
    previous_action: Optional[tuple[int, int, int, int, int]] = None
    shaping_lambda_pre_contact: float = DEFAULT_SHAPING_LAMBDA
    shaping_lambda_post_contact: float = DEFAULT_SHAPING_LAMBDA
    shaping_log_clip: float = DEFAULT_SHAPING_LOG_CLIP
    shaping_floor_frac: float = DEFAULT_SHAPING_FLOOR_FRAC
    last_unshaped_prior: Optional[Array] = None

    def set_previous_action(
        self, action: Optional[tuple[int, int, int, int, int]]
    ) -> None:
        self.previous_action = action

    def _shape_root(self, prior: Array, obs, memory, mask, belief) -> Array:
        self.last_unshaped_prior = np.array(prior, dtype=np.float64, copy=True)
        lam = (
            self.shaping_lambda_post_contact
            if enemy_is_visible(obs, memory)
            else self.shaping_lambda_pre_contact
        )
        return apply_pre_contact_prior(
            prior,
            obs,
            memory,
            mask=mask,
            belief=belief,
            lam=float(lam),
            log_clip=float(self.shaping_log_clip),
            floor_frac=float(self.shaping_floor_frac),
        )

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
        shape: bool = False,
    ) -> tuple[Array, float]:
        x = self._tensor(obs, memory, belief)
        policy, pass_logit, wdl_logits = self.session.forward_policy_wdl(x)
        mask_np = np.asarray(play_mask(obs, memory), dtype=bool)
        mask = torch.from_numpy(mask_np).unsqueeze(0)
        prior_t = legal_normalized_policy(policy, pass_logit, mask)
        prior = prior_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
        if shape:
            prior = self._shape_root(prior, obs, memory, mask_np, belief)
        value_t = backup_value(wdl_logits, from_root=from_root)
        return prior, float(value_t.squeeze(0).item())

    def evaluate_many(
        self,
        items: list[tuple[object, VisibleMemory, BeliefState, bool, bool]],
    ) -> list[tuple[Array, float]]:
        """Batched evaluation. Items are ``(obs, mem, belief, from_root, shape)``.

        ``from_root`` picks the value perspective; ``shape`` applies the root
        prior blend. Leaves run ``(True, False)``: root perspective, unshaped.
        """
        if not items:
            return []
        if len(items) == 1:
            obs, memory, belief, from_root, shape = items[0]
            return [
                self.evaluate(
                    obs, memory, belief, from_root=from_root, shape=shape
                )
            ]
        tensors = []
        masks = []
        for obs, memory, belief, _from_root, _shape in items:
            tensors.append(self._tensor(obs, memory, belief).squeeze(0))
            masks.append(np.asarray(play_mask(obs, memory), dtype=bool))
        x = torch.stack(tensors, dim=0)
        policy, pass_logit, wdl_logits = self.session.forward_policy_wdl(x)
        mask_t = torch.from_numpy(np.stack(masks, axis=0))
        prior_t = legal_normalized_policy(policy, pass_logit, mask_t)
        values_t = backup_value(wdl_logits, from_root=True)
        results: list[tuple[Array, float]] = []
        for i, (obs_i, mem_i, belief_i, from_root, shape) in enumerate(items):
            prior = prior_t[i].detach().cpu().numpy().astype(np.float64)
            value = float(values_t[i].item())
            if not from_root:
                value = -value
            if shape:
                prior = self._shape_root(prior, obs_i, mem_i, masks[i], belief_i)
            results.append((prior, value))
        return results

    def policy_priors_many(
        self,
        items: list[tuple[object, VisibleMemory, BeliefState]],
    ) -> list[Array]:
        """Batched policy-only priors for enemy tables. Does not compute WDL."""
        if not items:
            return []
        tensors = []
        masks = []
        for obs, memory, belief in items:
            tensors.append(self._tensor(obs, memory, belief).squeeze(0))
            masks.append(np.asarray(legal_mask(obs, memory), dtype=bool))
        x = torch.stack(tensors, dim=0)
        policy, pass_logit = self.session.forward_policy(x)
        mask_t = torch.from_numpy(np.stack(masks, axis=0))
        prior_t = legal_normalized_policy(policy, pass_logit, mask_t)
        return [
            prior_t[i].detach().cpu().numpy().astype(np.float64)
            for i in range(prior_t.shape[0])
        ]

    def policy_logits(self, tensors: Array) -> Array:
        """Belief-proposal PolicyFn: ``(B,49,21,21) -> (B,3970)`` logits."""
        batch = np.asarray(tensors, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[None, ...]
        x = torch.from_numpy(batch)
        policy, pass_logit = self.session.forward_policy(x)
        logits = flatten_policy_logits(policy, pass_logit)
        return logits.detach().cpu().numpy().astype(np.float64)


@dataclass
class ShapedUniformEvaluator:
    """Network-free arm of the Part 17 C1 ablation.

    Same play mask, same shaping blend, same hard rules as ``NetworkEvaluator``
    — only the learned prior and value are gone. A flat prior over the play mask
    means the blend output is the heuristic ranking alone, so the contrast
    against the network arm measures exactly what the network contributes.
    """

    shaping_lambda_pre_contact: float = DEFAULT_SHAPING_LAMBDA
    shaping_lambda_post_contact: float = DEFAULT_SHAPING_LAMBDA
    shaping_log_clip: float = DEFAULT_SHAPING_LOG_CLIP
    shaping_floor_frac: float = DEFAULT_SHAPING_FLOOR_FRAC
    previous_action: Optional[tuple[int, int, int, int, int]] = None
    last_unshaped_prior: Optional[Array] = None

    def set_previous_action(
        self, action: Optional[tuple[int, int, int, int, int]]
    ) -> None:
        self.previous_action = action

    def _uniform(self, mask: Array) -> Array:
        prior = np.zeros(mask.shape, dtype=np.float64)
        n = int(mask.sum())
        if n <= 0:
            prior[-1] = 1.0
            return prior
        prior[mask] = 1.0 / float(n)
        return prior

    def evaluate(
        self,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
        *,
        from_root: bool,
        shape: bool = False,
    ) -> tuple[Array, float]:
        if from_root:
            mask = np.asarray(play_mask(obs, memory), dtype=bool)
            prior = self._uniform(mask)
            if not shape:
                return prior, 0.0
            self.last_unshaped_prior = np.array(prior, dtype=np.float64, copy=True)
            lam = (
                self.shaping_lambda_post_contact
                if enemy_is_visible(obs, memory)
                else self.shaping_lambda_pre_contact
            )
            prior = apply_pre_contact_prior(
                prior,
                obs,
                memory,
                mask=mask,
                belief=belief,
                lam=float(lam),
                log_clip=float(self.shaping_log_clip),
                floor_frac=float(self.shaping_floor_frac),
            )
            return prior, 0.0
        return self._uniform(np.asarray(legal_mask(obs, memory), dtype=bool)), 0.0

    def evaluate_many(
        self,
        items: list[tuple[object, VisibleMemory, BeliefState, bool, bool]],
    ) -> list[tuple[Array, float]]:
        return [
            self.evaluate(obs, mem, belief, from_root=from_root, shape=shape)
            for obs, mem, belief, from_root, shape in items
        ]

    def policy_priors_many(
        self,
        items: list[tuple[object, VisibleMemory, BeliefState]],
    ) -> list[Array]:
        return [
            self._uniform(np.asarray(legal_mask(obs, mem), dtype=bool))
            for obs, mem, _belief in items
        ]
