"""Enemy-action proposals for the belief filter.

Proposal policy injection is separate from filtering so deterministic fixtures
do not require a trained checkpoint. The default proposal is uniform over
legal enemy actions (plus pass).
"""
from __future__ import annotations

import hashlib
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from action import decode_action, legal_mask
from belief import Action5, BeliefState, Particle, as_action5, pass_action
from memory import VisibleMemory, update_memory
from observe import emit_observation, observation_hash
from tensor import BeliefSummary, build_tensor, zero_belief

Array = np.ndarray

# Batch ceiling matches the runtime enemy-proposal budget.
MAX_PROPOSAL_BATCH = 64


class PolicyModel(Protocol):
    """Minimal policy interface: logits over the 3970-action space."""

    def policy_logits(self, tensors: Array) -> Array:
        """``tensors`` is ``(B, 49, 21, 21)``; returns ``(B, 3970)`` logits."""
        ...


PolicyFn = Callable[[Array], Array]  # (B,49,21,21) -> (B,3970) logits


def _softmax_masked(logits: Array, mask: Array) -> Array:
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(logits, dtype=np.float64)
    if not np.any(mask):
        # Pass only.
        from action import PASS_INDEX

        out[PASS_INDEX] = 1.0
        return out
    clipped = np.where(mask, logits, -1e9)
    clipped = clipped - clipped.max()
    exp = np.exp(clipped)
    exp = np.where(mask, exp, 0.0)
    total = exp.sum()
    if total <= 0.0:
        from action import PASS_INDEX

        out[PASS_INDEX] = 1.0
        return out
    return exp / total


def enemy_info_tensor(
    particle: Particle,
    belief: BeliefState,
    *,
    level_zero_belief: Optional[BeliefSummary] = None,
) -> Array:
    """Build the enemy-perspective tensor for one particle (level-zero belief)."""
    enemy_obs = emit_observation(particle.state, belief.enemy_seat)
    mem = update_memory(particle.enemy_memory, enemy_obs)
    H, W = int(enemy_obs.H), int(enemy_obs.W)
    summary = level_zero_belief if level_zero_belief is not None else zero_belief(H, W)
    return build_tensor(
        enemy_obs,
        mem,
        belief=summary,
        previous_action=particle.enemy_prev_action,
    )


def dedupe_enemy_tensors(
    tensors: Sequence[Array],
) -> tuple[list[Array], list[int]]:
    """Return unique tensors and a map from particle index → unique index."""
    unique: list[Array] = []
    index_of: dict[bytes, int] = {}
    mapping: list[int] = []
    for tensor in tensors:
        key = np.asarray(tensor, dtype=np.float32).tobytes()
        digest = hashlib.sha256(key).digest()
        if digest not in index_of:
            index_of[digest] = len(unique)
            unique.append(np.asarray(tensor, dtype=np.float32))
        mapping.append(index_of[digest])
    return unique, mapping


def uniform_legal_probs(obs, memory: VisibleMemory) -> Array:
    """Uniform distribution over currently legal actions (always includes pass)."""
    mask = legal_mask(obs, memory)
    n = int(mask.sum())
    probs = np.zeros(mask.shape, dtype=np.float64)
    if n == 0:
        from action import PASS_INDEX

        probs[PASS_INDEX] = 1.0
        return probs
    probs[mask] = 1.0 / float(n)
    return probs


def sample_from_probs(probs: Array, rng: np.random.Generator) -> Action5:
    idx = int(rng.choice(len(probs), p=probs))
    return as_action5(decode_action(idx))


def propose_enemy_actions(
    belief: BeliefState,
    rng: np.random.Generator,
    *,
    policy: Optional[PolicyFn] = None,
    top_k_force: int = 0,
) -> list[Action5]:
    """Sample one enemy action per particle.

    When ``policy`` is None, each particle draws uniformly from its legal mask.
    When ``policy`` is set, tensors are deduplicated and evaluated in one batch
    of at most 64 unique inputs.
    """
    if belief.n == 0:
        return []

    # Build enemy obs + memory + legal probs per particle.
    enemy_obs_list = []
    enemy_mem_list = []
    tensors: list[Array] = []
    for particle in belief.particles:
        enemy_obs = emit_observation(particle.state, belief.enemy_seat)
        enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
        enemy_obs_list.append(enemy_obs)
        enemy_mem_list.append(enemy_mem)
        tensors.append(enemy_info_tensor(particle, belief))

    if policy is None:
        actions: list[Action5] = []
        for obs, mem in zip(enemy_obs_list, enemy_mem_list):
            actions.append(sample_from_probs(uniform_legal_probs(obs, mem), rng))
        return actions

    unique, mapping = dedupe_enemy_tensors(tensors)
    # Cap batch; if somehow larger, evaluate in chunks.
    all_logits: list[Array] = []
    for start in range(0, len(unique), MAX_PROPOSAL_BATCH):
        batch = np.stack(unique[start : start + MAX_PROPOSAL_BATCH], axis=0)
        logits = np.asarray(policy(batch), dtype=np.float64)
        if logits.ndim == 1:
            logits = logits[None, :]
        all_logits.append(logits)
    stacked = np.concatenate(all_logits, axis=0)

    actions = []
    for p_idx, u_idx in enumerate(mapping):
        obs = enemy_obs_list[p_idx]
        mem = enemy_mem_list[p_idx]
        mask = legal_mask(obs, mem)
        probs = _softmax_masked(stacked[u_idx], mask)
        if top_k_force > 0:
            # Optional: mix in top-k for recovery beam seeding (caller may ignore).
            pass
        actions.append(sample_from_probs(probs, rng))
    return actions


def policy_action_probs(
    belief: BeliefState,
    particle: Particle,
    *,
    policy: Optional[PolicyFn] = None,
) -> Array:
    """Return a length-3970 probability vector for one particle's enemy seat."""
    enemy_obs = emit_observation(particle.state, belief.enemy_seat)
    enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
    mask = legal_mask(enemy_obs, enemy_mem)
    if policy is None:
        return uniform_legal_probs(enemy_obs, enemy_mem)
    tensor = enemy_info_tensor(particle, belief)[None, ...]
    logits = np.asarray(policy(tensor), dtype=np.float64).reshape(-1)
    return _softmax_masked(logits, mask)


def top_legal_actions(
    probs: Array,
    mask: Array,
    k: int,
) -> list[Action5]:
    """Highest-prior legal actions, always including pass when legal."""
    from action import PASS_INDEX

    scores = np.where(mask, probs, -1.0)
    order = np.argsort(-scores)
    out: list[Action5] = []
    seen: set[Action5] = set()
    # Pass first.
    if mask[PASS_INDEX]:
        a = pass_action()
        out.append(a)
        seen.add(a)
    for idx in order:
        if not mask[int(idx)]:
            continue
        action = as_action5(decode_action(int(idx)))
        if action in seen:
            continue
        out.append(action)
        seen.add(action)
        if len(out) >= k + 1:  # pass + top k
            break
    return out


def enemy_info_key(particle: Particle, belief: BeliefState) -> bytes:
    """Hash of the enemy observation for table / batch deduplication."""
    enemy_obs = emit_observation(particle.state, belief.enemy_seat)
    return observation_hash(enemy_obs)
