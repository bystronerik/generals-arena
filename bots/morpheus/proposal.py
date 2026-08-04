"""Enemy-action proposals for the belief filter.

Proposal policy injection is separate from filtering so deterministic fixtures
do not require a trained checkpoint. The default proposal is uniform over
legal enemy actions (plus pass).
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from action import PASS_INDEX, decode_action, encode_action, legal_mask
from belief import Action5, BeliefState, Particle, as_action5, pass_action
from hashing import memory_digest
from memory import VisibleMemory, update_memory
from observe import emit_observation, observation_hash
from tensor import BeliefSummary, build_tensor, zero_belief

Array = np.ndarray

# Batch ceiling matches the runtime enemy-proposal budget.
MAX_PROPOSAL_BATCH = 64
# Legal masks with this many (or fewer) actions skip tensor + policy work.
# Pass is always legal, so early turns are typically a single legal action.
SINGLETON_LEGAL_MAX = 1


@dataclass
class ProposalTelemetry:
    """Passive counters for one ``propose_enemy_actions`` call."""

    n_particles: int = 0
    n_singleton_particles: int = 0
    n_unique_info_keys: int = 0
    n_unique_policy_inputs: int = 0
    n_policy_batches: int = 0

    def clear(self) -> None:
        self.n_particles = 0
        self.n_singleton_particles = 0
        self.n_unique_info_keys = 0
        self.n_unique_policy_inputs = 0
        self.n_policy_batches = 0


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
        out[PASS_INDEX] = 1.0
        return out
    clipped = np.where(mask, logits, -1e9)
    clipped = clipped - clipped.max()
    exp = np.exp(clipped)
    exp = np.where(mask, exp, 0.0)
    total = exp.sum()
    if total <= 0.0:
        out[PASS_INDEX] = 1.0
        return out
    return exp / total


def _singleton_probs(mask: Array) -> Array:
    """One-hot over the sole legal action (pass if the mask is empty)."""
    probs = np.zeros(np.asarray(mask).shape, dtype=np.float64)
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        probs[PASS_INDEX] = 1.0
    else:
        probs[int(idxs[0])] = 1.0
    return probs


def enemy_info_tensor(
    particle: Particle,
    belief: BeliefState,
    *,
    enemy_obs=None,
    enemy_mem: Optional[VisibleMemory] = None,
    level_zero_belief: Optional[BeliefSummary] = None,
) -> Array:
    """Build the enemy-perspective tensor for one particle (level-zero belief).

    Callers that already built ``enemy_obs`` / ``enemy_mem`` must pass them so
    observation and memory work is not repeated.
    """
    if enemy_obs is None:
        enemy_obs = emit_observation(
            particle.state, belief.enemy_seat, as_arrays=True
        )
    if enemy_mem is None:
        enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
    H, W = int(enemy_obs.H), int(enemy_obs.W)
    summary = level_zero_belief if level_zero_belief is not None else zero_belief(H, W)
    return build_tensor(
        enemy_obs,
        enemy_mem,
        belief=summary,
        previous_action=particle.enemy_prev_action,
    )


def proposal_info_key(
    enemy_obs,
    enemy_mem: VisibleMemory,
    previous_action: Optional[Action5],
) -> bytes:
    """Pre-tensor dedupe key: enemy observation, memory, previous enemy action."""
    prev_idx = (
        int(encode_action(previous_action))
        if previous_action is not None
        else -1
    )
    h = hashlib.sha256()
    h.update(observation_hash(enemy_obs))
    h.update(memory_digest(enemy_mem))
    h.update(struct.pack("<i", prev_idx))
    return h.digest()


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


def dedupe_proposal_keys(
    keys: Sequence[bytes],
) -> tuple[list[int], list[int]]:
    """Map particle indices onto unique proposal keys.

    Returns ``(unique_particle_indices, mapping)`` where ``mapping[p]`` is the
    index into ``unique_particle_indices``.
    """
    unique_indices: list[int] = []
    index_of: dict[bytes, int] = {}
    mapping: list[int] = []
    for p_idx, key in enumerate(keys):
        if key not in index_of:
            index_of[key] = len(unique_indices)
            unique_indices.append(p_idx)
        mapping.append(index_of[key])
    return unique_indices, mapping


def uniform_legal_probs(obs, memory: VisibleMemory) -> Array:
    """Uniform distribution over currently legal actions (always includes pass)."""
    mask = legal_mask(obs, memory)
    n = int(mask.sum())
    probs = np.zeros(mask.shape, dtype=np.float64)
    if n == 0:
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
    max_proposal_batch: int = MAX_PROPOSAL_BATCH,
    telemetry: Optional[ProposalTelemetry] = None,
) -> list[Action5]:
    """Sample one enemy action per particle.

    When ``policy`` is None, each particle draws uniformly from its legal mask.
    When ``policy`` is set, particles with at most ``SINGLETON_LEGAL_MAX`` legal
    actions skip tensor construction and the network (the action is
    deterministic). Remaining enemy information keys are deduplicated before
    tensor construction and evaluated in batches of at most
    ``max_proposal_batch`` unique inputs. Sampling still walks particles in
    order so the RNG stream matches the full-policy path.

    When ``telemetry`` is set, fill proposal counters for the measurement path.
    Telemetry never changes sampled actions.
    """
    if telemetry is not None:
        telemetry.clear()

    if belief.n == 0:
        return []
    batch_cap = max(1, int(max_proposal_batch))

    # Build enemy obs + memory once per particle (array grids on the hot path).
    enemy_obs_list = []
    enemy_mem_list = []
    for particle in belief.particles:
        enemy_obs = emit_observation(
            particle.state, belief.enemy_seat, as_arrays=True
        )
        enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
        enemy_obs_list.append(enemy_obs)
        enemy_mem_list.append(enemy_mem)

    keys = [
        proposal_info_key(obs, mem, particle.enemy_prev_action)
        for obs, mem, particle in zip(
            enemy_obs_list, enemy_mem_list, belief.particles
        )
    ]
    if telemetry is not None:
        telemetry.n_particles = int(belief.n)
        telemetry.n_unique_info_keys = len(set(keys))

    if policy is None:
        actions: list[Action5] = []
        n_singleton = 0
        for obs, mem in zip(enemy_obs_list, enemy_mem_list):
            probs = uniform_legal_probs(obs, mem)
            if int(np.count_nonzero(probs > 0.0)) <= SINGLETON_LEGAL_MAX:
                n_singleton += 1
            actions.append(sample_from_probs(probs, rng))
        if telemetry is not None:
            telemetry.n_singleton_particles = n_singleton
        return actions

    masks = [
        legal_mask(obs, mem)
        for obs, mem in zip(enemy_obs_list, enemy_mem_list)
    ]
    multi_indices = [
        p_idx
        for p_idx, mask in enumerate(masks)
        if int(mask.sum()) > SINGLETON_LEGAL_MAX
    ]
    if telemetry is not None:
        telemetry.n_singleton_particles = int(belief.n) - len(multi_indices)

    multi_logits: dict[int, Array] = {}
    n_policy_batches = 0
    n_unique_policy_inputs = 0
    if multi_indices:
        multi_keys = [keys[p_idx] for p_idx in multi_indices]
        unique_local, mapping_local = dedupe_proposal_keys(multi_keys)
        n_unique_policy_inputs = len(unique_local)

        unique_tensors: list[Array] = []
        for local_u in unique_local:
            p_idx = multi_indices[local_u]
            particle = belief.particles[p_idx]
            unique_tensors.append(
                enemy_info_tensor(
                    particle,
                    belief,
                    enemy_obs=enemy_obs_list[p_idx],
                    enemy_mem=enemy_mem_list[p_idx],
                )
            )

        all_logits: list[Array] = []
        for start in range(0, len(unique_tensors), batch_cap):
            batch = np.stack(unique_tensors[start : start + batch_cap], axis=0)
            logits = np.asarray(policy(batch), dtype=np.float64)
            if logits.ndim == 1:
                logits = logits[None, :]
            all_logits.append(logits)
            n_policy_batches += 1
        stacked = np.concatenate(all_logits, axis=0)

        for local_p, u_idx in enumerate(mapping_local):
            multi_logits[multi_indices[local_p]] = stacked[u_idx]

    if telemetry is not None:
        telemetry.n_unique_policy_inputs = n_unique_policy_inputs
        telemetry.n_policy_batches = n_policy_batches

    actions = []
    for p_idx, mask in enumerate(masks):
        if p_idx in multi_logits:
            probs = _softmax_masked(multi_logits[p_idx], mask)
            if top_k_force > 0:
                # Optional: mix in top-k for recovery beam seeding (caller may ignore).
                pass
            actions.append(sample_from_probs(probs, rng))
        else:
            actions.append(sample_from_probs(_singleton_probs(mask), rng))
    return actions


def policy_action_probs(
    belief: BeliefState,
    particle: Particle,
    *,
    policy: Optional[PolicyFn] = None,
) -> Array:
    """Return a length-3970 probability vector for one particle's enemy seat."""
    enemy_obs = emit_observation(
        particle.state, belief.enemy_seat, as_arrays=True
    )
    enemy_mem = update_memory(particle.enemy_memory, enemy_obs)
    mask = legal_mask(enemy_obs, enemy_mem)
    if policy is None:
        return uniform_legal_probs(enemy_obs, enemy_mem)
    tensor = enemy_info_tensor(
        particle, belief, enemy_obs=enemy_obs, enemy_mem=enemy_mem
    )[None, ...]
    logits = np.asarray(policy(tensor), dtype=np.float64).reshape(-1)
    return _softmax_masked(logits, mask)


def top_legal_actions(
    probs: Array,
    mask: Array,
    k: int,
) -> list[Action5]:
    """Highest-prior legal actions, always including pass when legal."""
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
    enemy_obs = emit_observation(
        particle.state, belief.enemy_seat, as_arrays=True
    )
    return observation_hash(enemy_obs)
