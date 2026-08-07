"""Enemy priors use policy-only inference; root/leaf keep policy+WDL."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.morpheus

from action import N_ACTIONS, PASS_INDEX, legal_mask
from belief import BeliefConfig, initialize_belief
from evaluator import NetworkEvaluator
from memory import empty_memory, update_memory
from network import BOARD, IN_CHANNELS, POLICY_CHANNELS
from observe import emit_observation
from search import SearchConfig, SearchController
from state import create_initial_state


@dataclass
class _CountingSession:
    """Minimal inference stub that counts policy vs policy+WDL calls."""

    policy_calls: int = 0
    policy_wdl_calls: int = 0
    last_policy_batch: int = 0
    last_wdl_batch: int = 0

    def forward_policy(self, x: torch.Tensor):
        x = torch.as_tensor(x)
        self.policy_calls += 1
        self.last_policy_batch = int(x.shape[0])
        b = int(x.shape[0])
        policy = torch.zeros(b, POLICY_CHANNELS, BOARD, BOARD)
        pass_logit = torch.zeros(b, 1)
        # Prefer pass so legal normalization stays well-defined.
        pass_logit[:, 0] = 1.0
        return policy, pass_logit

    def forward_policy_wdl(self, x: torch.Tensor):
        x = torch.as_tensor(x)
        self.policy_wdl_calls += 1
        self.last_wdl_batch = int(x.shape[0])
        b = int(x.shape[0])
        policy = torch.zeros(b, POLICY_CHANNELS, BOARD, BOARD)
        pass_logit = torch.zeros(b, 1)
        pass_logit[:, 0] = 1.0
        wdl = torch.zeros(b, 3)
        wdl[:, 1] = 1.0
        return policy, pass_logit, wdl


def _ctx(seed: int = 0):
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[0, 0] = 1
    grid[7, 7] = 2
    state = create_initial_state(grid)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(8, 8), obs)
    rng = np.random.default_rng(seed)
    belief = initialize_belief(
        obs,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=4, min_general_distance=5),
    )
    return obs, mem, belief, rng


def test_enemy_priors_use_policy_only_not_wdl():
    obs, mem, belief, rng = _ctx(2)
    session = _CountingSession()
    ev = NetworkEvaluator(session)
    ctl = SearchController(
        seat=0,
        evaluator=ev,
        config=SearchConfig(depth=2, pending_batch=2, n_particles=4),
        rng=rng,
    )
    ctl.ensure_root(obs, mem, belief)
    # Root evaluation needs WDL.
    assert session.policy_wdl_calls >= 1
    wdl_after_root = session.policy_wdl_calls
    policy_after_root = session.policy_calls

    outcome = ctl.select_path(belief, freeze_snapshot=True)
    from search import EnemyPriorRequest

    assert isinstance(outcome, EnemyPriorRequest)
    n = ctl.materialize_enemy_priors([outcome], belief)
    assert n >= 1
    assert session.policy_calls > policy_after_root
    assert session.policy_wdl_calls == wdl_after_root


def test_policy_priors_many_matches_evaluate_prior_channel():
    obs, mem, belief, rng = _ctx(3)
    session = _CountingSession()
    ev = NetworkEvaluator(session)
    items = [(obs, mem, belief)]
    priors = ev.policy_priors_many(items)
    assert len(priors) == 1
    prior_only = priors[0]
    prior_wdl, _value = ev.evaluate(obs, mem, belief, from_root=True)
    mask = legal_mask(obs, mem)
    assert prior_only.shape == (N_ACTIONS,)
    assert np.allclose(prior_only[mask], prior_wdl[mask])
    assert prior_only[PASS_INDEX] > 0.0
    assert session.policy_calls >= 1
    assert session.policy_wdl_calls >= 1
