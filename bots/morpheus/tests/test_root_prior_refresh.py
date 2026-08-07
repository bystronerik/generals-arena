"""Regression: live network prior must refresh root candidates after a stale open."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import N_ACTIONS, PASS_INDEX, legal_mask
from belief import BeliefConfig, initialize_belief
from memory import empty_memory, update_memory
from observe import emit_observation
from search import SearchConfig, SearchController, ScriptedEvaluator
from state import create_initial_state
from tree import InfoNode


def _obs_mem_belief(seed: int = 0):
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[0, 0] = 1
    grid[7, 7] = 2
    state = create_initial_state(grid)
    armies = np.asarray(state.armies, dtype=np.int32).copy()
    armies[0, 0] = 5
    state = state._replace(armies=armies)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(8, 8), obs)
    rng = np.random.default_rng(seed)
    belief = initialize_belief(
        obs,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=4, min_general_distance=3),
    )
    return obs, mem, belief, rng


def test_refresh_self_priors_from_full_network_prior():
    node = InfoNode(key=b"k", turn=0, memory_digest=b"m", obs_hash=b"o", history_digest=b"h")
    node.widen_self(PASS_INDEX, 1.0)
    assert float(node.prior[0]) == pytest.approx(1.0)

    full = np.zeros(N_ACTIONS, dtype=np.float64)
    full[PASS_INDEX] = 0.05
    expand = 378
    full[expand] = 0.90
    full[357] = 0.05
    node.widen_self(expand, 0.0)  # old broken widen path would keep mass 0
    node.widen_self(357, 0.0)
    node.refresh_self_priors(full)
    assert PASS_INDEX in node.actions and expand in node.actions
    pass_i = node.actions.index(PASS_INDEX)
    exp_i = node.actions.index(expand)
    assert float(node.prior[exp_i]) == pytest.approx(0.90)
    assert float(node.prior[pass_i]) == pytest.approx(0.05)


def test_expand_self_candidates_refreshes_after_stale_prior():
    """Stale first expand, then a second expand with live prior must unlock moves."""
    obs, mem, belief, rng = _obs_mem_belief()
    mask = legal_mask(obs, mem)
    legal_idx = [i for i in range(N_ACTIONS) if mask[i] and i != PASS_INDEX]
    assert len(legal_idx) >= 2, "test board must have multiple expands"
    stale_target = legal_idx[0]
    live_target = legal_idx[1]
    stale = np.zeros(N_ACTIONS, dtype=np.float64)
    stale[stale_target] = 1.0
    live = np.zeros(N_ACTIONS, dtype=np.float64)
    live[live_target] = 0.85
    live[stale_target] = 0.15

    ctl = SearchController(
        seat=0,
        evaluator=ScriptedEvaluator(prior=stale, value=0.0),
        config=SearchConfig(n_particles=4),
        rng=rng,
    )
    root = ctl.ensure_root(obs, mem, belief)
    assert PASS_INDEX not in root.actions
    assert stale_target in root.actions
    root.N = 16

    ctl.last_root_prior = live
    ctl._expand_self_candidates(root, obs, mem, live)
    assert live_target in root.actions
    ti = root.actions.index(live_target)
    si = root.actions.index(stale_target)
    assert float(root.prior[ti]) > float(root.prior[si])
    assert float(root.prior[ti]) == pytest.approx(0.85)


def test_select_path_widen_uses_last_root_prior_not_rebuilt_mass():
    obs, mem, belief, rng = _obs_mem_belief()
    mask = legal_mask(obs, mem)
    legal_idx = [i for i in range(N_ACTIONS) if mask[i] and i != PASS_INDEX]
    assert legal_idx, "test board must have a legal expand"
    first = legal_idx[0]
    target = legal_idx[1] if len(legal_idx) > 1 else legal_idx[0]
    stale = np.zeros(N_ACTIONS, dtype=np.float64)
    stale[first] = 1.0
    ctl = SearchController(
        seat=0,
        evaluator=ScriptedEvaluator(prior=stale, value=0.5),
        config=SearchConfig(n_particles=4, pending_batch=2, depth=2),
        rng=rng,
    )
    root = ctl.ensure_root(obs, mem, belief)
    root.N = 16

    live = np.zeros(N_ACTIONS, dtype=np.float64)
    live[target] = 1.0
    ctl.last_root_prior = live
    prior_for_self = np.asarray(ctl.last_root_prior, dtype=np.float64)
    ctl._widen_if_needed(root, obs, mem, prior_for_self, enemy=False)
    assert target in root.actions
    assert float(root.prior[root.actions.index(target)]) == pytest.approx(1.0)
