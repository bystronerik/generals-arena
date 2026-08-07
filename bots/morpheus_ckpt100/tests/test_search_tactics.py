"""Part 06 — tactical suite selects from the correct simultaneous matrix."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import PASS_INDEX, legal_mask
from belief import BeliefConfig, initialize_belief
from matrix import (
    accumulate_average_strategy,
    effective_q,
    matrix_utilities,
    mixed_strategy,
    normalize_average_strategy,
    regret_plus_update,
)
from memory import empty_memory, update_memory
from observe import emit_observation
from search import SearchConfig, SearchController, UniformEvaluator
from state import create_initial_state
from tactics import (
    load_tactical_suite,
    mandatory_action_indices,
    policy_ordered_candidates,
    suite_path,
)


def _case_best_self(case: dict) -> int:
    names = case["self_actions"]
    return names.index(case["expected_best_self"])


def test_tactical_suite_matrix_argmax():
    """Each suite case's expected row is best against the enemy mixed strategy."""
    cases = load_tactical_suite(suite_path())
    assert len(cases) >= 5
    kinds = {c["name"] for c in cases}
    assert kinds >= {"chase", "reinforcement", "castle", "mutual_capture", "deathtouch"}

    for case in cases:
        q = np.asarray(case["q"], dtype=np.float64)
        visits = np.asarray(case["visits"], dtype=np.float64)
        prior_a = np.asarray(case["prior_self"], dtype=np.float64)
        prior_b = np.asarray(case["prior_enemy"], dtype=np.float64)
        q_eff = effective_q(visits, q, float(case.get("first_play", 0.0)))
        # Enemy plays its prior (zero regrets).
        sigma_b = mixed_strategy(np.zeros_like(prior_b), prior_b, n=0)
        sigma_a = mixed_strategy(np.zeros_like(prior_a), prior_a, n=0)
        u_self, _, _ = matrix_utilities(sigma_a, sigma_b, q_eff)
        # With positive visits everywhere, select by u_self (matrix row value).
        best = int(np.argmax(u_self))
        assert best == _case_best_self(case), (
            f"{case['name']}: expected {_case_best_self(case)} got {best} "
            f"u_self={u_self}"
        )


def test_tactical_suite_drives_average_strategy():
    """Repeated regret-plus updates on the fixture Q concentrate on the best row."""
    cases = load_tactical_suite(suite_path())
    for case in cases:
        q = np.asarray(case["q"], dtype=np.float64)
        visits = np.asarray(case["visits"], dtype=np.float64)
        prior_a = np.asarray(case["prior_self"], dtype=np.float64)
        prior_b = np.asarray(case["prior_enemy"], dtype=np.float64)
        regret = np.zeros_like(prior_a)
        avg = np.zeros_like(prior_a)
        q_eff = effective_q(visits, q, float(case.get("first_play", 0.0)))
        for n in range(1, 64):
            sigma_a = mixed_strategy(regret, prior_a, n=n)
            sigma_b = mixed_strategy(np.zeros_like(prior_b), prior_b, n=n)
            u_self, _, v = matrix_utilities(sigma_a, sigma_b, q_eff)
            regret = regret_plus_update(regret, u_self, v, maximizing=True)
            avg = accumulate_average_strategy(avg, sigma_a)
        s = normalize_average_strategy(avg)
        assert int(np.argmax(s)) == _case_best_self(case), f"{case['name']}: S_A={s}"



def test_mandatory_candidates_include_pass_and_enemy_interaction():
    # Adjacent generals so the enemy is visible and capturable.
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[2, 1] = 1
    grid[2, 2] = 2
    state = create_initial_state(grid)
    armies = state.armies.copy()
    armies[2, 1] = 10
    armies[2, 2] = 10
    state = state._replace(armies=armies)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(5, 5), obs)
    mandatory = mandatory_action_indices(obs, mem)
    assert PASS_INDEX in mandatory
    assert any(i != PASS_INDEX for i in mandatory)
    mask = legal_mask(obs, mem)
    prior = np.zeros(mask.shape)
    prior[mask] = 1.0
    cands = policy_ordered_candidates(prior, mask, mandatory=mandatory, limit=8)
    assert PASS_INDEX in cands


def test_search_runs_batch_without_partial_root_stats():
    grid = np.zeros((8, 8), dtype=np.int32)
    grid[0, 0] = 1
    grid[7, 7] = 2
    state = create_initial_state(grid)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(8, 8), obs)
    rng = np.random.default_rng(0)
    belief = initialize_belief(
        obs, seat=0, rng=rng, config=BeliefConfig(n_particles=4, min_general_distance=5)
    )
    ctl = SearchController(
        seat=0,
        evaluator=UniformEvaluator(value=0.0),
        config=SearchConfig(depth=2, pending_batch=2, n_particles=4),
        rng=rng,
    )
    ctl.ensure_root(obs, mem, belief)
    assert ctl.tree.completed_simulations == 0
    n = ctl.run_batch(belief, n_sims=2)
    assert n == 2
    assert ctl.tree.completed_simulations == 2
    action = ctl.best_action_or_pass()
    assert len(action) == 5
