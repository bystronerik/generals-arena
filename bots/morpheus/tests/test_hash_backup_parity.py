"""Part 09a Phase 4 — fixed-seed search parity after hashing/backup speedups."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import PASS_INDEX, legal_mask
from belief import BeliefConfig, initialize_belief
from hashing import (
    enemy_info_hash,
    enemy_info_hash_prehashed,
    info_state_key,
    info_state_key_prehashed,
    memory_digest,
)
from memory import empty_memory, update_memory
from observe import emit_observation, observation_hash
from search import SearchConfig, SearchController, ScriptedEvaluator, UniformEvaluator
from state import create_initial_state
from tactics import mandatory_action_indices, policy_ordered_candidates


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
    return obs, mem, belief


def _run(seed: int, sims: int, *, scripted: bool):
    obs, mem, belief = _ctx(seed)
    mask = legal_mask(obs, mem)
    if scripted:
        prior = np.zeros(mask.shape, dtype=np.float64)
        legal = np.flatnonzero(mask)
        for i, idx in enumerate(legal):
            prior[idx] = float(len(legal) - i)
        ev = ScriptedEvaluator(prior=prior, value=0.1)
    else:
        ev = UniformEvaluator(0.0)
    ctl = SearchController(
        seat=0,
        evaluator=ev,
        config=SearchConfig(depth=4, pending_batch=4, n_particles=4),
        rng=np.random.default_rng(seed),
    )
    ctl.ensure_root(obs, mem, belief)
    while ctl.tree.completed_simulations < sims:
        n = min(4, sims - ctl.tree.completed_simulations)
        ctl.run_batch(belief, n_sims=n)
    root = ctl.tree.root
    assert root is not None
    visits = sorted(float(t.visits.sum()) for t in root.enemy_tables.values())
    return {
        "N": int(root.N),
        "actions": list(root.actions),
        "avg_strategy": [float(x) for x in root.avg_strategy],
        "best_action": list(ctl.best_action_or_pass()),
        "completed": int(ctl.tree.completed_simulations),
        "n_enemy_tables": len(root.enemy_tables),
        "table_visits_sorted": visits,
    }


# Baseline captured on Phase-3 HEAD before Phase-4 edits.
_BASELINES = {
    ("scripted", 7, 16): {
        "N": 16,
        "actions": [3969],
        "avg_strategy": [16.0],
        "best_action": [1, 0, 0, 0, 0],
        "completed": 16,
        "n_enemy_tables": 3,
        "table_visits_sorted": [3.0, 6.0, 7.0],
    },
    ("scripted", 11, 8): {
        "N": 8,
        "actions": [3969],
        "avg_strategy": [8.0],
        "best_action": [1, 0, 0, 0, 0],
        "completed": 8,
        "n_enemy_tables": 2,
        "table_visits_sorted": [3.0, 5.0],
    },
    ("uniform", 3, 12): {
        "N": 12,
        "actions": [3969],
        "avg_strategy": [12.0],
        "best_action": [1, 0, 0, 0, 0],
        "completed": 12,
        "n_enemy_tables": 3,
        "table_visits_sorted": [2.0, 5.0, 5.0],
    },
}


@pytest.mark.parametrize(
    "kind,seed,sims",
    [
        ("scripted", 7, 16),
        ("scripted", 11, 8),
        ("uniform", 3, 12),
    ],
)
def test_fixed_seed_root_stats_match_pre_phase4(kind, seed, sims):
    got = _run(seed, sims, scripted=(kind == "scripted"))
    assert got == _BASELINES[(kind, seed, sims)]


def test_prehashed_info_state_key_matches():
    obs, mem, _ = _ctx(0)
    obs_h = observation_hash(obs)
    mem_d = memory_digest(mem)
    assert info_state_key(obs.turn, mem, obs_h) == info_state_key_prehashed(
        obs.turn, mem_d, obs_h
    )
    assert enemy_info_hash(obs, mem) == enemy_info_hash_prehashed(obs_h, mem_d)


def test_policy_ordered_sorts_only_legal():
    mask = np.zeros(3970, dtype=bool)
    mask[PASS_INDEX] = True
    mask[10] = True
    mask[20] = True
    mask[30] = True
    prior = np.zeros(3970, dtype=np.float64)
    prior[10] = 0.1
    prior[20] = 0.9
    prior[30] = 0.5
    prior[PASS_INDEX] = 0.0
    cands = policy_ordered_candidates(
        prior, mask, mandatory=[PASS_INDEX], limit=4
    )
    assert cands[0] == PASS_INDEX
    assert cands[1:] == [20, 30, 10]


def test_enemy_weight_cache_stable_across_backups():
    obs, mem, belief = _ctx(5)
    ctl = SearchController(
        seat=0,
        evaluator=UniformEvaluator(0.0),
        config=SearchConfig(depth=2, pending_batch=2, n_particles=4),
        rng=np.random.default_rng(5),
    )
    ctl.ensure_root(obs, mem, belief)
    ctl.run_batch(belief, n_sims=2)
    root = ctl.tree.root
    assert root is not None
    version = root.reservoir.version
    h1, w1 = ctl.tree.enemy_weights(root)
    h2, w2 = ctl.tree.enemy_weights(root)
    assert h1 == h2
    assert np.allclose(w1, w2)
    assert root._enemy_hash_version == version
    # Second call must reuse the cache (same list object).
    assert root._enemy_hash_cache is not None


def test_pending_leaf_batch_eight_fixed_seed_stable():
    """Pending leaf batch 8 completes the floor and repeats under a fixed seed."""

    def _run():
        obs, mem, belief = _ctx(9)
        ctl = SearchController(
            seat=0,
            evaluator=UniformEvaluator(0.0),
            config=SearchConfig(depth=4, pending_batch=8, n_particles=4),
            rng=np.random.default_rng(9),
        )
        ctl.ensure_root(obs, mem, belief)
        while ctl.tree.completed_simulations < 8:
            remaining = 8 - ctl.tree.completed_simulations
            ctl.run_batch(belief, n_sims=remaining)
        return {
            "completed": int(ctl.tree.completed_simulations),
            "best_action": list(ctl.best_action_or_pass()),
            "N": int(ctl.tree.root.N),
        }

    first = _run()
    second = _run()
    assert first["completed"] == 8
    assert first == second
