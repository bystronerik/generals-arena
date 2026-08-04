"""Part 07 — p99 estimator, admission, work order, and probe passivity."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import PASS_INDEX, legal_mask
from belief import BeliefConfig, initialize_belief
from memory import empty_memory, update_memory
from observe import emit_observation
from runtime import (
    COST_COMPONENTS,
    DEFAULT_OFFLINE_P99_MS,
    FallbackLevel,
    FakeClock,
    NearestRankP99Estimator,
    RuntimeConfig,
    RuntimeController,
    highest_prior_legal,
    nearest_rank_p99,
    select_degraded_action,
)
from search import ScriptedEvaluator, SearchController, UniformEvaluator
from state import create_initial_state


def _board_obs(seed: int = 0, size: int = 8):
    grid = np.zeros((size, size), dtype=np.int32)
    grid[0, 0] = 1
    grid[size - 1, size - 1] = 2
    state = create_initial_state(grid)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(size, size), obs)
    rng = np.random.default_rng(seed)
    belief = initialize_belief(
        obs,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=4, min_general_distance=3),
    )
    return state, obs, mem, belief, rng


def test_nearest_rank_p99_deterministic():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    # n=10, ceil(0.99*10)=10 → 100
    assert nearest_rank_p99(samples) == 100.0
    assert nearest_rank_p99([10.0]) == 10.0


def test_p99_warmup_uses_max_of_offline_and_locals():
    est = NearestRankP99Estimator(window=4, offline_p99_ms=50.0)
    assert est.forecast() == 50.0
    est.observe(10.0)
    assert est.forecast() == 50.0  # offline still dominates
    est.observe(80.0)
    assert est.forecast() == 80.0  # local max wins
    est.observe(20.0)
    est.observe(30.0)
    assert est.warmed_up
    # window full: nearest-rank p99 of [10,80,20,30]
    assert est.forecast() == nearest_rank_p99([10.0, 80.0, 20.0, 30.0])


def test_p99_window_evicts_oldest():
    est = NearestRankP99Estimator(window=3, offline_p99_ms=1.0)
    for x in (1.0, 2.0, 100.0):
        est.observe(x)
    assert est.forecast() == 100.0
    est.observe(3.0)  # drops 1.0
    assert set(est._samples) == {2.0, 100.0, 3.0}


def test_admission_rejects_when_forecast_plus_guard_exceeds_remaining():
    clock = FakeClock(0.0)
    cfg = RuntimeConfig(
        normal_deadline_ms=100.0,
        admission_guard_ms=10.0,
        offline_p99_ms={**DEFAULT_OFFLINE_P99_MS, "leaf_batch": 50.0},
    )
    ctl = RuntimeController(
        seat=0,
        H=8,
        W=8,
        config=cfg,
        clock=clock,
        fixed_forecasts_ms={"leaf_batch": 50.0, "selection": 1.0, "backup": 1.0},
        charge_fixed_forecasts=True,
        evaluator=UniformEvaluator(0.0),
    )
    ctl._turn_start = 0.0
    deadline = 0.100  # 100 ms absolute
    assert ctl.can_admit("leaf_batch", deadline)  # 50+10 <= 100
    clock.advance(45.0)  # 55 ms left
    assert not ctl.can_admit("leaf_batch", deadline)  # 50+10 > 55


def test_fallback_order_pass_then_policy_then_search():
    _, obs, mem, belief, rng = _board_obs()
    prior = np.zeros(3970, dtype=np.float64)
    prior[PASS_INDEX] = 0.1
    # Prefer a non-pass legal action if any exist; else pass.
    mask = legal_mask(obs, mem)
    legal_idx = int(np.flatnonzero(mask)[0])
    prior[legal_idx] = 0.9
    ev = ScriptedEvaluator(prior=prior, value=0.0)
    search = SearchController(
        seat=0, evaluator=ev, rng=rng, config=__import__("search").SearchConfig(
            n_particles=4, pending_batch=1, depth=2
        )
    )

    action, level = select_degraded_action(
        completed_simulations=0,
        has_root_result=False,
        policy_fallback=None,
        search=search,
    )
    assert action == (1, 0, 0, 0, 0) and level is FallbackLevel.PASS

    policy = highest_prior_legal(prior, mask)
    action, level = select_degraded_action(
        completed_simulations=0,
        has_root_result=True,
        policy_fallback=policy,
        search=search,
    )
    assert action == policy and level is FallbackLevel.POLICY

    search.ensure_root(obs, mem, belief)
    for _ in range(3):
        search.run_batch(belief, n_sims=1)
    assert 1 <= search.tree.completed_simulations <= 7
    action, level = select_degraded_action(
        completed_simulations=search.tree.completed_simulations,
        has_root_result=True,
        policy_fallback=policy,
        search=search,
    )
    assert level is FallbackLevel.VISIT
    assert action == search.best_action_by_visits()

    while search.tree.completed_simulations < 8:
        search.run_batch(belief, n_sims=1)
    action, level = select_degraded_action(
        completed_simulations=search.tree.completed_simulations,
        has_root_result=True,
        policy_fallback=policy,
        search=search,
    )
    assert level is FallbackLevel.AVERAGE
    assert action == search.best_action()


def test_work_order_stores_pass_before_optional_work():
    """Fallback is pass until root succeeds; partial work never clears it early."""
    clock = FakeClock(0.0)
    # Make every optional component too expensive → stay on pass.
    heavy = {name: 1000.0 for name in COST_COMPONENTS}
    ctl = RuntimeController(
        seat=0,
        H=8,
        W=8,
        config=RuntimeConfig(
            normal_deadline_ms=125.0,
            first_move_limit_ms=125.0,
            admission_guard_ms=10.0,
            n_particles=4,
        ),
        clock=clock,
        fixed_forecasts_ms=heavy,
        charge_fixed_forecasts=True,
        evaluator=UniformEvaluator(0.0),
        rng=np.random.default_rng(0),
    )
    _, obs, _, _, _ = _board_obs()
    # Force setup without using admitted path: decide still begins with PASS.
    action = ctl.decide(obs)
    assert action == (1, 0, 0, 0, 0)
    assert ctl.fallback_level == FallbackLevel.PASS.value
    assert ctl.completed_simulations == 0


def test_partial_simulation_does_not_change_statistics():
    clock = FakeClock(0.0)
    _, obs, mem, belief, rng = _board_obs(1)
    # Cheap selection, expensive leaf → select then discard.
    forecasts = {
        **{n: 0.1 for n in COST_COMPONENTS},
        "belief_tensor": 0.1,
        "belief_proposal": 0.1,
        "particle_transitions": 0.1,
        "root_inference": 0.1,
        "selection": 0.1,
        "leaf_batch": 80.0,
        "backup": 0.1,
        "reply": 0.1,
        "hashing": 0.1,
    }
    ctl = RuntimeController(
        seat=0,
        H=8,
        W=8,
        config=RuntimeConfig(
            normal_deadline_ms=100.0,
            first_move_limit_ms=5000.0,
            admission_guard_ms=10.0,
            n_particles=4,
            target_simulations=8,
            pending_leaf_batch=4,
        ),
        clock=clock,
        fixed_forecasts_ms=forecasts,
        charge_fixed_forecasts=True,
        evaluator=UniformEvaluator(0.0),
        rng=rng,
    )
    # First move: setup + root with cheap forecasts.
    action = ctl.decide(obs)
    sims_after_first = ctl.completed_simulations
    # Advance clock so leaf_batch no longer fits; next turn's search selects
    # nothing that backs up if leaf cannot admit.
    # Rebuild a second observation by reusing first (protocol allows).
    clock.advance(0.0)
    # Spend budget so only selection fits, not leaf_batch (80+10).
    # After root+belief on turn 2, remaining must be < 90.
    forecasts2 = dict(forecasts)
    forecasts2["leaf_batch"] = 1000.0
    ctl.fixed_forecasts_ms = forecasts2
    ctl._last_action = action
    # Manually run search discard path.
    ctl.search.ensure_root(obs, mem, belief)
    before = ctl.search.tree.completed_simulations
    # Snapshot root N
    root_n = ctl.search.tree.root.N if ctl.search.tree.root else 0
    ctl._turn_start = clock()
    deadline = clock() + 0.050  # 50 ms — leaf cannot admit
    ctl._run_search(belief, deadline)
    assert ctl.search.tree.completed_simulations == before
    assert (ctl.search.tree.root.N if ctl.search.tree.root else 0) == root_n
    assert sims_after_first >= 0


def test_tree_bound_holds():
    _, obs, mem, belief, rng = _board_obs(2)
    ctl = RuntimeController(
        seat=0,
        H=8,
        W=8,
        config=RuntimeConfig(max_tree_nodes=32, n_particles=4, target_simulations=16),
        evaluator=UniformEvaluator(0.0),
        rng=rng,
    )
    ctl.first_move_setup(obs)
    ctl.search.ensure_root(obs, mem, belief)
    for _ in range(20):
        ctl.search.run_batch(belief, n_sims=4)
    assert len(ctl.search.tree.nodes) <= 32


def test_probe_is_passive_and_keys_are_declared():
    from arena.instrument.probes import load_probe, probe_extras
    from arena.records.telemetry_schema import validate_extras
    from pathlib import Path

    bot_dir = Path(__file__).resolve().parents[1]
    probe = load_probe(bot_dir)
    assert probe is not None

    class _Stub:
        move_ms = 12
        search_iters = 3
        completed_simulations = 3
        forward_equivalents = 5
        belief_ess = 4000
        recovery = 0
        tree_size = 7
        fallback_level = "visit"
        cost_belief_ms = 4
        cost_root_ms = 2
        cost_search_ms = 6
        cost_reply_ms = 1

    extras = probe_extras(probe, _Stub())
    validated = validate_extras(extras)
    assert validated["completed_simulations"] == 3
    assert validated["fallback_level"] == "visit"
    # Probe must not import into the agent closure (fingerprint rule).
    agent_src = (bot_dir / "agent.py").read_text(encoding="utf-8")
    runtime_src = (bot_dir / "runtime.py").read_text(encoding="utf-8")
    assert "import probe" not in agent_src
    assert "import probe" not in runtime_src
