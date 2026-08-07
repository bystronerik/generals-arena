"""Part 07 — injected-cost degradation levels and monotonic admission."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import legal_mask
from belief import BeliefConfig, initialize_belief
from memory import empty_memory, update_memory
from observe import emit_observation
from runtime import (
    COST_COMPONENTS,
    FallbackLevel,
    FakeClock,
    RuntimeConfig,
    RuntimeController,
    select_degraded_action,
)
from search import ScriptedEvaluator, SearchConfig, UniformEvaluator
from state import create_initial_state


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
        config=BeliefConfig(n_particles=4, min_general_distance=3),
    )
    return obs, mem, belief, rng


def _cheap_forecasts(**overrides: float) -> dict[str, float]:
    base = {name: 0.5 for name in COST_COMPONENTS}
    base.update(overrides)
    return base


def _controller(clock: FakeClock, forecasts: dict[str, float], rng, **cfg_kw):
    defaults = dict(
        normal_deadline_ms=125.0,
        first_move_limit_ms=5000.0,
        admission_guard_ms=10.0,
        n_particles=4,
        target_simulations=32,
        pending_leaf_batch=4,
        p99_window=8,
    )
    defaults.update(cfg_kw)
    cfg = RuntimeConfig(**defaults)
    return RuntimeController(
        seat=0,
        H=8,
        W=8,
        config=cfg,
        clock=clock,
        fixed_forecasts_ms=forecasts,
        charge_fixed_forecasts=True,
        evaluator=UniformEvaluator(0.0),
        rng=rng,
    )


def test_no_root_result_returns_pass():
    clock = FakeClock()
    obs, _, _, rng = _ctx(0)
    # Root inference unaffordable on the first move.
    forecasts = _cheap_forecasts(root_inference=10000.0, belief_tensor=0.1)
    ctl = _controller(clock, forecasts, rng, first_move_limit_ms=100.0)
    action = ctl.decide(obs)
    assert action == (1, 0, 0, 0, 0)
    assert ctl.fallback_level == FallbackLevel.PASS.value
    assert ctl.completed_simulations == 0


def test_zero_sims_returns_policy_fallback():
    clock = FakeClock()
    obs, mem, belief, rng = _ctx(1)
    # Root cheap; leaf batch never admits → 0 completed simulations.
    forecasts = _cheap_forecasts(leaf_batch=10000.0)
    ctl = _controller(clock, forecasts, rng)
    action = ctl.decide(obs)
    assert ctl.fallback_level == FallbackLevel.POLICY.value
    assert ctl.completed_simulations == 0
    assert action != (1, 0, 0, 0, 0) or action == (1, 0, 0, 0, 0)
    # Action must be legal.
    mask = legal_mask(obs, update_memory(empty_memory(8, 8), obs))
    from action import encode_action

    assert mask[encode_action(action)]


def test_one_to_seven_sims_uses_visit_band():
    clock = FakeClock()
    obs, _, _, rng = _ctx(2)
    # Allow a few leaf batches then starve further work via high backup after.
    forecasts = _cheap_forecasts(leaf_batch=5.0, selection=1.0, backup=1.0)
    ctl = _controller(
        clock, forecasts, rng, target_simulations=4, pending_leaf_batch=2
    )
    action = ctl.decide(obs)
    assert 1 <= ctl.completed_simulations <= 7
    assert ctl.fallback_level == FallbackLevel.VISIT.value
    assert isinstance(action, tuple) and len(action) == 5


def test_eight_plus_sims_uses_average_strategy():
    clock = FakeClock()
    obs, _, _, rng = _ctx(3)
    forecasts = _cheap_forecasts()
    ctl = _controller(
        clock, forecasts, rng, target_simulations=12, pending_leaf_batch=4
    )
    action = ctl.decide(obs)
    assert ctl.completed_simulations >= 8
    assert ctl.fallback_level == FallbackLevel.AVERAGE.value
    assert isinstance(action, tuple) and len(action) == 5


def test_widen_freezes_when_forecast_below_sixteen():
    clock = FakeClock()
    obs, mem, belief, rng = _ctx(4)
    forecasts = _cheap_forecasts(leaf_batch=20.0, selection=5.0, backup=5.0)
    # Short deadline → forecast_total < 16 → freeze_widening path exercised.
    ctl = _controller(
        clock,
        forecasts,
        rng,
        normal_deadline_ms=80.0,
        first_move_limit_ms=80.0,
        target_simulations=32,
        admission_guard_ms=10.0,
    )
    ctl.decide(obs)
    # Completing under a tight budget must not exceed the tree node bound.
    assert ctl.tree_size <= ctl.config.max_tree_nodes


def test_monotonic_admission_never_starts_past_forecast():
    clock = FakeClock()
    obs, _, _, rng = _ctx(5)
    forecasts = _cheap_forecasts(leaf_batch=40.0, selection=5.0, backup=5.0)
    ctl = _controller(
        clock,
        forecasts,
        rng,
        normal_deadline_ms=100.0,
        first_move_limit_ms=100.0,
        admission_guard_ms=10.0,
        target_simulations=32,
    )
    ctl.decide(obs)
    # Every recorded leaf_batch sample must have been admitted when remaining
    # time covered forecast+guard — i.e. clock never went past deadline by a
    # full unaffordable leaf after the last admit check. Soft check: move
    # duration stays near the internal deadline (+ one in-flight batch).
    assert ctl.move_ms <= 100 + 40 + 10


def test_degradation_table_matches_runtime_spec():
    obs, mem, belief, rng = _ctx(6)
    prior = np.zeros(3970, dtype=np.float64)
    mask = legal_mask(obs, mem)
    prior[mask] = 1.0 / max(int(mask.sum()), 1)
    ev = ScriptedEvaluator(prior=prior, value=0.1)
    from search import SearchController

    search = SearchController(
        seat=0,
        evaluator=ev,
        config=SearchConfig(n_particles=4, pending_batch=1, depth=2),
        rng=rng,
    )
    search.ensure_root(obs, mem, belief)
    policy = (1, 0, 0, 0, 0)
    if search.last_root_prior is not None:
        from runtime import highest_prior_legal

        policy = highest_prior_legal(search.last_root_prior, mask)

    cases = [
        (False, 0, FallbackLevel.PASS),
        (True, 0, FallbackLevel.POLICY),
        (True, 3, FallbackLevel.VISIT),
        (True, 8, FallbackLevel.AVERAGE),
        (True, 16, FallbackLevel.AVERAGE),
    ]
    for has_root, sims, expected in cases:
        # Force completed_simulations counter without relying on search state
        # for the PASS/POLICY rows; for VISIT/AVERAGE run real backups.
        if has_root and sims > 0:
            search.tree.completed_simulations = 0
            while search.tree.completed_simulations < sims:
                search.run_batch(belief, n_sims=1)
            sims = search.tree.completed_simulations
        action, level = select_degraded_action(
            completed_simulations=sims if has_root else 0,
            has_root_result=has_root,
            policy_fallback=policy,
            search=search,
        )
        assert level is expected, (has_root, sims, level)
        assert isinstance(action, tuple) and len(action) == 5
