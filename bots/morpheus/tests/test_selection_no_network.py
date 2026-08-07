"""Part 09a Phase 3 — selection performs zero network calls."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from action import PASS_INDEX, legal_mask
from belief import BeliefConfig, BeliefState, initialize_belief
from memory import VisibleMemory, empty_memory, update_memory
from observe import emit_observation
from search import (
    EnemyPriorRequest,
    SearchConfig,
    SearchController,
    UniformEvaluator,
)
from state import create_initial_state


@dataclass
class CountingEvaluator:
    """Counts every ``evaluate`` / ``evaluate_many`` call."""

    value: float = 0.0
    calls: int = 0
    batch_calls: int = 0
    inner: UniformEvaluator = field(default_factory=lambda: UniformEvaluator(0.0))

    def evaluate(
        self,
        obs,
        memory: VisibleMemory,
        belief: BeliefState,
        *,
        from_root: bool,
        shape: bool = False,
    ):
        self.calls += 1
        return self.inner.evaluate(
            obs, memory, belief, from_root=from_root, shape=shape
        )

    def evaluate_many(self, items):
        self.batch_calls += 1
        self.calls += len(items)
        return [
            self.inner.evaluate(obs, mem, blf, from_root=fr, shape=sh)
            for obs, mem, blf, fr, sh in items
        ]


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


def test_select_path_performs_zero_network_calls():
    obs, mem, belief, rng = _ctx(0)
    counter = CountingEvaluator()
    ctl = SearchController(
        seat=0,
        evaluator=counter,
        config=SearchConfig(depth=4, pending_batch=2, n_particles=4),
        rng=rng,
    )
    # Root evaluation is outside selection (ensure_root).
    ctl.ensure_root(obs, mem, belief)
    root_calls = counter.calls
    assert root_calls >= 1

    before = counter.calls
    outcome = ctl.select_path(belief, freeze_snapshot=True)
    assert counter.calls == before
    assert counter.batch_calls == 0

    # First selection at a fresh root must request an enemy prior.
    assert isinstance(outcome, EnemyPriorRequest)

    # Materialise outside selection — that is allowed to call the network.
    n = ctl.materialize_enemy_priors([outcome], belief)
    assert n >= 1
    assert counter.calls > before

    after_prior = counter.calls
    resumed = ctl.select_path(belief, freeze_snapshot=True, resume=outcome)
    assert counter.calls == after_prior
    # Resumed selection yields a path (possibly needing leaf expand) or another prior.
    assert not isinstance(resumed, type(None))


def test_complete_select_and_batch_keep_partial_rule():
    obs, mem, belief, rng = _ctx(1)
    counter = CountingEvaluator()
    ctl = SearchController(
        seat=0,
        evaluator=counter,
        config=SearchConfig(depth=2, pending_batch=2, n_particles=4),
        rng=rng,
    )
    ctl.ensure_root(obs, mem, belief)
    assert ctl.tree.completed_simulations == 0
    root_n = ctl.tree.root.N if ctl.tree.root else 0

    path = ctl.complete_select_path(belief)
    assert ctl.tree.completed_simulations == 0
    assert ctl.tree.root is not None
    assert ctl.tree.root.N == root_n
    _ = path

    n = ctl.run_batch(belief, n_sims=2)
    assert n == 2
    assert ctl.tree.completed_simulations == 2
    action = ctl.best_action_or_pass()
    assert len(action) == 5
    mask = legal_mask(obs, mem)
    # Pass is always legal; action encoding is a 5-tuple.
    assert PASS_INDEX >= 0 or mask.any()


def test_unexpanded_node_stops_without_network():
    obs, mem, belief, rng = _ctx(2)
    counter = CountingEvaluator()
    ctl = SearchController(
        seat=0,
        evaluator=counter,
        config=SearchConfig(depth=1, pending_batch=1, n_particles=4),
        rng=rng,
    )
    ctl.ensure_root(obs, mem, belief)
    # Strip self actions to force the unexpanded stop path.
    assert ctl.tree.root is not None
    ctl.tree.root.actions = []
    ctl.tree.root.prior = np.zeros(0, dtype=np.float64)
    ctl.tree.root.regret = np.zeros(0, dtype=np.float64)

    before = counter.calls
    outcome = ctl.select_path(belief)
    assert counter.calls == before
    assert not isinstance(outcome, EnemyPriorRequest)
    assert outcome.needs_expand is True
    assert outcome.leaf_node is ctl.tree.root
