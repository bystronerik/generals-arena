"""Cross-turn enemy-prior cache: no repeat forwards for a known info hash.

The tree is rebuilt on most turns (exact obs-hash reuse is rare), so enemy
priors were re-materialized every turn (~36 ms/turn measured). The controller
cache is content-addressed by the enemy info hash and survives tree resets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
    enemy_calls: int = 0
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
        if not from_root:
            self.enemy_calls += 1
        return self.inner.evaluate(obs, memory, belief, from_root=from_root)


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


def _drain_priors(ctl: SearchController, belief) -> None:
    """Materialize every enemy prior the root's particles can request."""
    for _ in range(16):
        outcome = ctl.select_path(belief)
        if isinstance(outcome, EnemyPriorRequest):
            ctl.materialize_enemy_priors([outcome], belief)
        else:
            break


def test_cache_survives_tree_reset_and_skips_repeat_forwards():
    obs, mem, belief, rng = _ctx(0)
    ev = CountingEvaluator()
    ctl = SearchController(
        seat=0,
        evaluator=ev,
        config=SearchConfig(depth=2, pending_batch=2, n_particles=4),
        rng=rng,
    )
    ctl.ensure_root(obs, mem, belief)
    for _ in range(8):
        _drain_priors(ctl, belief)
    assert ev.enemy_calls >= 1
    assert len(ctl.enemy_prior_cache) >= 1

    # Simulate the common turn boundary: exact reuse failed, tree dropped.
    calls_before = ev.enemy_calls
    ctl.tree.clear()
    ctl.ensure_root(obs, mem, belief)
    for _ in range(8):
        outcome = ctl.select_path(belief)
        # Every hash was cached above: selection installs tables from the
        # cache and never has to pause for materialization.
        assert not isinstance(outcome, EnemyPriorRequest)
    assert ev.enemy_calls == calls_before
    assert ctl.enemy_prior_cache_hits >= 1


def test_cache_is_bounded_lru():
    obs, mem, belief, rng = _ctx(1)
    ctl = SearchController(
        seat=0,
        evaluator=CountingEvaluator(),
        config=SearchConfig(
            depth=2, pending_batch=2, n_particles=4, enemy_prior_cache_size=2
        ),
        rng=rng,
    )
    for i in range(5):
        ctl._store_enemy_prior(bytes([i]) * 8, np.full(3970, float(i)))
    assert len(ctl.enemy_prior_cache) == 2
    # Oldest entries evicted; newest kept.
    assert bytes([4]) * 8 in ctl.enemy_prior_cache
    assert bytes([0]) * 8 not in ctl.enemy_prior_cache
