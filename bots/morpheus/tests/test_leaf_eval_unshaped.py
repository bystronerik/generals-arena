"""Leaf evaluations are root-perspective but never shaped.

Regression for the ``from_root`` conflation: every leaf paid the heuristic
blend (half of all search time) and each leaf call clobbered
``last_unshaped_prior``, corrupting the "who's deciding" probe fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from belief import BeliefConfig, BeliefState, initialize_belief
from evaluator import ShapedUniformEvaluator
from memory import VisibleMemory, empty_memory, update_memory
from observe import emit_observation
from search import SearchConfig, SearchController, UniformEvaluator
from state import create_initial_state


@dataclass
class FlagRecordingEvaluator:
    """Records the (from_root, shape) flags of every evaluation."""

    flags: list[tuple[bool, bool]] = field(default_factory=list)
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
        self.flags.append((bool(from_root), bool(shape)))
        return self.inner.evaluate(obs, memory, belief, from_root=from_root)

    def evaluate_many(self, items):
        return [
            self.evaluate(obs, mem, blf, from_root=fr, shape=sh)
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


def test_only_root_evaluations_request_shaping():
    obs, mem, belief, rng = _ctx(0)
    ev = FlagRecordingEvaluator()
    ctl = SearchController(
        seat=0,
        evaluator=ev,
        config=SearchConfig(depth=3, pending_batch=2, n_particles=4),
        rng=rng,
    )
    ctl.ensure_root(obs, mem, belief)
    assert ev.flags == [(True, True)]

    ctl.run_batch(belief, n_sims=4)
    leaf_flags = ev.flags[1:]
    assert leaf_flags, "search must have evaluated at least one non-root item"
    # Leaves: root perspective, unshaped. Enemy priors go through
    # policy_priors_many / from_root=False and never set shape.
    for from_root, shape in leaf_flags:
        assert shape is False
        if from_root:
            continue  # leaf batch
        # enemy-prior fallback path: enemy perspective, unshaped
        assert from_root is False


def test_last_unshaped_prior_survives_leaf_batches():
    obs, mem, belief, rng = _ctx(1)
    ev = ShapedUniformEvaluator()
    ctl = SearchController(
        seat=0,
        evaluator=ev,
        config=SearchConfig(depth=3, pending_batch=2, n_particles=4),
        rng=rng,
    )
    ctl.ensure_root(obs, mem, belief)
    root_unshaped = np.array(ev.last_unshaped_prior, copy=True)
    ctl.run_batch(belief, n_sims=4)
    assert ev.last_unshaped_prior is not None
    np.testing.assert_array_equal(ev.last_unshaped_prior, root_unshaped)
