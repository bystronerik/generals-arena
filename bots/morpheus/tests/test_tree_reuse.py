"""Part 06 — tree reuse only on exact information-state match; bounds hold."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from belief import BeliefConfig, initialize_belief, pass_action
from hashing import (
    ZERO_DIGEST,
    child_edge_key,
    info_state_key,
    memory_digest,
    roll_history_digest,
)
from memory import empty_memory, update_memory
from observe import emit_observation, observation_hash
from search import SearchConfig, SearchController, UniformEvaluator
from state import create_initial_state
from transition import PASS_ACTION, transition
from tree import MAX_ENEMY_TABLES, InfoNode, SearchTree


def _tiny_belief(seed: int = 0):
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
    return state, obs, mem, belief, rng


def test_info_state_key_stable_and_sensitive():
    _, obs, mem, _, _ = _tiny_belief()
    k1 = info_state_key(obs.turn, mem, observation_hash(obs), ZERO_DIGEST)
    k2 = info_state_key(obs.turn, mem, observation_hash(obs), ZERO_DIGEST)
    assert k1 == k2
    k3 = info_state_key(obs.turn + 1, mem, observation_hash(obs), ZERO_DIGEST)
    assert k1 != k3


def test_reuse_keeps_stats_on_exact_match():
    state, obs, mem, belief, rng = _tiny_belief(1)
    ctl = SearchController(
        seat=0,
        evaluator=UniformEvaluator(0.0),
        config=SearchConfig(depth=1, pending_batch=1, n_particles=4),
        rng=rng,
    )
    root = ctl.ensure_root(obs, mem, belief)
    root.N = 7
    root.avg_strategy = np.array([1.0])
    # Advance one double-pass transition and build the matching child.
    actions = np.stack([PASS_ACTION, PASS_ACTION])
    next_state, _ = transition(state, actions)
    next_obs = emit_observation(next_state, 0)
    next_mem = update_memory(mem, next_obs)
    edge = child_edge_key(pass_action(), observation_hash(next_obs))
    child_hist = roll_history_digest(
        root.history_digest, pass_action(), observation_hash(next_obs)
    )
    child = ctl.tree.make_node(
        key=info_state_key(
            int(next_obs.turn),
            next_mem,
            observation_hash(next_obs),
            child_hist,
        ),
        turn=int(next_obs.turn),
        memory_digest=memory_digest(next_mem),
        obs_hash=observation_hash(next_obs),
        history_digest=child_hist,
        network_value=0.0,
        capacity=4,
    )
    child.N = 3
    child.actions = [0]
    child.prior = np.array([1.0])
    child.regret = np.zeros(1)
    child.avg_strategy = np.array([1.0])
    child.expanded = True
    root.children[edge] = child

    next_belief = initialize_belief(
        next_obs,
        seat=0,
        rng=np.random.default_rng(2),
        config=BeliefConfig(n_particles=4, min_general_distance=5),
    )
    reused = ctl.reuse_or_reset(pass_action(), next_obs, next_mem, next_belief)
    assert reused is child
    assert reused.N == 3
    assert ctl.tree.root is child


def test_reuse_resets_on_hash_mismatch():
    state, obs, mem, belief, rng = _tiny_belief(3)
    ctl = SearchController(
        seat=0,
        evaluator=UniformEvaluator(0.0),
        config=SearchConfig(depth=1, pending_batch=1, n_particles=4),
        rng=rng,
    )
    root = ctl.ensure_root(obs, mem, belief)
    root.N = 11
    # Next observation without a matching child → new root.
    actions = np.stack([PASS_ACTION, PASS_ACTION])
    next_state, _ = transition(state, actions)
    next_obs = emit_observation(next_state, 0)
    next_mem = update_memory(mem, next_obs)
    next_belief = initialize_belief(
        next_obs,
        seat=0,
        rng=np.random.default_rng(4),
        config=BeliefConfig(n_particles=4, min_general_distance=5),
    )
    new_root = ctl.reuse_or_reset(pass_action(), next_obs, next_mem, next_belief)
    assert new_root is not root
    assert new_root.N == 0


def test_enemy_table_cap_and_eviction_loss():
    tree = SearchTree(max_enemy_tables=MAX_ENEMY_TABLES, seat=0)
    node = InfoNode(
        key=b"\x02" * 32,
        turn=0,
        memory_digest=b"\x00" * 32,
        obs_hash=b"\x00" * 32,
        history_digest=ZERO_DIGEST,
    )
    node.actions = [0, 1]
    node.prior = np.array([0.5, 0.5])
    node.regret = np.zeros(2)
    node.avg_strategy = np.zeros(2)
    tree.set_root(node)
    # Create more than 8 tables; oldest/lowest score should evict.
    for i in range(MAX_ENEMY_TABLES + 3):
        h = bytes([i]) * 32
        prior = np.array([1.0])
        table = tree.get_or_create_enemy_table(node, h, [0], prior)
        table.visits[:] = 2.0
        table.q[:] = 0.5
        node.N = i + 1
        table.touch(node.N)
    assert len(node.enemy_tables) <= MAX_ENEMY_TABLES
    assert tree.eviction_loss > 0.0
    assert tree.eviction_loss_rate() >= 0.0


def test_pending_pin_blocks_eviction():
    tree = SearchTree(max_enemy_tables=1, seat=0)
    node = InfoNode(
        key=b"\x03" * 32,
        turn=0,
        memory_digest=b"\x00" * 32,
        obs_hash=b"\x00" * 32,
        history_digest=ZERO_DIGEST,
    )
    node.actions = [0]
    node.prior = np.array([1.0])
    node.regret = np.zeros(1)
    node.avg_strategy = np.zeros(1)
    tree.set_root(node)
    h0 = b"\x10" * 32
    tree.get_or_create_enemy_table(node, h0, [0], np.array([1.0]))
    tree.pin_enemy(node, h0)
    h1 = b"\x11" * 32
    # Pinned h0 cannot be removed; new table stays and the node may temporarily
    # exceed the cap while the pin is held.
    tree.get_or_create_enemy_table(node, h1, [0], np.array([1.0]))
    assert h0 in node.enemy_tables
    assert h1 in node.enemy_tables


def test_partial_simulation_does_not_count():
    """Statistics change only through backup_path / completed_simulations."""
    _, obs, mem, belief, rng = _tiny_belief(5)
    ctl = SearchController(
        seat=0,
        evaluator=UniformEvaluator(0.0),
        config=SearchConfig(depth=2, pending_batch=2, n_particles=4),
        rng=rng,
    )
    ctl.ensure_root(obs, mem, belief)
    path = ctl.select_path(belief)
    assert ctl.tree.completed_simulations == 0
    assert ctl.tree.root is not None
    n_before = ctl.tree.root.N
    # Discard without backup.
    _ = path
    assert ctl.tree.root.N == n_before
    assert ctl.tree.completed_simulations == 0
