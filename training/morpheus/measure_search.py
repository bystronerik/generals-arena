"""Search-resource measurements for Morpheus Part 06.

Runs the tactical suite matrices plus a small live search sample and reports
table hit rate, eviction loss, memory, and completed simulations.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_BOT = _REPO / "bots" / "morpheus"
for entry in (_REPO, _REPO / "bots", _BOT):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)

from belief import BeliefConfig, initialize_belief
from matrix import effective_q, matrix_utilities, mixed_strategy
from memory import empty_memory, update_memory
from observe import emit_observation
from search import SearchConfig, SearchController, UniformEvaluator
from state import create_initial_state
from tactics import load_tactical_suite, suite_path
from tree import MAX_ENEMY_TABLES, MAX_TREE_NODES


def _suite_case_ok(case: dict) -> bool:
    q = np.asarray(case["q"], dtype=np.float64)
    visits = np.asarray(case["visits"], dtype=np.float64)
    prior_a = np.asarray(case["prior_self"], dtype=np.float64)
    prior_b = np.asarray(case["prior_enemy"], dtype=np.float64)
    q_eff = effective_q(visits, q, float(case.get("first_play", 0.0)))
    sigma_b = mixed_strategy(np.zeros_like(prior_b), prior_b, n=0)
    sigma_a = mixed_strategy(np.zeros_like(prior_a), prior_a, n=0)
    u_self, _, _ = matrix_utilities(sigma_a, sigma_b, q_eff)
    best = int(np.argmax(u_self))
    expected = case["self_actions"].index(case["expected_best_self"])
    return best == expected


def measure_search_resources(
    suite: Path | None = None,
    *,
    live_sims: int = 16,
) -> dict[str, Any]:
    """Tactical suite correctness plus live search resource counters."""
    path = Path(suite) if suite is not None else suite_path()
    cases = load_tactical_suite(path)
    case_results = []
    for case in cases:
        ok = _suite_case_ok(case)
        case_results.append({"name": case["name"], "ok": ok})

    # Live search sample for hit rate / eviction / memory / completed sims.
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[0, 0] = 1
    grid[9, 9] = 2
    state = create_initial_state(grid)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(10, 10), obs)
    rng = np.random.default_rng(0)
    belief = initialize_belief(
        obs,
        seat=0,
        rng=rng,
        config=BeliefConfig(n_particles=8, min_general_distance=5),
    )
    ctl = SearchController(
        seat=0,
        evaluator=UniformEvaluator(0.0),
        config=SearchConfig(
            depth=4,
            pending_batch=4,
            n_particles=8,
            max_enemy_tables=MAX_ENEMY_TABLES,
            max_nodes=MAX_TREE_NODES,
        ),
        rng=rng,
    )
    ctl.ensure_root(obs, mem, belief)
    t0 = time.perf_counter()
    remaining = live_sims
    while remaining > 0:
        n = ctl.run_batch(belief, n_sims=min(4, remaining))
        remaining -= n
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Force eviction path for the measurement.
    node = ctl.tree.root
    assert node is not None
    for i in range(MAX_ENEMY_TABLES + 2):
        h = bytes([(i + 20) % 256]) * 32
        ctl.tree.get_or_create_enemy_table(node, h, [0], np.array([1.0]))

    import sys as _sys

    tree_bytes = (
        sum(map(len, ctl.tree.nodes)) * 32
        + len(ctl.tree.nodes) * 256  # rough overhead
    )
    # Prefer a real sizeof on the node dict when available.
    try:
        tree_bytes = int(_sys.getsizeof(ctl.tree.nodes))
        for n in ctl.tree.nodes.values():
            tree_bytes += int(_sys.getsizeof(n.enemy_tables))
            for table in n.enemy_tables.values():
                tree_bytes += int(table.visits.nbytes + table.q.nbytes)
    except Exception:
        pass

    return {
        "suite": str(path),
        "suite_cases": case_results,
        "suite_pass": all(c["ok"] for c in case_results),
        "table_hit_rate": ctl.tree.table_hit_rate(),
        "table_hits": ctl.tree.table_hits,
        "table_misses": ctl.tree.table_misses,
        "eviction_loss": ctl.tree.eviction_loss,
        "eviction_loss_rate": ctl.tree.eviction_loss_rate(),
        "completed_simulations": ctl.tree.completed_simulations,
        "tree_nodes": len(ctl.tree.nodes),
        "max_tree_nodes": MAX_TREE_NODES,
        "max_enemy_tables": MAX_ENEMY_TABLES,
        "approx_tree_bytes": tree_bytes,
        "live_search_ms": elapsed_ms,
        "defaults": {
            "self_widening_cap": 16,
            "enemy_widening_cap": 12,
            "depth": 16,
            "enemy_tables": 8,
            "tree_nodes": 4096,
            "exploration_floor": 0.05,
            "pending_leaf_batch": 4,
        },
    }
