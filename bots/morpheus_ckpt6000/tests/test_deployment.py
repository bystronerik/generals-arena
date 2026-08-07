"""Part 09 — deployment config loading and NetworkEvaluator contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from deployment import (  # noqa: E402
    DeploymentConfig,
    load_deployment,
    save_deployment,
)
from evaluator import NetworkEvaluator  # noqa: E402
from inference import load_default_session  # noqa: E402
from memory import empty_memory, update_memory  # noqa: E402
from observe import emit_observation  # noqa: E402
from runtime import COST_COMPONENTS as RUNTIME_COST_COMPONENTS  # noqa: E402
from state import create_initial_state  # noqa: E402


def _minimal_deployment(**overrides) -> DeploymentConfig:
    base = DeploymentConfig(
        offline_p99_ms={name: 1.0 for name in RUNTIME_COST_COMPONENTS},
        p99_window=32,
        admission_guard_ms=10.0,
        n_particles=32,
        target_simulations=8,
        max_proposal_batch=16,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_deployment_round_trip(tmp_path: Path):
    cfg = _minimal_deployment(qualification_host="test-host")
    path = tmp_path / "deployment.json"
    save_deployment(cfg, path)
    loaded = load_deployment(path)
    assert loaded.p99_estimator_type == "nearest_rank_empirical"
    assert loaded.p99_warmup_rule
    assert loaded.admission_guard_ms == 10.0
    assert set(loaded.offline_p99_ms) >= set(RUNTIME_COST_COMPONENTS)
    rt = loaded.to_runtime_config()
    assert rt.n_particles == 32
    assert rt.max_proposal_batch == 16
    assert rt.p99_window == 32


def test_deployment_rejects_missing_offline_component(tmp_path: Path):
    cfg = _minimal_deployment()
    data = cfg.to_dict()
    del data["offline_p99_ms"]["backup"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="offline_p99_ms"):
        load_deployment(path)


def test_network_evaluator_shapes():
    session = load_default_session()
    ev = NetworkEvaluator(session)
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[0, 0] = 1
    grid[9, 9] = 2
    state = create_initial_state(grid)
    obs = emit_observation(state, 0)
    mem = update_memory(empty_memory(10, 10), obs)
    from belief import BeliefConfig, initialize_belief

    belief = initialize_belief(
        obs,
        seat=0,
        rng=np.random.default_rng(0),
        config=BeliefConfig(n_particles=4, min_general_distance=5),
    )
    prior, value = ev.evaluate(obs, mem, belief, from_root=True)
    assert prior.shape == (3970,)
    assert abs(float(prior.sum()) - 1.0) < 1e-5
    assert isinstance(value, float)
    logits = ev.policy_logits(np.zeros((2, 49, 21, 21), dtype=np.float32))
    assert logits.shape == (2, 3970)
