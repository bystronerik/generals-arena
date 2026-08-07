"""Deployment manifest validation — pure JSON + dataclass, no model load.

Kept out of the ``morpheus`` marker (unlike ``test_deployment.py``, which loads a
real checkpoint) so the shipped-manifest regression runs in the default suite.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from deployment import (  # noqa: E402
    BOT_DIR,
    DEFAULT_DEPLOYMENT_PATH,
    DeploymentConfig,
    load_deployment,
)
from runtime import COST_COMPONENTS as RUNTIME_COST_COMPONENTS  # noqa: E402

REPO_ROOT = BOT_DIR.parent.parent
OPERATOR_CONFIG_PATH = REPO_ROOT / "scripts/configs/morpheus/online-runtime.json"


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


@pytest.mark.parametrize(
    "path",
    [DEFAULT_DEPLOYMENT_PATH, OPERATOR_CONFIG_PATH],
    ids=["bot_closure", "operator_copy"],
)
def test_shipped_manifests_have_finite_offline_p99(path: Path):
    """A non-finite seed locks its component out of admission for 64 turns."""
    cfg = load_deployment(path)
    for name in RUNTIME_COST_COMPONENTS:
        value = cfg.offline_p99_ms[name]
        assert math.isfinite(value), f"{path.name}: {name} offline p99 is not finite"
        assert value >= 0.0, f"{path.name}: {name} offline p99 is negative"


def test_deployment_rejects_nan_offline_component(tmp_path: Path):
    data = _minimal_deployment().to_dict()
    data["offline_p99_ms"]["enemy_prior_batch"] = float("nan")
    path = tmp_path / "nan.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="offline_p99_ms"):
        load_deployment(path)


def test_deployment_rejects_negative_offline_component(tmp_path: Path):
    data = _minimal_deployment().to_dict()
    data["offline_p99_ms"]["leaf_batch"] = -1.0
    path = tmp_path / "negative.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="offline_p99_ms"):
        load_deployment(path)
