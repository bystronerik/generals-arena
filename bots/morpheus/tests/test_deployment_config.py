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


@pytest.mark.parametrize(
    "path",
    [DEFAULT_DEPLOYMENT_PATH, OPERATOR_CONFIG_PATH],
    ids=["bot_closure", "operator_copy"],
)
def test_shipped_manifests_carry_bounded_shaping(path: Path):
    """Part 17: the shipped blend must be bounded, and the two copies agree."""
    cfg = load_deployment(path)
    assert cfg.evaluator == "network"
    assert math.isfinite(cfg.shaping_log_clip) and cfg.shaping_log_clip > 0.0
    assert 0.0 <= cfg.shaping_floor_frac < 1.0
    for lam in (cfg.shaping_lambda_pre_contact, cfg.shaping_lambda_post_contact):
        assert 0.0 <= lam <= 1.0


def test_bot_and_operator_shaping_match():
    bot = load_deployment(DEFAULT_DEPLOYMENT_PATH)
    operator = load_deployment(OPERATOR_CONFIG_PATH)
    for name in (
        "shaping_lambda_pre_contact",
        "shaping_lambda_post_contact",
        "shaping_log_clip",
        "shaping_floor_frac",
        "evaluator",
    ):
        assert getattr(bot, name) == getattr(operator, name), name


def test_runtime_config_carries_shaping_knobs():
    cfg = _minimal_deployment(
        shaping_lambda_pre_contact=0.25,
        shaping_lambda_post_contact=0.5,
        shaping_log_clip=1.5,
        shaping_floor_frac=0.01,
    )
    runtime = cfg.to_runtime_config()
    assert runtime.shaping_lambda_pre_contact == pytest.approx(0.25)
    assert runtime.shaping_lambda_post_contact == pytest.approx(0.5)
    assert runtime.shaping_log_clip == pytest.approx(1.5)
    assert runtime.shaping_floor_frac == pytest.approx(0.01)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shaping_lambda_pre_contact", 1.5),
        ("shaping_lambda_post_contact", -0.1),
        ("shaping_log_clip", 0.0),
        ("shaping_log_clip", float("inf")),
        ("shaping_floor_frac", 1.0),
    ],
)
def test_deployment_rejects_out_of_range_shaping(tmp_path: Path, field, value):
    data = _minimal_deployment().to_dict()
    data[field] = value
    path = tmp_path / "shaping.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=field):
        load_deployment(path)


def test_deployment_rejects_unknown_evaluator(tmp_path: Path):
    data = _minimal_deployment().to_dict()
    data["evaluator"] = "oracle"
    path = tmp_path / "evaluator.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="evaluator"):
        load_deployment(path)


def test_shaping_variant_is_a_distinct_closure(tmp_path: Path):
    """Part 17 C1/C2 arms must fork the content hash, not read an env var."""
    import shutil
    import subprocess
    import sys

    from arena.records.fingerprint import content_hash_for_dir

    script = REPO_ROOT / "scripts" / "morpheus_shaping_variant.py"
    name = "pytestarm"
    dest = REPO_ROOT / "bots" / f"morpheus-{name}"
    if dest.exists():
        shutil.rmtree(dest)
    try:
        subprocess.run(
            [sys.executable, str(script), name, "--evaluator", "uniform",
             "--lambda", "0.25"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        cfg = load_deployment(dest / "deployment.json")
        assert cfg.evaluator == "uniform"
        assert cfg.shaping_lambda_pre_contact == pytest.approx(0.25)
        assert cfg.shaping_lambda_post_contact == pytest.approx(0.25)
        # Everything not overridden must match the source closure exactly.
        base = load_deployment(DEFAULT_DEPLOYMENT_PATH)
        assert cfg.offline_p99_ms == base.offline_p99_ms
        assert cfg.target_simulations == base.target_simulations
        assert cfg.shaping_log_clip == base.shaping_log_clip
        # Same model bytes (symlinked artifact), different rated identity.
        assert (dest / "artifact").is_symlink()
        assert (dest / "artifact" / "manifest.json").is_file()
        assert content_hash_for_dir(dest) != content_hash_for_dir(BOT_DIR)
        # Bot-local tests must not ride along into a measurement arm.
        assert not (dest / "tests").exists()
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def test_run_sh_pins_every_math_backend_to_one_thread():
    """OpenMP/BLAS size their pools at library init — Python is too late."""
    run_sh = (BOT_DIR / "run.sh").read_text(encoding="utf-8")
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert f"export {var}=1" in run_sh, f"run.sh must pin {var}"
    # The exports must precede exec, or the interpreter never sees them.
    assert run_sh.index("export OMP_NUM_THREADS=1") < run_sh.index("exec ")


def test_play_and_calibration_pin_the_same_thread_count():
    """A bot calibrated single-threaded must not play multi-threaded.

    This mismatch made the whole offline_p99_ms block describe a machine the bot
    never ran on, and it locked search out entirely. Pin the invariant, not just
    the value.
    """
    import re

    agent_src = (BOT_DIR / "agent.py").read_text(encoding="utf-8")
    m = re.search(r"^TORCH_NUM_THREADS\s*=\s*(\d+)", agent_src, re.M)
    assert m, "agent.py must declare TORCH_NUM_THREADS"
    play_threads = int(m.group(1))
    assert play_threads == 1

    measure_src = (
        REPO_ROOT / "training/morpheus/measure_online.py"
    ).read_text(encoding="utf-8")
    assert "torch.set_num_threads(1)" in measure_src, (
        "calibration no longer pins one thread; play and measurement have "
        "diverged (see docs/bots/morpheus/thread-pinning.md)"
    )
    # And the bot must actually apply it before loading the session.
    assert "_pin_torch_threads()" in agent_src
    assert agent_src.index("_pin_torch_threads()", agent_src.index("def __init__")) < (
        agent_src.index("load_default_session()", agent_src.index("def __init__"))
    )
