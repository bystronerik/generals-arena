#!/usr/bin/env python3
"""Phase 3+ training entry for the joe pipeline (Modal, GPU).

Runs ``training/joe`` PPO training on a Modal GPU with checkpoints on the
``morpheus-training`` Volume under the fresh ``/vol/joe/`` prefix (plan
section 6). The engine SHA is read locally and pinned into the run manifest.

Smoke run (Phase 3 verification — ~30-40 min on one H100):

    modal run scripts/joe_modal_train.py --tier M --smoke > /tmp/joe_smoke.log 2>&1 &
    modal app list           # then check startup a few minutes in (AGENTS.md)
    modal app logs <app-id>

Full run (Phase 4): same entry without --smoke; resumable via
--init-checkpoint/--ema-checkpoint/--iteration-offset overrides.

Never pipe the output through tail/head — redirect to a file (AGENTS.md).
All repo imports live inside functions: Modal re-imports this file in the
container, and module scope must survive both sides.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "jax[cuda12]==0.11.0",
        "equinox",
        "optax",
        "pyyaml",
    )
    .add_local_dir(
        str(REPO / "competition-module"),
        remote_path="/root/competition-module",
        copy=True,
    )
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root"})
    .add_local_file(str(REPO / "training" / "__init__.py"),
                    remote_path="/root/training/__init__.py")
    .add_local_dir(str(REPO / "training" / "joe"),
                   remote_path="/root/training/joe",
                   ignore=["**/__pycache__", "tests/**"])
)

app = modal.App("joe-train")
VOLUME = modal.Volume.from_name("morpheus-training", create_if_missing=True)


@app.function(image=IMAGE, gpu="H100", timeout=23 * 3600,
              volumes={"/vol": VOLUME})
def train_remote(cfg_dict: dict, engine_sha: str) -> dict:
    """Runs in the container: build Config, train, checkpoint to /vol/joe/."""
    import os

    import jax

    from training.joe.config import Config
    from training.joe.main import run

    # Persistent JIT cache on the Volume: pool-generation kernels alone cost
    # ~40 s cold (Phase 1), and restarts of a resumable run should skip that.
    os.makedirs("/vol/joe/jax-cache", exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", "/vol/joe/jax-cache")

    cfg = Config.from_dict(cfg_dict, source="cfg_dict")
    ckpt_dir = f"/vol/joe/{cfg.run_name}"
    print(f"Checkpoints: {ckpt_dir}", flush=True)
    run(cfg, ckpt_dir, engine_sha=engine_sha, on_checkpoint=VOLUME.commit)
    VOLUME.commit()
    return {"run_name": cfg.run_name, "ckpt_dir": ckpt_dir}


def _local_engine_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO / "competition-module"), "rev-parse", "HEAD"],
        text=True).strip()


@app.local_entrypoint()
def main(tier: str = "M", smoke: bool = False, run_name: str = "",
         overrides: str = ""):
    """overrides: JSON object of Config field overrides, applied last."""
    import yaml

    with open(REPO / "training" / "joe" / "configs" / f"{tier}.yaml") as f:
        cfg_dict = yaml.safe_load(f)

    if smoke:
        # Phase 3 smoke: ~30-40 min at stage 0 (distance 2-6), frequent
        # evals to show win-rate vs random climbing, one checkpoint cycle.
        cfg_dict.update(
            num_iters=80,
            eval_every=5,
            eval_games=256,
            ckpt_every=40,
            save_every=40,
        )
        stamp = time.strftime("%Y%m%d-%H%M")
        cfg_dict["run_name"] = run_name or f"joe-{tier}-smoke-{stamp}"
    elif run_name:
        cfg_dict["run_name"] = run_name

    if overrides:
        cfg_dict.update(json.loads(overrides))

    engine_sha = _local_engine_sha()
    print(f"Launching {cfg_dict['run_name']} (tier {tier}, "
          f"engine {engine_sha[:12]})", flush=True)
    result = train_remote.remote(cfg_dict, engine_sha)
    print(f"DONE: {json.dumps(result)}", flush=True)
