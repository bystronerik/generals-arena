#!/usr/bin/env python3
"""Phase 3+ training entry for the joe pipeline (Modal, GPU).

Runs ``training/joe`` PPO training on a Modal GPU with checkpoints on the
``morpheus-training`` Volume under the fresh ``/vol/joe/`` prefix (plan
section 6). The engine SHA is read locally and pinned into the run manifest.

Smoke run (Phase 3 verification — ~30-40 min on one H100):

    modal run scripts/joe_modal_train.py --tier M --smoke > /tmp/joe_smoke.log 2>&1 &
    modal app list           # then check startup a few minutes in (AGENTS.md)
    modal app logs <app-id>

Full run (Phase 4): same entry without --smoke. Re-launching with the same
--run-name resumes from the run dir's state.json on the Volume (global step
and curriculum stage); --init-checkpoint/--iteration-offset stay as the
manual path when no state file exists.

Never pipe the output through tail/head — redirect to a file (AGENTS.md).
All repo imports live inside functions: Modal re-imports this file in the
container, and module scope must survive both sides.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
if modal.is_local():
    sys.path.insert(0, str(REPO))

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "jax[cuda12]==0.11.0",
        "equinox",
        "optax",
        "pyyaml",
        "boto3",
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

R2_KEYS = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
           "R2_BUCKET")


def _r2_secret():
    """R2 credentials from the local gitignored .env, as a Modal Secret.

    Values never appear in code or the image; they are injected into the
    container environment at call time. In the container this module is
    re-imported without the local .env, so the secret is empty there —
    only the locally built one is ever attached to a call.
    """
    if not modal.is_local():
        return modal.Secret.from_dict({})
    from training.joe.store import load_dotenv

    load_dotenv()
    return modal.Secret.from_dict(
        {k: os.environ[k] for k in R2_KEYS if os.environ.get(k)})


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
    stats = jax.local_devices()[0].memory_stats() or {}
    peak_gib = round(stats.get("peak_bytes_in_use", 0) / 2**30, 1)
    print(f"Peak device memory: {peak_gib} GiB", flush=True)
    return {"run_name": cfg.run_name, "ckpt_dir": ckpt_dir,
            "peak_device_gib": peak_gib}


@app.function(image=IMAGE, gpu="H100:2", timeout=23 * 3600,
              volumes={"/vol": VOLUME}, secrets=[_r2_secret()])
def train_r2_h100x2(run_name: str, cfg_overrides: str) -> dict:
    """Resume an R2-backed run (vast/GCP lineage) on 2x H100.

    Runs the same ``training.joe.vast_boot`` the other platforms use: R2
    lease, checkpoint restore, heartbeat, ``CheckpointUploader``. The JIT
    cache lives on the Volume; checkpoints stay container-local because
    R2 is the durable store. ``cfg_overrides`` is the per-platform
    ``JOE_CONFIG_OVERRIDES`` JSON — on 2 GPUs the per-device ``num_envs``
    and ``minibatch_size`` must be half the single-GPU values or the
    recipe silently doubles.
    """
    import time as _time

    os.environ["RUN_NAME"] = run_name
    os.environ["JOE_ROOT"] = "/vol/joe-r2"      # jax-cache persists here
    os.environ["CKPT_DIR"] = "/root/joe/ckpt"   # ephemeral; R2 is durable
    os.environ["CONTAINER_ID"] = (
        f"modal-{os.environ.get('MODAL_TASK_ID', int(_time.time()))}")
    if cfg_overrides:
        os.environ["JOE_CONFIG_OVERRIDES"] = cfg_overrides

    # On vast/GCP the onstart script tees stdout into JOE_TRAIN_LOG and
    # vast_boot mirrors that file to R2 every minute. There is no onstart
    # here, so recreate the tee in-process; without it the R2 prefix gets
    # no logs/train-*.log and the only log copy is Modal's own capture.
    boot_id = _time.strftime("%Y%m%d-%H%M%S")
    log_path = "/root/joe/logs/train.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.environ["JOE_BOOT_ID"] = boot_id
    os.environ["JOE_TRAIN_LOG"] = log_path
    log_file = open(log_path, "a", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for st in self._streams:
                st.write(data)

        def flush(self):
            for st in self._streams:
                st.flush()

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    from training.joe import vast_boot

    try:
        vast_boot.main([])
    finally:
        VOLUME.commit()
    return {"run_name": run_name, "mode": "r2-h100x2"}


def _local_engine_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO / "competition-module"), "rev-parse", "HEAD"],
        text=True).strip()


@app.local_entrypoint()
def main(tier: str = "M", smoke: bool = False, run_name: str = "",
         overrides: str = "", r2_run: str = "", r2_overrides: str = ""):
    """overrides: JSON object of Config field overrides, applied last.

    --r2-run <name>: instead of a Volume-backed run, resume the named
    R2-backed run on 2x H100 through vast_boot (use --detach for long
    runs). --r2-overrides is the JOE_CONFIG_OVERRIDES JSON; on 2 GPUs
    pass at least {"num_envs": <half>, "minibatch_size": <half>}.
    """
    if r2_run:
        print(f"Resuming R2 run {r2_run} on 2x H100 "
              f"(overrides: {r2_overrides or 'none'})", flush=True)
        call = train_r2_h100x2.spawn(r2_run, r2_overrides)
        print(f"SPAWNED: {call.object_id} — run with --detach so the "
              f"train call outlives this process; watch it with "
              f"`modal app logs` and the R2 heartbeat", flush=True)
        return

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
