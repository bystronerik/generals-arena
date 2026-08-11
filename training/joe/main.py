"""Entry point: load config, initialize everything, run training.

Ported from AverageJoe ``main.py``. The heavy lifting is in ``run()`` so the
Modal entry (``scripts/joe_modal_train.py``) can call it directly with a
checkpoint directory on the Volume and an ``on_checkpoint`` commit hook.
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import fields

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import yaml

from training.joe.config import Config
from training.joe.logger import Logger
from training.joe.networks import build_network, get_network_bundle
from training.joe.train.ppo import train


def detect_engine_sha() -> str:
    """The competition-module commit — pinned into every run manifest
    (the 2026-07-27 move-order tiebreak flip changes results)."""
    try:
        repo = os.path.join(os.path.dirname(__file__), "..", "..", "competition-module")
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_optimizer(cfg):
    """Grad clip + Adam with the configured LR schedule (per optimizer step)."""
    steps_per_iter = cfg.num_epochs * (
        2 * cfg.num_envs * cfg.num_steps // cfg.minibatch_size)

    if cfg.lr_schedule == "power_law":
        num = cfg.lr_power_law_numerator
        exp = cfg.lr_power_law_exponent
        lr_min = cfg.lr_power_law_min
        lr_max = cfg.lr_power_law_max
        offset = cfg.iteration_offset * steps_per_iter

        def lr(step):
            iteration = (step + offset) / steps_per_iter + 1.0
            raw = num / (iteration ** exp)
            return jnp.clip(raw, lr_min, lr_max)
    elif cfg.lr_decay_iters > 0:
        lr = optax.linear_schedule(
            init_value=cfg.lr, end_value=cfg.final_lr,
            transition_steps=cfg.lr_decay_iters * steps_per_iter)
    else:
        lr = cfg.lr

    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr),
    )


def run(cfg: Config, ckpt_dir: str, engine_sha: str | None = None,
        on_checkpoint=None):
    """Build the network and optimizer, write the run manifest, train."""
    # TF32 matmul: free speedup on Ampere+, no accuracy loss for training
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    engine_sha = engine_sha or detect_engine_sha()
    bundle = get_network_bundle(cfg.network)

    print(f"JAX PPO with {cfg.network} ({bundle['cls'].__name__})")
    print(f"Grid: {cfg.min_grid_size}-{cfg.max_grid_size} "
          f"(padded to {cfg.pad_to}), Envs: {cfg.num_envs}")
    print(f"Engine SHA: {engine_sha}")
    print(f"Devices ({jax.device_count()}): {jax.devices()}")
    print(flush=True)

    key = jrandom.PRNGKey(cfg.seed)
    key, net_key = jrandom.split(key)
    network = build_network(cfg, net_key)

    optimizer = make_optimizer(cfg)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    # Load checkpoint: (network, opt_state) tuple first, network-only fallback
    if cfg.init_checkpoint:
        try:
            network, opt_state = eqx.tree_deserialise_leaves(
                cfg.init_checkpoint, (network, opt_state))
            print(f"Loaded weights + optimizer state from {cfg.init_checkpoint}")
        except Exception:
            network = eqx.tree_deserialise_leaves(cfg.init_checkpoint, network)
            opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
            print(f"Loaded weights from {cfg.init_checkpoint} (fresh optimizer)")

    params = eqx.filter(network, eqx.is_array)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Parameters: {n_params:,}", flush=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    manifest = {
        "run_name": cfg.run_name,
        "engine_sha": engine_sha,
        "n_params": n_params,
        "started": time.time(),
        "config": cfg.to_dict(),
    }
    with open(os.path.join(ckpt_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, default_flow_style=False)

    logger = Logger(ckpt_dir, hparams=cfg.to_dict())
    if on_checkpoint is not None:
        on_checkpoint()  # persist the manifest before the long haul

    network, opt_state, ema_network = train(
        cfg, network, optimizer, opt_state, logger, key, bundle,
        ckpt_dir, engine_sha, on_checkpoint=on_checkpoint)

    logger.finish()
    final_path = os.path.join(ckpt_dir, f"{cfg.run_name}_final.eqx")
    eqx.tree_serialise_leaves(final_path, (network, opt_state))
    final_ema = os.path.join(ckpt_dir, f"{cfg.run_name}_ema_final.eqx")
    eqx.tree_serialise_leaves(final_ema, ema_network)
    if on_checkpoint is not None:
        on_checkpoint()
    print(f"\nDone! Model saved to {final_path} (+ EMA {final_ema})")
    return network, ema_network


def parse_args():
    parser = argparse.ArgumentParser(description="Joe PPO training (competition rules)")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "configs", "M.yaml"))
    parser.add_argument("--ckpt-dir", type=str, default="")
    parser.add_argument("--engine-sha", type=str, default="")
    for f in fields(Config):
        flag = f"--{f.name}"
        if f.type is int or f.type == "int":
            parser.add_argument(flag, type=int, default=None)
        elif f.type is float or f.type == "float":
            parser.add_argument(flag, type=float, default=None)
        elif f.type is bool or f.type == "bool":
            parser.add_argument(flag, action="store_true", default=None)
        else:
            parser.add_argument(flag, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    for f in fields(Config):
        cli_val = getattr(args, f.name, None)
        if cli_val is not None:
            object.__setattr__(cfg, f.name, cli_val)
    ckpt_dir = args.ckpt_dir or os.path.join("data", "joe", cfg.run_name)
    run(cfg, ckpt_dir, engine_sha=args.engine_sha or None)


if __name__ == "__main__":
    main()
