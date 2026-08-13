#!/usr/bin/env python3
"""Export a joe EMA checkpoint into ``bots/joe/artifact/``.

Phase 5 of docs/research/strategies/averagejoe-competition-plan.md: the
deployed bot plays the EMA weights (the deployment policy, plan section 5).
This script fetches one EMA checkpoint — by default the newest complete set
of the run in R2 (``training/joe/store.py``), or a local checkpoint dir with
a ``--local-dir`` — verifies it, sanity-loads it into the bot's own network
class, and writes:

    bots/joe/artifact/ema.eqx        the weights (gitignored, ~33 MB)
    bots/joe/artifact/manifest.json  architecture + checkpoint provenance

The weights file is inside the bot directory, so it is part of the bot's
content hash: every export forks the rating identity, and the manifest's
sha256 + R2 key make the exact weights recoverable for any rated version.

Usage:

    .venv/bin/python scripts/joe_export_bot.py --run-name joe-M-vast-20260813-0213
    .venv/bin/python scripts/joe_export_bot.py --run-name <run> --step 3000
    .venv/bin/python scripts/joe_export_bot.py --local-dir data/joe/<run>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "joe"))

BOT_ARTIFACT = REPO / "bots" / "joe" / "artifact"
WEIGHTS_NAME = "ema.eqx"

# Constructor fields the bot needs to rebuild the exact network pytree.
# history_size is not a Config field — training always used the network
# default (build_network forwards only _NET_CFG_FIELDS).
ARCH_FIELDS = ("depth", "embed_dim", "n_head", "ff_factor", "patch_size",
               "value_loss", "num_bins", "v_min", "v_max")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_from_r2(run_name: str, step: int | None, dest: Path) -> dict:
    """Download one EMA blob + the run config; return provenance fields."""
    import yaml

    from training.joe.store import R2Store

    store = R2Store.from_env()
    latest = store.resolve_latest(run_name)
    if latest is None:
        raise SystemExit(f"No complete checkpoint set in R2 for run '{run_name}'")
    state = latest["state"]

    ema_path = dest / WEIGHTS_NAME
    if step is None or step == state["global_step"]:
        state_used = state
        store.download_file(latest["objects"]["ema"]["key"], str(ema_path),
                            expected=latest["objects"]["ema"])
    else:
        # A step-named set older than latest: its state file names the blobs.
        # No stored checksum reference for non-latest sets; the manifest
        # records the hash of what was actually downloaded.
        state_used = store._get_json(
            store.key(run_name, "state", f"state-{step}.json"))
        if state_used is None:
            raise SystemExit(
                f"No state-{step}.json in R2 for run '{run_name}' "
                f"(latest is step {state['global_step']})")
        store.download_file(
            store.key(run_name, "checkpoints", state_used["files"]["ema"]),
            str(ema_path))

    cfg_path = dest / "config.download.yaml"
    store.download_run_file(run_name, "config.yaml", str(cfg_path))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg_path.unlink()

    return {
        "config": cfg,
        "checkpoint": {
            "source": "r2",
            "r2_key": store.key(run_name, "checkpoints",
                                state_used["files"]["ema"]),
            "run_name": state_used["run_name"],
            "global_step": state_used["global_step"],
            "curriculum_stage": state_used["curriculum_stage"],
            "last_eval_wr": state_used["last_eval_wr"],
            "engine_sha": state_used["engine_sha"],
        },
    }


def fetch_from_local(local_dir: Path, dest: Path) -> dict:
    """Copy the EMA named by the dir's state.json; return provenance."""
    import shutil

    import yaml

    from training.joe.state import read_state

    state = read_state(str(local_dir))
    if state is None:
        raise SystemExit(f"No v2 state.json in {local_dir}")
    src = local_dir / state["files"]["ema"]
    shutil.copyfile(src, dest / WEIGHTS_NAME)
    with open(local_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return {
        "config": cfg,
        "checkpoint": {
            "source": str(local_dir),
            "run_name": state["run_name"],
            "global_step": state["global_step"],
            "curriculum_stage": state["curriculum_stage"],
            "last_eval_wr": state["last_eval_wr"],
            "engine_sha": state["engine_sha"],
        },
    }


def sanity_load(arch: dict, weights_path: Path) -> int:
    """Deserialize into the bot's network class and run one forward pass."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from joe_net import HistoryTransformer

    pad_to = int(arch["pad_to"])
    template = HistoryTransformer(
        grid_size=pad_to, pad_to=pad_to,
        history_size=int(arch["history_size"]),
        patch_size=int(arch["patch_size"]), depth=int(arch["depth"]),
        embed_dim=int(arch["embed_dim"]), n_head=int(arch["n_head"]),
        ff_factor=int(arch["ff_factor"]), use_bf16=False,
        value_loss=arch["value_loss"], num_bins=int(arch["num_bins"]),
        v_min=float(arch["v_min"]), v_max=float(arch["v_max"]),
        key=jrandom.PRNGKey(0))
    net = eqx.tree_deserialise_leaves(str(weights_path), template)

    obs = jnp.zeros((25 + 2 * int(arch["history_size"]), pad_to, pad_to))
    move = jnp.zeros((pad_to, pad_to, 4), dtype=bool)
    build = jnp.zeros((pad_to, pad_to), dtype=bool)
    temporal = jnp.zeros((2, net.temporal_window))
    logits, value, _ = net._forward(obs, move, build, temporal)
    if not bool(jnp.isfinite(value)) or not bool(jnp.all(jnp.isfinite(
            jnp.where(logits < -1e8, 0.0, logits)))):
        raise SystemExit("Sanity forward pass produced non-finite outputs")
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_array)))
    print(f"Sanity load OK: {n_params:,} params, value(zeros)={float(value):+.3f}")
    return n_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-name", help="R2 run to export (default source)")
    parser.add_argument("--step", type=int, default=None,
                        help="specific global step (default: newest complete set)")
    parser.add_argument("--local-dir", type=Path, default=None,
                        help="local checkpoint dir with state.json instead of R2")
    parser.add_argument("--out", type=Path, default=BOT_ARTIFACT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.local_dir is not None:
        prov = fetch_from_local(args.local_dir, args.out)
    elif args.run_name:
        prov = fetch_from_r2(args.run_name, args.step, args.out)
    else:
        raise SystemExit("Pass --run-name (R2) or --local-dir")

    cfg = prov["config"]
    arch = {"network": cfg["network"], "pad_to": int(cfg["pad_to"]),
            "history_size": 7}
    arch.update({k: cfg[k] for k in ARCH_FIELDS})

    weights_path = args.out / WEIGHTS_NAME
    n_params = sanity_load(arch, weights_path)

    manifest = {
        "schema": 1,
        "weights": WEIGHTS_NAME,
        "weights_sha256": _sha256(weights_path),
        "weights_size": weights_path.stat().st_size,
        "n_params": n_params,
        "network": arch,
        "checkpoint": prov["checkpoint"],
        "exported_at": time.time(),
    }
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    ck = prov["checkpoint"]
    print(f"Exported {ck['run_name']} step {ck['global_step']} "
          f"(stage {ck['curriculum_stage']}, eval wr {ck['last_eval_wr']:.1%}) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
