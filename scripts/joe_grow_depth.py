#!/usr/bin/env python3
"""Grow the trained depth-5 M net to depth 7 and seed a step-0 run in R2.

Mechanics: docs/research/strategies/joe-depth7-growth-plan.md, sections
2-4. The tool fetches the base run's latest checkpoint set from R2, grows
both lineages (train net from the checkpoint tuple, and the EMA net — the
deployment policy), verifies exact forward parity, and fabricates a
complete v2 step-0 checkpoint set for the new run:

  <run>_0.eqx       (grown_network, fresh make_optimizer(cfg) state)
  <run>_ema_0.eqx   grown EMA net
  state.json        schema 2, global_step 0, curriculum_stage 0

Without --upload the set stays local (procedure step 3: surgery + local
verification). With --upload it goes to R2 via the ordered
``upload_checkpoint`` protocol and the tool verifies ``resolve_latest``
now returns it (procedure step 4). Ordering is the safety-critical part:
the seed must be in R2 before the instance boots, or vast_boot silently
trains a random-init depth-7 net from scratch.

The --run-name here must be the exact name later passed to
``joe_vast_train.py launch --tier M7 --run-name <run>``: ``main.run``
refuses a state.json whose run_name differs from the config's. The tool
refuses a target run that already has a latest.json.

    python scripts/joe_grow_depth.py --base-run joe-M-vast-<stamp> \
        --run-name joe-M7-vast-<stamp>            # local surgery + parity
    python scripts/joe_grow_depth.py --base-run ... --run-name ... --upload
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.random as jrandom  # noqa: E402

from training.joe.config import Config  # noqa: E402
from training.joe.grow import assert_forward_parity, grow_depth  # noqa: E402
from training.joe.launch import validate_run_name  # noqa: E402
from training.joe.main import detect_engine_sha, make_optimizer  # noqa: E402
from training.joe.networks import build_network  # noqa: E402
from training.joe.state import write_state  # noqa: E402
from training.joe.store import R2Store, load_dotenv  # noqa: E402

DEFAULT_CONFIG = REPO / "training" / "joe" / "configs" / "M7.yaml"


def _n_params(tree):
    return sum(x.size for x in jax.tree.leaves(eqx.filter(tree, eqx.is_array)))


def _load_base(store, base_run, latest, dest_dir):
    """Download the base set + run config; deserialize both lineages."""
    store.download_checkpoint(latest, dest_dir)
    base_cfg_path = os.path.join(dest_dir, "base_config.yaml")
    store.download_run_file(base_run, "config.yaml", base_cfg_path)
    base_cfg = Config.from_yaml(base_cfg_path)

    # Same template path main.run uses, so deserialization matches
    # leaf-for-leaf: build_network + make_optimizer(base_cfg).init.
    net_template = build_network(base_cfg, jrandom.PRNGKey(base_cfg.seed))
    opt_template = make_optimizer(base_cfg).init(
        eqx.filter(net_template, eqx.is_array))

    files = latest["state"]["files"]
    base_net, _ = eqx.tree_deserialise_leaves(
        os.path.join(dest_dir, files["full"]), (net_template, opt_template))
    base_ema = eqx.tree_deserialise_leaves(
        os.path.join(dest_dir, files["ema"]), net_template)
    return base_cfg, base_net, base_ema


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Function-preserving depth growth + step-0 R2 seed "
                    "(growth plan sections 2-4)")
    parser.add_argument("--base-run", required=True,
                        help="R2 run name holding the trained base set")
    parser.add_argument("--run-name", required=True,
                        help="new run name; must equal the later "
                             "joe_vast_train.py launch --run-name")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="target config (default configs/M7.yaml)")
    parser.add_argument("--out-dir", default="",
                        help="local dir for the step-0 set "
                             "(default data/joe/<run-name>/seed)")
    parser.add_argument("--num-samples", type=int, default=64,
                        help="parity-check sample count (plan: >= 64)")
    parser.add_argument("--upload", action="store_true",
                        help="upload the verified set to R2 and confirm "
                             "resolve_latest returns it")
    args = parser.parse_args(argv)

    run_name = validate_run_name(args.run_name)
    cfg = Config.from_yaml(args.config)
    object.__setattr__(cfg, "run_name", run_name)

    load_dotenv()
    store = R2Store.from_env()
    if store.resolve_latest(run_name) is not None:
        raise SystemExit(
            f"{run_name} already has a latest.json in R2; seeding must "
            "happen before the run's first checkpoint. Pick a fresh run "
            "name (or destroy --purge-r2 the old run).")
    base_latest = store.resolve_latest(args.base_run)
    if base_latest is None:
        raise SystemExit(
            f"no latest.json for base run {args.base_run}; nothing to grow")

    base_state = base_latest["state"]
    print(f"base {args.base_run}: global step {base_state['global_step']}, "
          f"engine {base_state['engine_sha'][:12]}", flush=True)
    for role, ref in sorted(base_latest["objects"].items()):
        # Record these SHAs in the plan's run log (section 10).
        print(f"  {role}: {ref['key']} sha256={ref['sha256']}", flush=True)

    with tempfile.TemporaryDirectory(prefix="joe-grow-") as td:
        base_cfg, base_net, base_ema = _load_base(
            store, args.base_run, base_latest, td)

        grow_key = jrandom.PRNGKey(cfg.seed)
        grown_net = grow_depth(base_net, cfg, grow_key)
        # Same key: the EMA lineage gets the same fresh-block init.
        grown_ema = grow_depth(base_ema, cfg, grow_key)

        print(f"parity check: train lineage ({args.num_samples} samples)",
              flush=True)
        assert_forward_parity(base_net, grown_net, args.num_samples)
        print(f"parity check: EMA lineage ({args.num_samples} samples)",
              flush=True)
        assert_forward_parity(base_ema, grown_ema, args.num_samples)
        print(f"parity OK: exact logit/value equality on "
              f"{args.num_samples} masked inputs per lineage", flush=True)
        print(f"parameters: base {_n_params(base_net):,} -> grown "
              f"{_n_params(grown_net):,} "
              f"(depth {base_cfg.depth} -> {cfg.depth})", flush=True)

        opt_state = make_optimizer(cfg).init(
            eqx.filter(grown_net, eqx.is_array))

        out_dir = args.out_dir or os.path.join(
            "data", "joe", run_name, "seed")
        os.makedirs(out_dir, exist_ok=True)
        full_name = f"{run_name}_0.eqx"
        ema_name = f"{run_name}_ema_0.eqx"
        eqx.tree_serialise_leaves(
            os.path.join(out_dir, full_name), (grown_net, opt_state))
        eqx.tree_serialise_leaves(
            os.path.join(out_dir, ema_name), grown_ema)
        state = write_state(out_dir, run_name, 0, 0, 0.0,
                            detect_engine_sha(),
                            files={"full": full_name, "ema": ema_name})
        print(f"wrote step-0 set to {out_dir}: {full_name}, {ema_name}, "
              f"state.json", flush=True)

    if not args.upload:
        print("local only (no --upload). Next: the local seeded smoke "
              "(plan step 3), then re-run with --upload before launch.",
              flush=True)
        return

    if store.resolve_latest(run_name) is not None:
        raise SystemExit(
            f"{run_name} gained a latest.json while the surgery ran; "
            "refusing to overwrite it")
    uploaded = store.upload_checkpoint(out_dir, state)
    check = store.resolve_latest(run_name)
    if check is None or check["state"]["global_step"] != 0 or \
            check["objects"] != uploaded["objects"]:
        raise SystemExit(
            f"upload verification failed: resolve_latest({run_name}) does "
            f"not return the seeded step-0 set — do NOT launch")
    print(f"R2 seeded and verified: resolve_latest({run_name}) returns the "
          f"step-0 set", flush=True)
    print(f"launch with: python scripts/joe_vast_train.py launch "
          f"--tier M7 --run-name {run_name}", flush=True)
    print("then confirm the boot log line 'Resuming from ...: global step 0' "
          "a few minutes in (plan step 4)", flush=True)


if __name__ == "__main__":
    main()
