#!/usr/bin/env python3
"""Grow a trained joe net and seed a step-0 continuation run in R2.

Two growth operations, selected by ``--op`` (the file name is historical:
depth growth came first):

- ``--op depth`` — splice identity blocks, depth 5 -> 7. Mechanics in
  docs/research/strategies/joe-depth7-growth-plan.md, sections 2-4.
  Forward parity is required to be bitwise.
- ``--op ff`` — widen every block's FFN, ff x3 -> ff x4. Mechanics in
  docs/research/strategies/joe-ff4-growth-plan.md, sections 2-3. The FFN
  GEMM shapes change, so parity is bitwise-first with a tolerance +
  argmax-equality fallback (``--parity-atol``).

The tool fetches the base run's latest checkpoint set from R2, grows both
lineages (train net from the checkpoint tuple, and the EMA net — the
deployment policy), verifies forward parity, and fabricates a complete v2
step-0 checkpoint set for the new run:

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
``joe_vast_train.py launch --tier <tier> --run-name <run>``: ``main.run``
refuses a state.json whose run_name differs from the config's. The tool
refuses a target run that already has a latest.json.

    python scripts/joe_grow_depth.py --base-run joe-M-vast-<stamp> \
        --run-name joe-M7-vast-<stamp>            # local surgery + parity
    python scripts/joe_grow_depth.py --base-run ... --run-name ... --upload
    python scripts/joe_grow_depth.py --op ff --base-run joe-M7-vast-<stamp> \
        --run-name joe-M7F4-vast-<stamp> --expect-params 13581658
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
from training.joe.grow import (  # noqa: E402
    assert_forward_parity, grow_depth, grow_ff)
from training.joe.launch import validate_run_name  # noqa: E402
from training.joe.main import detect_engine_sha, make_optimizer  # noqa: E402
from training.joe.networks import build_network  # noqa: E402
from training.joe.state import write_state  # noqa: E402
from training.joe.store import R2Store, load_dotenv  # noqa: E402

CONFIG_DIR = REPO / "training" / "joe" / "configs"
# One default target config per growth op — the tier each op was written for.
DEFAULT_CONFIG = {"depth": CONFIG_DIR / "M7.yaml",
                  "ff": CONFIG_DIR / "M7F4.yaml"}
GROW_FN = {"depth": grow_depth, "ff": grow_ff}
# The depth graft leaves every GEMM shape untouched, so bitwise equality is
# the claim. The FF graft changes the FFN GEMM shapes (ff4 plan section 2).
PARITY_ATOL = {"depth": None, "ff": 1e-5}


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
        description="Function-preserving growth surgery + step-0 R2 seed "
                    "(depth plan sections 2-4, ff4 plan sections 2-3)")
    parser.add_argument("--op", choices=sorted(GROW_FN), default="depth",
                        help="growth operation: depth splices identity "
                             "blocks, ff widens every block's FFN "
                             "(default depth)")
    parser.add_argument("--base-run", required=True,
                        help="R2 run name holding the trained base set")
    parser.add_argument("--run-name", required=True,
                        help="new run name; must equal the later "
                             "joe_vast_train.py launch --run-name")
    parser.add_argument("--config", default="",
                        help="target config (default: the op's tier — "
                             "configs/M7.yaml for depth, configs/M7F4.yaml "
                             "for ff)")
    parser.add_argument("--out-dir", default="",
                        help="local dir for the step-0 set "
                             "(default data/joe/<run-name>/seed)")
    parser.add_argument("--num-samples", type=int, default=64,
                        help="parity-check sample count (plan: >= 64)")
    parser.add_argument("--parity-atol", type=float, default=None,
                        help="override the op's parity tolerance; the ff "
                             "default is 1e-5 with argmax equality, depth "
                             "demands bitwise equality")
    parser.add_argument("--expect-params", type=int, default=0,
                        help="refuse unless the grown net has exactly this "
                             "parameter count (ff4 plan step 3: 13581658)")
    parser.add_argument("--upload", action="store_true",
                        help="upload the verified set to R2 and confirm "
                             "resolve_latest returns it")
    args = parser.parse_args(argv)

    config_path = args.config or str(DEFAULT_CONFIG[args.op])
    grow = GROW_FN[args.op]
    atol = PARITY_ATOL[args.op] if args.parity_atol is None else args.parity_atol

    run_name = validate_run_name(args.run_name)
    cfg = Config.from_yaml(config_path)
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
        grown_net = grow(base_net, cfg, grow_key)
        # Same key: the EMA lineage gets the same fresh-unit init.
        grown_ema = grow(base_ema, cfg, grow_key)

        for lineage, old_net, new_net in (("train", base_net, grown_net),
                                          ("EMA", base_ema, grown_ema)):
            print(f"parity check: {lineage} lineage "
                  f"({args.num_samples} samples)", flush=True)
            report = assert_forward_parity(
                old_net, new_net, args.num_samples, atol=atol)
            # Record these numbers in the plan's run log.
            diffs = ", ".join(f"{k} {v:.3g}"
                              for k, v in report["max_abs_diff"].items())
            print(f"  parity OK ({'bitwise' if report['bitwise'] else 'atol'})"
                  f": max abs diff {diffs}, argmax agreement "
                  f"{report['argmax_agreement']:.4f}", flush=True)
        grown_params = _n_params(grown_net)
        print(f"parameters: base {_n_params(base_net):,} -> grown "
              f"{grown_params:,} (depth {base_cfg.depth} -> {cfg.depth}, "
              f"ff_factor {base_cfg.ff_factor} -> {cfg.ff_factor})",
              flush=True)
        if args.expect_params and grown_params != args.expect_params:
            raise SystemExit(
                f"grown net has {grown_params:,} parameters, expected "
                f"{args.expect_params:,}; refusing to seed")

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
          f"--tier {Path(config_path).stem} --run-name {run_name}", flush=True)
    print("then confirm the boot log line 'Resuming from ...: global step 0' "
          "a few minutes in (plan step 4)", flush=True)


if __name__ == "__main__":
    main()
