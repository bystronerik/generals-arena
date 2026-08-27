#!/usr/bin/env python3
"""Kernel-level probe of the joe rollout and PPO update on one CUDA GPU.

Answers the three questions the phase profile could not
(docs/research/measurements/joe-train-phase-profile.md leaves the
transformer as one opaque number):

1. **Are the per-Linear output transposes materialized?** Every
   ``eqx.nn.Linear`` under ``vmap`` computes ``W @ x``, so the jaxpr emits
   ``dot_general`` with the output as ``(features, batch, tokens)`` and a
   transpose back. If XLA folds that into the dot's output layout the
   transposes are free; if it emits real copy/transpose fusions they cost a
   full read+write of the largest tensors in the network.

2. **Is ``Precision.HIGH`` selecting a multi-pass bf16 GEMM algorithm?**
   ``main.run`` sets ``jax_default_matmul_precision("tensorfloat32")``,
   which stamps HIGH on every dot. On bf16 operands that should be inert;
   if it maps to ``dot_bf16_bf16_f32_x3`` the loop pays 3x on every GEMM.

3. **How much of the wall clock is actually GEMM?** Aggregates the device
   kernels from a ``jax.profiler`` trace. This needs no peak-FLOP figure
   for the card, which is the point: the achieved-vs-peak argument cannot
   be made for silicon whose bf16 dense peak we do not have a trustworthy
   number for.

Runs anywhere with a CUDA device. Nothing here is vast- or Modal-specific.

    python scripts/joe_gpu_kernel_probe.py --config training/joe/configs/X16.yaml

The HLO dump directory must be set before jax is imported, so the XLA_FLAGS
assignment below sits above every jax import on purpose.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.path.join(
        REPO, "training", "joe", "configs", "X16.yaml"))
    p.add_argument("--pool-size", type=int, default=8192,
                   help="Map pool is irrelevant to transformer kernels; a "
                        "small one saves minutes of paid GPU time.")
    p.add_argument("--num-envs", type=int, default=0, help="0 = config value")
    p.add_argument("--num-steps", type=int, default=0, help="0 = config value")
    p.add_argument("--minibatch-size", type=int, default=0,
                   help="0 = config value; must divide the kept-sample count")
    p.add_argument("--hlo-dir", default="/tmp/joe-hlo")
    p.add_argument("--trace-dir", default="/tmp/joe-trace")
    p.add_argument("--out", default="/tmp/joe-kernel-probe.json")
    p.add_argument("--reps", type=int, default=2,
                   help="Timed rollout+PPO pairs inside the trace.")
    return p.parse_args()


ARGS = parse_args()

# Must precede any jax import.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + f" --xla_dump_to={ARGS.hlo_dir}"
).strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import equinox as eqx          # noqa: E402
import jax                     # noqa: E402
import jax.numpy as jnp        # noqa: E402
import jax.random as jrandom   # noqa: E402
import optax                   # noqa: E402

from training.joe.config import Config                                # noqa: E402
from training.joe.env import (make_competition_env,                   # noqa: E402
                              preset_min_generals_distance)
from training.joe.networks import build_network, get_network_bundle   # noqa: E402
from training.joe.train.ppo import (_replicate, compute_gae,          # noqa: E402
                                    make_value_loss_fn, ppo_update)
from training.joe.train.rollout_selfplay import collect_rollout       # noqa: E402


def log(msg):
    print(f"[probe] {msg}", flush=True)


# ---------------------------------------------------------------------------
# HLO analysis
# ---------------------------------------------------------------------------

# A materialized transpose/copy on a Linear output looks like a fusion whose
# root is transpose/copy over a rank-3 tensor with the feature dim leading.
_TRANSPOSE_OP = re.compile(r"^\s*%?\S+\s*=\s*(\S+)\s+(transpose|copy|bitcast)\(",
                           re.MULTILINE)
_DOT_ALGO = re.compile(r"algorithm=(\S+?)[,\s\)]")
_FUSION_KIND = re.compile(r"kind=(\w+)")


def _analyze_one(text):
    """The two structural facts for one HLO module."""
    algos = collections.Counter(_DOT_ALGO.findall(text))
    ops = collections.Counter()
    shapes = collections.Counter()
    for m in re.finditer(
            r"=\s*(\w+\[[0-9,]*\])\s*(transpose|copy|bitcast)\(", text):
        shape, op = m.group(1), m.group(2)
        ops[op] += 1
        shapes[f"{op} {shape}"] += 1
    return {
        "dot_algorithms": dict(algos),
        "multi_pass_bf16": sum(n for a, n in algos.items()
                               if "_x3" in a or "_x6" in a),
        "transpose_like_ops": dict(ops),
        "transpose_like_top": shapes.most_common(15),
        "fusion_kinds": dict(collections.Counter(_FUSION_KIND.findall(text))),
    }


def analyze_hlo(hlo_dir):
    """Read the optimized HLO dumps and report the two structural facts."""
    files = sorted(glob.glob(os.path.join(hlo_dir, "*after_optimizations.txt")))
    if not files:
        files = sorted(glob.glob(os.path.join(hlo_dir, "*optimizations*.txt")))
    out = {"files": [os.path.basename(f) for f in files]}
    if not files:
        out["error"] = f"no optimized HLO under {hlo_dir}"
        return out

    # The rollout and the PPO update are separate modules; both matter, so
    # report the largest few rather than folding them together. The map-pool
    # generator compiles 16 large modules of its own and would otherwise win
    # on size and be analyzed instead of the network -- it has no bearing on
    # transformer kernels, so drop it along with the trivial helpers.
    skip = ("make_pool_batch", "threefry", "convert_element_type",
            "_linspace", "jit_unstack", "dynamic_slice", "jit_stage",
            "broadcast_in_dim", "jit_iota", "generate_grid",
            "jit_init_state", "jit_reset")
    named = [f for f in files
             if not any(k in os.path.basename(f) for k in skip)]
    out["excluded_modules"] = len(files) - len(named)
    top = sorted(named or files, key=os.path.getsize, reverse=True)[:3]
    out["modules"] = [{"name": os.path.basename(f),
                       "bytes": os.path.getsize(f)} for f in top]
    per_module = {}
    for f in top:
        per_module[os.path.basename(f)] = _analyze_one(
            open(f, errors="replace").read())
    out["per_module"] = per_module

    biggest = top[0]
    out["analyzed"] = os.path.basename(biggest)
    out["analyzed_bytes"] = os.path.getsize(biggest)
    text = open(biggest, errors="replace").read()

    # 2. dot algorithms actually selected
    algos = collections.Counter(_DOT_ALGO.findall(text))
    out["dot_algorithms"] = dict(algos)
    out["multi_pass_bf16"] = sum(
        n for a, n in algos.items() if "_x3" in a or "_x6" in a)

    # 1. transpose / copy operations, bucketed by result dtype+shape
    ops = collections.Counter()
    shapes = collections.Counter()
    for m in re.finditer(
            r"=\s*(\w+\[[0-9,]*\])\s*(transpose|copy|bitcast)\(", text):
        shape, op = m.group(1), m.group(2)
        ops[op] += 1
        shapes[f"{op} {shape}"] += 1
    out["transpose_like_ops"] = dict(ops)
    out["transpose_like_top"] = shapes.most_common(20)

    # Fusions whose root is a transpose/copy are the expensive case; a
    # bitcast is free (a view), which is exactly the distinction that
    # decides whether proposal #2 is worth anything.
    out["fusion_kinds"] = dict(collections.Counter(_FUSION_KIND.findall(text)))
    out["hlo_line_count"] = text.count("\n")
    return out


# ---------------------------------------------------------------------------
# Trace analysis
# ---------------------------------------------------------------------------

_GEMM_HINTS = ("gemm", "cutlass", "cublas", "sgemm", "hgemm", "dot", "matmul",
               "ampere", "sm80", "sm90", "turing", "wgrad", "implicit")
_ATTN_HINTS = ("softmax", "attention", "fmha", "flash")
_ELEM_HINTS = ("fusion", "elementwise", "loop", "copy", "transpose", "reduce",
               "broadcast", "convert", "select", "add", "mul", "layer_norm")


def classify(name):
    n = name.lower()
    if any(h in n for h in _GEMM_HINTS):
        return "gemm"
    if any(h in n for h in _ATTN_HINTS):
        return "attention/softmax"
    if any(h in n for h in _ELEM_HINTS):
        return "elementwise/fusion"
    return "other"


def analyze_trace(trace_dir):
    """Aggregate device kernel time from the profiler's Chrome trace."""
    pats = [os.path.join(trace_dir, "**", "*.trace.json.gz"),
            os.path.join(trace_dir, "**", "*.trace.json")]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    out = {"trace_files": [os.path.basename(f) for f in files]}
    if not files:
        out["error"] = f"no trace under {trace_dir}"
        return out

    path = max(files, key=os.path.getsize)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    # Identify GPU device pids by their process_name metadata. XLA names
    # them "/device:GPU:0 (pid N)" or similar; anything with "GPU" and not
    # "host" is a device stream.
    pid_name = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid_name[e.get("pid")] = e.get("args", {}).get("name", "")
    gpu_pids = {p for p, n in pid_name.items()
                if "gpu" in n.lower() and "host" not in n.lower()}
    out["process_names"] = sorted(set(pid_name.values()))[:20]

    by_name = collections.Counter()
    by_class = collections.Counter()
    total = 0.0
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu_pids:
            continue
        dur = float(e.get("dur", 0.0))
        name = e.get("name", "?")
        by_name[name] += dur
        by_class[classify(name)] += dur
        total += dur

    out["device_us_total"] = round(total, 1)
    out["by_class_us"] = {k: round(v, 1) for k, v in by_class.most_common()}
    out["by_class_pct"] = {k: round(100 * v / total, 1)
                           for k, v in by_class.most_common()} if total else {}
    out["top_kernels"] = [
        {"name": n[:110], "us": round(v, 1),
         "pct": round(100 * v / total, 2) if total else 0.0,
         "class": classify(n)}
        for n, v in by_name.most_common(35)]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dev = jax.devices()[0]
    cc = getattr(dev, "compute_capability", "?")
    ND = jax.device_count()
    log(f"device={dev} compute_capability={cc} jax={jax.__version__} "
        f"devices={ND}")

    cfg = Config.from_yaml(ARGS.config)
    object.__setattr__(cfg, "pool_size", ARGS.pool_size)
    if ARGS.num_envs:
        object.__setattr__(cfg, "num_envs", ARGS.num_envs)
    if ARGS.num_steps:
        object.__setattr__(cfg, "num_steps", ARGS.num_steps)
    if ARGS.minibatch_size:
        object.__setattr__(cfg, "minibatch_size", ARGS.minibatch_size)

    # The training entry sets this; it is one of the things under test.
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")

    N, T = cfg.num_envs, cfg.num_steps
    log(f"config={os.path.basename(ARGS.config)} depth={cfg.depth} "
        f"embed={cfg.embed_dim} ff={cfg.ff_factor} heads={cfg.n_head} "
        f"head_dim={cfg.embed_dim // cfg.n_head} bf16={cfg.use_bf16}")
    log(f"shape: {N} envs x {T} steps x 2 seats = {2 * N * T:,} samples/iter, "
        f"minibatch {cfg.minibatch_size}")

    bundle = get_network_bundle(cfg.network)
    augment_fn = bundle["augment_obs"]
    init_obs_state_fn = bundle["init_obs_state"]
    value_loss_fn = make_value_loss_fn(cfg)

    t0 = time.perf_counter()
    env = make_competition_env(preset_min_generals_distance(), None,
                               cfg.pool_size)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    jax.block_until_ready(pool.armies)
    log(f"pool {cfg.pool_size} in {time.perf_counter() - t0:.1f}s")

    network = build_network(cfg, jrandom.PRNGKey(1))
    params, static = eqx.partition(network, eqx.is_array)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    log(f"parameters: {n_params:,}")

    optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                            optax.adam(cfg.lr))
    opt_state = optimizer.init(params)

    # Mirror train(): everything pmapped over the device axis, so the
    # rollout output already carries it and feeds the PPO step with no
    # extra copy. Replicating an 18 GB rollout tensor afterwards would
    # both double the memory and stop matching the production kernels.
    p_params = _replicate(params, ND)
    p_opt_state = _replicate(opt_state, ND)
    pool_rep = _replicate(pool, ND)

    p_init_envs = jax.pmap(
        lambda k: jax.vmap(env.init_state)(jrandom.split(k, N)))
    states = p_init_envs(jrandom.split(jrandom.PRNGKey(2), ND))

    single = init_obs_state_fn(cfg.pad_to, cfg.pad_to)
    batched = jax.tree.map(lambda x: jnp.tile(x, (N, *([1] * x.ndim))), single)
    osp0 = _replicate(batched, ND)
    osp1 = _replicate(batched, ND)
    keys = jrandom.split(jrandom.PRNGKey(3), ND)

    def _rollout(prm, st, key, a, b, pl):
        net = eqx.combine(prm, static)
        return collect_rollout(st, env, net, key, T, a, b, augment_fn, pl)

    p_rollout = jax.pmap(_rollout)

    log("compiling rollout (HLO dumps on this call) ...")
    t0 = time.perf_counter()
    states, rollout_data, keys, osp0, osp1 = p_rollout(
        p_params, states, keys, osp0, osp1, pool_rep)
    jax.block_until_ready(states)
    log(f"rollout compile+run {time.perf_counter() - t0:.1f}s")

    # ---- answer the structural questions immediately, before anything
    # else can fail or the instance can be preempted -------------------
    hlo = analyze_hlo(ARGS.hlo_dir)
    log("=" * 60)
    log("HLO FACTS")
    log(f"  dot algorithms: {hlo.get('dot_algorithms')}")
    log(f"  multi-pass bf16 dots (x3/x6): {hlo.get('multi_pass_bf16')}")
    log(f"  transpose-like op counts: {hlo.get('transpose_like_ops')}")
    for entry, n in (hlo.get("transpose_like_top") or [])[:12]:
        log(f"    {n:>5}  {entry}")
    log(f"  fusion kinds: {hlo.get('fusion_kinds')}")
    for mod, facts in (hlo.get("per_module") or {}).items():
        log(f"  -- {mod}")
        log(f"       dots: {facts['dot_algorithms']} "
            f"multi_pass={facts['multi_pass_bf16']}")
        log(f"       transpose-like: {facts['transpose_like_ops']}")
        for entry, n in facts["transpose_like_top"][:6]:
            log(f"         {n:>5}  {entry}")
    log("=" * 60)
    partial = {"hlo": hlo, "device": str(dev), "compute_capability": str(cc)}
    with open(ARGS.out, "w") as f:
        json.dump(partial, f, indent=2)
    log(f"partial results -> {ARGS.out}")

    # ---- PPO step, pmapped exactly as train() does --------------------
    (obs, move_masks, build_masks, temporal, actions, lps, vals,
     next_vals, rews, terminated, truncated, winners, owned) = rollout_data
    del rollout_data

    p_gae = jax.pmap(lambda r, v, nv, te, tr: compute_gae(
        r, v, nv, te, tr, cfg.gamma, cfg.gae_lambda))
    advs = p_gae(rews, vals, next_vals, terminated, truncated)
    rets = advs + vals
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    train_mask = 1.0 - truncated.astype(jnp.float32)

    per_device_total = T * 2 * N
    n_keep = int(per_device_total * cfg.adv_top_frac)
    n_keep = (n_keep // cfg.minibatch_size) * cfg.minibatch_size
    if n_keep == 0:
        raise SystemExit(
            f"minibatch_size {cfg.minibatch_size} exceeds the kept-sample "
            f"count {int(per_device_total * cfg.adv_top_frac)}")
    p_top = jax.pmap(lambda a: jax.lax.top_k(jnp.abs(a.reshape(-1)), n_keep)[1])
    sample_idx = p_top(advs)
    log(f"minibatches per epoch: {n_keep // cfg.minibatch_size} "
        f"({n_keep:,} kept of {per_device_total:,})")

    batch = (obs, move_masks, build_masks, temporal, actions, lps,
             advs, rets, train_mask)

    def _ppo_step(prm, ost, bt, key, sidx):
        net = eqx.combine(prm, static)
        net, ost, result = ppo_update(
            net, ost, bt, optimizer, key, cfg.clip_eps, cfg.vf_coef,
            0.01, cfg.minibatch_size, value_loss_fn, sidx)
        new_params, _ = eqx.partition(net, eqx.is_array)
        return new_params, ost, jax.lax.pmean(result, axis_name="devices")

    p_ppo_step = jax.pmap(_ppo_step, axis_name="devices")

    log("compiling PPO update ...")
    t0 = time.perf_counter()
    up, uo, res = p_ppo_step(p_params, p_opt_state, batch, keys, sample_idx)
    jax.block_until_ready(res["total_loss"])
    log(f"ppo compile+run {time.perf_counter() - t0:.1f}s")

    # ---- timed region under the profiler -----------------------------
    log(f"tracing {ARGS.reps} rollout+PPO pairs ...")
    wall0 = time.perf_counter()
    with jax.profiler.trace(ARGS.trace_dir):
        for _ in range(ARGS.reps):
            st, rd, _, _, _ = p_rollout(
                p_params, states, keys, osp0, osp1, pool_rep)
            jax.block_until_ready(st)
            del rd
            up, uo, res = p_ppo_step(p_params, p_opt_state, batch,
                                     keys, sample_idx)
            jax.block_until_ready(res["total_loss"])
    traced_wall = time.perf_counter() - wall0
    log(f"traced wall {traced_wall:.2f}s for {ARGS.reps} pairs "
        f"({traced_wall / ARGS.reps:.2f}s per rollout+PPO)")

    tr = analyze_trace(ARGS.trace_dir)
    log("=" * 60)
    log("DEVICE KERNEL SPLIT")
    log(f"  device time total: {tr.get('device_us_total', 0) / 1e6:.2f}s "
        f"of {traced_wall:.2f}s wall")
    for k, v in (tr.get("by_class_pct") or {}).items():
        log(f"    {k:<22} {v:>5.1f}%")
    log("  top kernels:")
    for row in (tr.get("top_kernels") or [])[:20]:
        log(f"    {row['pct']:>5.2f}%  {row['class']:<20} {row['name']}")
    log("=" * 60)

    mem = dev.memory_stats() or {}
    result = {
        "device": str(dev), "compute_capability": str(cc),
        "jax": jax.__version__, "config": os.path.basename(ARGS.config),
        "depth": cfg.depth, "embed_dim": cfg.embed_dim,
        "ff_factor": cfg.ff_factor, "n_head": cfg.n_head,
        "head_dim": cfg.embed_dim // cfg.n_head,
        "num_envs": N, "num_steps": T, "minibatch_size": cfg.minibatch_size,
        "minibatches": n_keep // cfg.minibatch_size,
        "n_params": n_params, "use_bf16": cfg.use_bf16, "devices": ND,
        "traced_wall_s": round(traced_wall, 3), "reps": ARGS.reps,
        "s_per_rollout_ppo_pair": round(traced_wall / ARGS.reps, 3),
        "peak_device_gib": round(mem.get("peak_bytes_in_use", 0) / 2**30, 2),
        "hlo": hlo, "trace": tr,
    }
    with open(ARGS.out, "w") as f:
        json.dump(result, f, indent=2)
    log(f"results -> {ARGS.out}")
    print("PROBE_JSON_BEGIN", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)
    print("PROBE_JSON_END", flush=True)


if __name__ == "__main__":
    main()
