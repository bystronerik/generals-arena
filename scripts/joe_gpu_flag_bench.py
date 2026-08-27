#!/usr/bin/env python3
"""Time one joe rollout under whatever XLA_FLAGS this process was given.

XLA flags are read when the backend initializes, so a flag sweep cannot
happen inside one process -- the driver runs this script once per flag
combination and compares the printed lines. Everything else is held fixed:
same config, same pool seed, same shape, same machine, back to back.

    XLA_FLAGS=--xla_gpu_triton_gemm_any=true python scripts/joe_gpu_flag_bench.py

Prints one ``FLAGBENCH`` line of JSON so the driver can collect results
without parsing prose. The rollout body is inlined rather than calling the
jitted ``collect_rollout`` (see joe_gpu_attention_ab.py for why that
matters).
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO = os.environ.get("JOE_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import equinox as eqx          # noqa: E402
import jax                     # noqa: E402
import jax.numpy as jnp        # noqa: E402
import jax.random as jrandom   # noqa: E402

from training.joe.config import Config                                # noqa: E402
from training.joe.env import (make_competition_env,                   # noqa: E402
                              preset_min_generals_distance)
from training.joe.networks import build_network, get_network_bundle   # noqa: E402
from training.joe.networks.common import reset_done_envs              # noqa: E402
from training.joe.train.ppo import _replicate                         # noqa: E402
from training.joe.train.rollout_selfplay import _observe_both         # noqa: E402

CONFIG = os.environ.get("JOE_CONFIG") or f"{REPO}/training/joe/configs/X16.yaml"
POOL = int(os.environ.get("JOE_POOL", "2048"))
T = int(os.environ.get("JOE_T", "32"))
ENVS = int(os.environ.get("JOE_ENVS", "0"))
REPS = int(os.environ.get("JOE_REPS", "2"))
LABEL = os.environ.get("JOE_LABEL", "baseline")


def main():
    dev = jax.devices()[0]
    ND = jax.device_count()
    cfg = Config.from_yaml(CONFIG)
    object.__setattr__(cfg, "pool_size", POOL)
    if ENVS:
        object.__setattr__(cfg, "num_envs", ENVS)
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    N = cfg.num_envs
    bundle = get_network_bundle(cfg.network)
    augment_fn = bundle["augment_obs"]
    init_obs = bundle["init_obs_state"]

    env = make_competition_env(preset_min_generals_distance(), None, POOL)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    jax.block_until_ready(pool.armies)

    net = build_network(cfg, jrandom.PRNGKey(1))
    params, static = eqx.partition(net, eqx.is_array)
    p_params = _replicate(params, ND)
    pool_rep = _replicate(pool, ND)
    states0 = jax.pmap(lambda k: jax.vmap(env.init_state)(
        jrandom.split(k, N)))(jrandom.split(jrandom.PRNGKey(2), ND))
    single = init_obs(cfg.pad_to, cfg.pad_to)
    osp0 = _replicate(jax.tree.map(
        lambda x: jnp.tile(x, (2 * N, *([1] * x.ndim))), single), ND)
    keys = jrandom.split(jrandom.PRNGKey(3), ND)

    def _r(prm, st, key, osp, pl):
        def body(carry, _):
            states, key, o = carry
            _, obs_arr, cost, mm, bm = _observe_both(states)
            oa, no = jax.vmap(augment_fn)(obs_arr, cost, o)
            oa = oa.astype(jnp.bfloat16)
            tp = jnp.stack([no.opponent_army_history,
                            no.opponent_land_history], axis=1)
            ks = jrandom.split(key, 2 * N + 1)
            key = ks[0]
            net_ = eqx.combine(prm, static)
            acts, vals, lps, _, _, _ = jax.vmap(
                net_, in_axes=(0, 0, 0, 0, 0, None))(
                oa, mm, bm, tp, ks[1:], None)
            ea = jnp.stack([acts[:N], acts[N:]], axis=1)
            ts, ns = jax.vmap(lambda s, a: env.step(s, a, pl))(states, ea)
            d = ts.terminated | ts.truncated
            o2 = reset_done_envs(no, jnp.concatenate([d, d]))
            return (ns, key, o2), (oa, mm, bm, tp, acts, lps, vals)
        (s, k, o), data = jax.lax.scan(body, (st, key, osp), None, length=T)
        return s, jnp.sum(data[6].astype(jnp.float32))

    args = (p_params, states0, keys, osp0, pool_rep)
    t0 = time.perf_counter()
    f = jax.pmap(_r)
    s_out, fp = f(*args)
    jax.block_until_ready(s_out.armies)
    compile_s = time.perf_counter() - t0
    fp = float(jnp.asarray(fp).ravel()[0])
    del s_out

    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = f(*args)
        jax.block_until_ready(out[0].armies)
        times.append(time.perf_counter() - t0)
        del out

    rec = {
        "label": LABEL,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "best_s": round(min(times), 4),
        "all_s": [round(t, 4) for t in times],
        "compile_s": round(compile_s, 1),
        "value_sum": fp,
        "num_envs": N, "num_steps": T,
        "device": str(dev),
        "peak_gib": round((dev.memory_stats() or {}).get(
            "peak_bytes_in_use", 0) / 2**30, 2),
    }
    print("FLAGBENCH " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
