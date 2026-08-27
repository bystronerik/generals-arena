#!/usr/bin/env python3
"""Ablate the transformer trunk out of the joe rollout, on one CUDA GPU.

Arms, all on identical weights at the X16 production shape. Only
``SelfAttentionLayer.__call__`` is swapped; nothing in ``training/joe`` is
edited on disk:

  base     the shipped pre-norm block (LN -> attention -> LN -> FF)
  no_silu  the same block with the feed-forward ``silu`` removed
  none     the identity -- the whole depth-16 trunk does no work

``base - none`` is the trunk's share of the rollout. The result on an RTX
PRO 6000 is that the share is 0.09 %: the rollout does not depend on the
network at all. The same patch applied to the PPO update, in
``joe_gpu_ppo_ablation.py``, takes it from 13.6 s to 0.22 s.

Arms compile first, then run in interleaved rounds, so host drift reaches
every arm equally. Read the last round, not the first: the first timed arm
of round 0 carries a warm-up cost of about 1 s and reads as a false win for
whatever runs after it.

Results: docs/research/measurements/joe-transformer-kernel-probe.md
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

REPO = os.environ.get("JOE_REPO") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from training.joe.config import Config
from training.joe.env import make_competition_env, preset_min_generals_distance
from training.joe.networks import build_network, get_network_bundle
from training.joe.networks.transformer import MultiHeadSelfAttention
from training.joe.train.ppo import _replicate
from training.joe.train.rollout_selfplay import collect_rollout

CONFIG = os.environ.get("JOE_AB_CONFIG",
                        f"{REPO}/training/joe/configs/X16.yaml")
POOL = int(os.environ.get("JOE_AB_POOL", "8192"))
REPS = int(os.environ.get("JOE_AB_REPS", "2"))
ROUNDS = int(os.environ.get("JOE_AB_ROUNDS", "3"))

from training.joe.networks.transformer import SelfAttentionLayer

_ORIG_LAYER = SelfAttentionLayer.__call__


def log(m):
    print(f"[ab] {m}", flush=True)


# --- sublayer ablations -------------------------------------------------
# Semantics change; shapes do not. The point is cost attribution: what does
# the rollout still cost with one sublayer removed? base - arm = that
# sublayer's share. "none" removes the whole block, so base - none is the
# entire depth-16 trunk and what remains is env + observations +
# augmentation + patch embed + heads.

def layer_ff_only(self, x):
    h = jax.vmap(self.norm2)(x)
    h = jax.nn.silu(jax.vmap(self.ff_linear1)(h))
    return x + jax.vmap(self.ff_linear2)(h)


def layer_attn_only(self, x):
    return x + self.attn(jax.vmap(self.norm1)(x))


def layer_no_silu(self, x):
    x = x + self.attn(jax.vmap(self.norm1)(x))
    h = jax.vmap(self.norm2)(x)
    h = jax.vmap(self.ff_linear1)(h)          # silu removed
    return x + jax.vmap(self.ff_linear2)(h)


def layer_identity(self, x):
    return x


ARMS = {
    "base":    _ORIG_LAYER,
    "no_silu": layer_no_silu,
    "none":    layer_identity,
}


def main():
    dev = jax.devices()[0]
    ND = jax.device_count()
    log(f"device={dev} cc={getattr(dev, 'compute_capability', '?')} "
        f"jax={jax.__version__}")

    cfg = Config.from_yaml(CONFIG)
    object.__setattr__(cfg, "pool_size", POOL)
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    N, T = cfg.num_envs, cfg.num_steps
    hd = cfg.embed_dim // cfg.n_head
    log(f"X16 depth={cfg.depth} embed={cfg.embed_dim} ff={cfg.ff_factor} "
        f"heads={cfg.n_head} head_dim={hd} (cudnn needs hd%8==0 -> "
        f"{hd % 8 == 0})")
    log(f"shape {N} envs x {T} steps, {2 * N * T:,} samples/iter")

    bundle = get_network_bundle(cfg.network)
    augment_fn = bundle["augment_obs"]
    init_obs_state_fn = bundle["init_obs_state"]

    t0 = time.perf_counter()
    env = make_competition_env(preset_min_generals_distance(), None, POOL)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    jax.block_until_ready(pool.armies)
    log(f"pool in {time.perf_counter() - t0:.1f}s")

    network = build_network(cfg, jrandom.PRNGKey(1))
    params, static = eqx.partition(network, eqx.is_array)
    p_params = _replicate(params, ND)
    pool_rep = _replicate(pool, ND)

    p_init = jax.pmap(lambda k: jax.vmap(env.init_state)(jrandom.split(k, N)))
    states = p_init(jrandom.split(jrandom.PRNGKey(2), ND))
    single = init_obs_state_fn(cfg.pad_to, cfg.pad_to)
    batched = jax.tree.map(lambda x: jnp.tile(x, (N, *([1] * x.ndim))), single)
    osp0 = _replicate(batched, ND)
    osp1 = _replicate(batched, ND)
    keys = jrandom.split(jrandom.PRNGKey(3), ND)

    def build(arm):
        SelfAttentionLayer.__call__ = ARMS[arm]

        def _rollout(prm, st, key, a, b, pl):
            net = eqx.combine(prm, static)
            return collect_rollout(st, env, net, key, T, a, b, augment_fn, pl)
        return jax.pmap(_rollout)

    fns, compile_s, failed = {}, {}, {}
    for arm in ARMS:
        log(f"compiling {arm} ...")
        t0 = time.perf_counter()
        try:
            f = build(arm)
            out = f(p_params, states, keys, osp0, osp1, pool_rep)
            jax.block_until_ready(out[0].armies)
            fns[arm] = f
            compile_s[arm] = round(time.perf_counter() - t0, 1)
            log(f"  {arm} compiled+ran in {compile_s[arm]}s")
            del out
        except Exception as e:
            failed[arm] = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"  {arm} FAILED: {failed[arm]}")
    SelfAttentionLayer.__call__ = _ORIG_LAYER

    times = {a: [] for a in fns}
    for rnd in range(ROUNDS):
        for arm, f in fns.items():
            t0 = time.perf_counter()
            for _ in range(REPS):
                out = f(p_params, states, keys, osp0, osp1, pool_rep)
                jax.block_until_ready(out[0].armies)
            dt = (time.perf_counter() - t0) / REPS
            del out
            times[arm].append(round(dt, 4))
            log(f"round {rnd} {arm:<6} {dt:.3f}s per rollout")

    best = {a: min(v) for a, v in times.items()}
    base = best.get("base")
    log("=" * 58)
    log("ROLLOUT SUBLAYER ABLATION (best of rounds)")
    for arm, v in best.items():
        rel = f"{base / v:.3f}x vs base" if base else ""
        log(f"  {arm:<6} {v:8.3f}s   {rel}")
    for arm, why in failed.items():
        log(f"  {arm:<6} FAILED  {why}")
    log("=" * 58)

    mem = dev.memory_stats() or {}
    res = {
        "device": str(dev), "cc": str(getattr(dev, "compute_capability", "?")),
        "config": os.path.basename(CONFIG), "num_envs": N, "num_steps": T,
        "depth": cfg.depth, "embed_dim": cfg.embed_dim, "head_dim": hd,
        "reps": REPS, "rounds": ROUNDS,
        "times_s": times, "best_s": best, "compile_s": compile_s,
        "failed": failed,
        "speedup_vs_base": {a: round(base / v, 4) for a, v in best.items()}
        if base else {},
        "peak_device_gib": round(mem.get("peak_bytes_in_use", 0) / 2**30, 2),
    }
    with open(os.environ.get("JOE_OUT", "/tmp/joe-trunk-ablate.json"),
              "w") as f:
        json.dump(res, f, indent=2)
    print("AB_JSON_BEGIN", flush=True)
    print(json.dumps(res, indent=2), flush=True)
    print("AB_JSON_END", flush=True)


if __name__ == "__main__":
    main()
