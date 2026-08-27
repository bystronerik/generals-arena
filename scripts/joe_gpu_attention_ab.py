#!/usr/bin/env python3
"""Same-host A/B of attention rewrites on the joe rollout, done correctly.

The first attempt at this measurement (2026-08-27) was invalid: it called
the ``@jax.jit``-decorated ``collect_rollout``, and a fresh ``pmap`` wrapper
per arm still hits the **nested jit cache**. The cache key is the callee's
identity plus static args and input avals, none of which change when a
Python method is monkeypatched, so every arm ran the first arm's binary and
all four arms timed within 0.16 % of each other. That reads as "no effect"
for any change.

Two defences here:

1. The scan body is **inlined**, so no jitted callee sits between the patch
   and the trace. ``jax.clear_caches()`` runs before each arm as well.
2. A **compiled-program guard**: each arm is lowered and compiled ahead of
   time, and its optimized HLO is hashed. Two arms sharing a hash are the
   same binary, which is exactly the failure above, and the run aborts
   instead of reporting a false null. The guard deliberately does *not*
   compare numeric output: folding q/k/v concatenates the weights along the
   **output** dimension, not the contraction dimension, so every element
   keeps the same dot product and the same accumulation order and the
   result is bit-identical by construction. A numeric guard flags that
   correct arm as a failure (it did, on the first attempt).

Arms, identical weights, X16 shape:

  base   the shipped MultiHeadSelfAttention (4 transposes, f32 softmax, a
         materialized (heads, seq, seq) score tensor)
  cudnn  jax.nn.dot_product_attention(implementation="cudnn"): flash
         attention, no score tensor in HBM, and no head transposes at all
  qkv    q/k/v folded into one GEMM by concatenating the three weight
         matrices at forward time (a computation, not a new Module field,
         so checkpoints and the joe-rs export layout are untouched)

Context: the 16 transformer blocks are 97.4 % of the rollout and 98.3 % of
the PPO update, so anything that lands inside a block is worth measuring.
"""
from __future__ import annotations

import hashlib
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

import equinox as eqx          # noqa: E402
import jax                     # noqa: E402
import jax.numpy as jnp        # noqa: E402
import jax.random as jrandom   # noqa: E402

from training.joe.config import Config                                # noqa: E402
from training.joe.env import (make_competition_env,                   # noqa: E402
                              preset_min_generals_distance)
from training.joe.networks import build_network, get_network_bundle   # noqa: E402
from training.joe.networks.common import reset_done_envs              # noqa: E402
from training.joe.networks.transformer import MultiHeadSelfAttention  # noqa: E402
from training.joe.train.ppo import _replicate                         # noqa: E402
from training.joe.train.rollout_selfplay import _observe_both         # noqa: E402

CONFIG = os.environ.get("JOE_CONFIG") or f"{REPO}/training/joe/configs/X16.yaml"
POOL = int(os.environ.get("JOE_POOL", "8192"))
T = int(os.environ.get("JOE_T", "64"))
ENVS = int(os.environ.get("JOE_ENVS", "0"))
REPS = int(os.environ.get("JOE_REPS", "1"))
ROUNDS = int(os.environ.get("JOE_ROUNDS", "3"))
OUT = os.environ.get("JOE_OUT", "/tmp/joe-attn-ab.json")

_ORIG = MultiHeadSelfAttention.__call__


def log(m):
    print(f"[attn-ab] {m}", flush=True)


def _qkv(self, x, fused):
    seq = x.shape[0]
    if fused:
        W = jnp.concatenate(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], axis=0)
        qkv = x @ W.T
        if self.q_proj.bias is not None:
            qkv = qkv + jnp.concatenate(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0)
        d = self.n_head * self.head_dim
        parts = (qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:])
    else:
        parts = (jax.vmap(self.q_proj)(x), jax.vmap(self.k_proj)(x),
                 jax.vmap(self.v_proj)(x))
    return tuple(p.reshape(seq, self.n_head, self.head_dim) for p in parts)


def make_attn(fused_qkv, use_cudnn):
    def __call__(self, x):
        seq = x.shape[0]
        q, k, v = _qkv(self, x, fused_qkv)
        scale = 1.0 / math.sqrt(self.head_dim)
        if use_cudnn:
            o = jax.nn.dot_product_attention(
                q[None], k[None], v[None], scale=scale,
                implementation="cudnn")[0]
            out = o.reshape(seq, -1)
        else:
            qt, kt, vt = (jnp.transpose(t, (1, 0, 2)) for t in (q, k, v))
            attn = jnp.matmul(qt, jnp.transpose(kt, (0, 2, 1))) * scale
            attn = jax.nn.softmax(attn.astype(jnp.float32),
                                  axis=-1).astype(qt.dtype)
            out = jnp.transpose(jnp.matmul(attn, vt), (1, 0, 2)).reshape(seq, -1)
        return jax.vmap(self.out_proj)(out)
    return __call__


ARMS = {
    "base":  _ORIG,
    "cudnn": make_attn(fused_qkv=False, use_cudnn=True),
    "qkv":   make_attn(fused_qkv=True,  use_cudnn=False),
}


def main():
    dev = jax.devices()[0]
    ND = jax.device_count()
    cfg = Config.from_yaml(CONFIG)
    object.__setattr__(cfg, "pool_size", POOL)
    if ENVS:
        object.__setattr__(cfg, "num_envs", ENVS)
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    N = cfg.num_envs
    hd = cfg.embed_dim // cfg.n_head
    log(f"device={dev} cc={getattr(dev, 'compute_capability', '?')} "
        f"depth={cfg.depth} embed={cfg.embed_dim} heads={cfg.n_head} "
        f"head_dim={hd} (cudnn needs hd%8==0 -> {hd % 8 == 0})")
    log(f"{N} envs x {T} steps")

    bundle = get_network_bundle(cfg.network)
    augment_fn = bundle["augment_obs"]
    init_obs = bundle["init_obs_state"]

    env = make_competition_env(preset_min_generals_distance(), None, POOL)
    t0 = time.perf_counter()
    pool, _ = env.reset(jrandom.PRNGKey(0))
    jax.block_until_ready(pool.armies)
    log(f"pool in {time.perf_counter() - t0:.1f}s")

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

    def build():
        """The scan body inlined -- no jitted callee between patch and trace."""
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
            # Fingerprint: the network's value output. Any real change to the
            # attention arithmetic moves it; if two arms match bit for bit,
            # the monkeypatch did not take.
            return s, jnp.sum(data[6].astype(jnp.float32))
        return jax.pmap(_r)

    fns, compile_s, fp, sig, failed = {}, {}, {}, {}, {}
    args = (p_params, states0, keys, osp0, pool_rep)
    for arm, impl in ARMS.items():
        log(f"compiling {arm} ...")
        jax.clear_caches()
        MultiHeadSelfAttention.__call__ = impl
        t0 = time.perf_counter()
        try:
            # Ahead-of-time: one compile per arm, reused for the signature
            # and for the timed calls.
            compiled = build().lower(*args).compile()
            txt = compiled.as_text()
            sig[arm] = (hashlib.md5(txt.encode()).hexdigest()[:12],
                        txt.count("__cublas$lt$matmul"),
                        txt.count("__cudnn$fmha"))
            s_out, fingerprint = compiled(*args)
            jax.block_until_ready(s_out.armies)
            fns[arm] = compiled
            fp[arm] = float(jnp.asarray(fingerprint).ravel()[0])
            compile_s[arm] = round(time.perf_counter() - t0, 1)
            log(f"  {arm} compiled+ran {compile_s[arm]}s  hlo={sig[arm][0]} "
                f"cublas_gemm={sig[arm][1]} cudnn_fmha={sig[arm][2]} "
                f"value_sum={fp[arm]:.4f}")
            del s_out
        except Exception as e:
            failed[arm] = f"{type(e).__name__}: {str(e)[:200]}"
            log(f"  {arm} FAILED {failed[arm]}")
        finally:
            MultiHeadSelfAttention.__call__ = _ORIG

    # --- the guard: distinct binaries, not distinct numbers --------------
    dupes = [(a, b) for i, a in enumerate(sig) for b in list(sig)[i + 1:]
             if sig[a][0] == sig[b][0]]
    if dupes:
        log("=" * 62)
        log("ABORT: arms compiled to identical HLO -> the patch did not take "
            "effect and any timing below would be meaningless.")
        for a, b in dupes:
            log(f"  {a} == {b}  (hlo {sig[a][0]})")
        log("=" * 62)
        json.dump({"aborted": "identical HLO", "signatures": sig},
                  open(OUT, "w"), indent=2)
        print("ATTN_JSON_END", flush=True)
        return
    log(f"guard passed: {len(sig)} distinct compiled programs")

    times = {a: [] for a in fns}
    for rnd in range(ROUNDS):
        for arm, f in fns.items():
            t0 = time.perf_counter()
            for _ in range(REPS):
                out = f(p_params, states0, keys, osp0, pool_rep)
                jax.block_until_ready(out[0].armies)
            dt = (time.perf_counter() - t0) / REPS
            del out
            times[arm].append(round(dt, 4))
            log(f"round {rnd} {arm:<6} {dt:.3f}s")

    best = {a: min(v) for a, v in times.items()}
    b0 = best.get("base")
    log("=" * 62)
    log(f"ATTENTION A/B (best of {ROUNDS} rounds, T={T})")
    for a, v in best.items():
        log(f"  {a:<6} {v:8.3f}s   {b0 / v:.3f}x vs base   "
            f"gemm={sig[a][1]} fmha={sig[a][2]}")
    log("=" * 62)

    res = {"device": str(dev), "cc": str(getattr(dev, "compute_capability", "?")),
           "config": os.path.basename(CONFIG), "num_envs": N, "num_steps": T,
           "depth": cfg.depth, "embed_dim": cfg.embed_dim, "head_dim": hd,
           "rounds": ROUNDS, "reps": REPS, "times_s": times, "best_s": best,
           "speedup_vs_base": {a: round(b0 / v, 4) for a, v in best.items()},
           "fingerprints": fp, "signatures": {k: list(v) for k, v in sig.items()},
           "compile_s": compile_s, "failed": failed,
           "peak_device_gib": round((dev.memory_stats() or {}).get(
               "peak_bytes_in_use", 0) / 2**30, 2)}
    json.dump(res, open(OUT, "w"), indent=2)
    print("ATTN_JSON_BEGIN", flush=True)
    print(json.dumps(res, indent=2), flush=True)
    print("ATTN_JSON_END", flush=True)


if __name__ == "__main__":
    main()
