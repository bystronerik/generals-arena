#!/usr/bin/env python3
"""Same-host A/B of attention rewrites on the joe PPO update (fwd+bwd).

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

cuDNN flash attention is worth +5.1 % on the rollout (forward only). The
PPO update is where the **backward** pass runs, and the blocks are 98.3 %
of it, so this is the other half of that result.

``ppo_update`` is a plain function, not ``@jax.jit``, so the monkeypatch
reaches the trace. The compiled-program guard runs anyway.
"""
from __future__ import annotations

import hashlib
import json
import traceback
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
from training.joe.train.ppo import (_replicate, compute_gae,           # noqa: E402
                                    make_value_loss_fn, ppo_update)
import optax                                                          # noqa: E402
from training.joe.train.rollout_selfplay import _observe_both         # noqa: E402

CONFIG = os.environ.get("JOE_CONFIG") or f"{REPO}/training/joe/configs/X16.yaml"
POOL = int(os.environ.get("JOE_POOL", "8192"))
T = int(os.environ.get("JOE_T", "64"))
ENVS = int(os.environ.get("JOE_ENVS", "0"))
REPS = int(os.environ.get("JOE_REPS", "1"))
MINIBATCH = int(os.environ.get("JOE_MINIBATCH", "0"))
ROUNDS = int(os.environ.get("JOE_ROUNDS", "3"))
OUT = os.environ.get("JOE_OUT", "/tmp/joe-ppo-attn-ab.json")

_ORIG = MultiHeadSelfAttention.__call__


def log(m):
    print(f"[ppo-attn] {m}", flush=True)


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


_ALL_ARMS = {
    "base":  _ORIG,
    "cudnn": make_attn(fused_qkv=False, use_cudnn=True),
    "qkv":   make_attn(fused_qkv=True,  use_cudnn=False),
}
ARMS = {k: _ALL_ARMS[k]
        for k in os.environ.get("JOE_ARMS", "base,cudnn").split(",")
        if k in _ALL_ARMS}


def main():
    dev = jax.devices()[0]
    ND = jax.device_count()
    cfg = Config.from_yaml(CONFIG)
    object.__setattr__(cfg, "pool_size", POOL)
    if ENVS:
        object.__setattr__(cfg, "num_envs", ENVS)
    if MINIBATCH:
        object.__setattr__(cfg, "minibatch_size", MINIBATCH)
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

    # One rollout with the shipped attention supplies identical PPO input
    # for every arm.
    def _roll(prm, st, key, osp, pl):
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
            return (ns, key, o2), (oa, mm, bm, tp, acts, lps, vals,
                                   jnp.concatenate([ts.terminated, ts.terminated]),
                                   jnp.concatenate([ts.truncated, ts.truncated]))
        (s, k, o), data = jax.lax.scan(body, (st, key, osp), None, length=T)
        return data

    log("collecting one rollout for PPO input ...")
    t0 = time.perf_counter()
    obs, mm, bm, tp, acts, lps, vals, term, trunc = jax.pmap(_roll)(
        p_params, states0, keys, osp0, pool_rep)
    jax.block_until_ready(obs)
    log(f"rollout {time.perf_counter() - t0:.1f}s")

    optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                            optax.adam(cfg.lr))
    opt_state = optimizer.init(params)
    p_opt = _replicate(opt_state, ND)
    vlf = make_value_loss_fn(cfg)

    nv = jnp.concatenate([vals[:, 1:], vals[:, -1:]], axis=1)
    advs = jax.pmap(lambda r, v, n_, te, tr: compute_gae(
        r, v, n_, te, tr, cfg.gamma, cfg.gae_lambda))(
        jnp.zeros_like(vals), vals, nv, term, trunc)
    rets = advs + vals
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    tmask = 1.0 - trunc.astype(jnp.float32)
    per_dev = T * 2 * N
    nk = (int(per_dev * cfg.adv_top_frac) // cfg.minibatch_size) * cfg.minibatch_size
    if nk == 0:
        raise SystemExit(
            f"minibatch_size {cfg.minibatch_size} exceeds the kept-sample "
            f"count {int(per_dev * cfg.adv_top_frac)}; set JOE_MINIBATCH")
    sidx = jax.pmap(lambda a: jax.lax.top_k(jnp.abs(a.reshape(-1)), nk)[1])(advs)
    batch = (obs, mm, bm, tp, acts, lps, advs, rets, tmask)
    log(f"minibatches {nk // cfg.minibatch_size} of {cfg.minibatch_size}")

    def build():
        def _step(prm, ost, bt, key, si):
            n_ = eqx.combine(prm, static)
            n_, ost, r = ppo_update(n_, ost, bt, optimizer, key, cfg.clip_eps,
                                    cfg.vf_coef, 0.01, cfg.minibatch_size,
                                    vlf, si)
            np_, _ = eqx.partition(n_, eqx.is_array)
            return np_, ost, jax.lax.pmean(r, axis_name="devices")
        return jax.pmap(_step, axis_name="devices")

    fns, compile_s, fp, sig, failed = {}, {}, {}, {}, {}
    args = (p_params, p_opt, batch, keys, sidx)
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
            _p, _o, res = compiled(*args)
            jax.block_until_ready(res["total_loss"])
            fns[arm] = compiled
            fp[arm] = float(jnp.asarray(res["total_loss"]).ravel()[0])
            compile_s[arm] = round(time.perf_counter() - t0, 1)
            log(f"  {arm} compiled+ran {compile_s[arm]}s  hlo={sig[arm][0]} "
                f"cublas_gemm={sig[arm][1]} cudnn_fmha={sig[arm][2]} "
                f"value_sum={fp[arm]:.4f}")
            del _p, _o, res
        except Exception as e:
            failed[arm] = f"{type(e).__name__}: {str(e)[:200]}"
            log(f"  {arm} FAILED {failed[arm]}")
        finally:
            MultiHeadSelfAttention.__call__ = _ORIG

    # --- the guard: distinct binaries, not distinct numbers --------------
    if len(sig) < 2:
        log("=" * 62)
        log(f"ABORT: only {len(sig)} arm(s) compiled; nothing to compare.")
        for a, w in failed.items():
            log(f"  {a}: {w}")
        log("=" * 62)
        json.dump({"aborted": "too few arms", "failed": failed},
                  open(OUT, "w"), indent=2)
        print("PPOATTN_JSON_END", flush=True)
        return
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
        print("PPOATTN_JSON_END", flush=True)
        return
    log(f"guard passed: {len(sig)} distinct compiled programs")

    # NOTE: these are the PPO step's arguments. An earlier revision called the
    # compiled PPO step with the rollout's arguments here, which threw during
    # the timed rounds and lost the whole run. Guarded so a failure cannot do
    # that again.
    times = {a: [] for a in fns}
    for rnd in range(ROUNDS):
        for arm, f in fns.items():
            try:
                t0 = time.perf_counter()
                for _ in range(REPS):
                    out = f(*args)
                    jax.block_until_ready(out[2]["total_loss"])
                dt = (time.perf_counter() - t0) / REPS
                del out
                times[arm].append(round(dt, 4))
                log(f"round {rnd} {arm:<6} {dt:.3f}s")
            except Exception as e:
                log(f"round {rnd} {arm:<6} FAILED {type(e).__name__}: {e}")
                log(traceback.format_exc())
                failed[f"{arm}@r{rnd}"] = f"{type(e).__name__}: {str(e)[:300]}"
    times = {a: v for a, v in times.items() if v}
    if not times:
        log("ABORT: no arm produced a timing")
        json.dump({"aborted": "no timings", "failed": failed,
                   "signatures": {k: list(v) for k, v in sig.items()}},
                  open(OUT, "w"), indent=2)
        print("PPOATTN_JSON_END", flush=True)
        return

    best = {a: min(v) for a, v in times.items()}
    b0 = best.get("base") or min(best.values())
    log("=" * 62)
    log(f"PPO-UPDATE ATTENTION A/B (best of {ROUNDS} rounds, T={T})")
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
    print("PPOATTN_JSON_BEGIN", flush=True)
    print(json.dumps(res, indent=2), flush=True)
    print("PPOATTN_JSON_END", flush=True)


if __name__ == "__main__":
    main()
