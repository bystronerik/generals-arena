#!/usr/bin/env python3
"""Bisect the joe rollout: what is the 16.1 s, given it is not the network?

The trunk ablation showed removing all 16 transformer blocks leaves the
rollout time unchanged, while the same patch collapses the PPO update from
13.6 s to 0.22 s. So the rollout's cost is somewhere else. The earlier T4
drill-down put env + observations + augmentation at 0.1 % of a rollout --
but that drill-down's scans returned **only carries**, while the real
``collect_rollout`` stacks twelve per-step outputs, of which ``obs_aug``
alone is (2N, 39, 21, 21) bf16 = 140.9 MB per step, 18.0 GB over 128 steps.

Arms (a faithful re-implementation of ``collect_rollout``'s scan body, with
stages switched off; the ``states`` carry chain is kept intact in every arm
so nothing is hoisted out of the loop or dead-coded):

  full        everything, stacking all 12 outputs   -- should match ~16.1 s
  no_stack    identical compute, scan returns one scalar per step
  env_stack   fixed action, no obs/augment/network, still stacks obs-shaped data
  env_only    fixed action, no obs/augment/network, no stacking

  full - no_stack   = the cost of stacking the rollout outputs
  no_stack          = the compute floor (env + obs + augment + net)
  env_only          = env.step alone

Run at two horizons: if the stacking cost doubles with T it is bandwidth; if
it quadruples, XLA is copying the whole accumulator each step (O(T^2)) and
that is a fixable pathology, not a law of physics.
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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from training.joe.config import Config
from training.joe.env import make_competition_env, preset_min_generals_distance
from training.joe.networks import build_network, get_network_bundle
from training.joe.networks.common import reset_done_envs
from training.joe.train.ppo import _replicate
from training.joe.train.rewards import win_lose_reward
from training.joe.train.rollout_selfplay import _observe_both

REPS = int(os.environ.get("JOE_REPS", "2"))
CONFIG = os.environ.get("JOE_CONFIG", "")
ENVS = int(os.environ.get("JOE_ENVS", "0"))
TLONG = int(os.environ.get("JOE_TLONG", "128"))
TSHORT = int(os.environ.get("JOE_TSHORT", "64"))
POOL = int(os.environ.get("JOE_POOL", "8192"))
OUT = os.environ.get("JOE_OUT", "/tmp/joe-bisect.json")
ROUNDS = int(os.environ.get("JOE_ROUNDS", "3"))


def log(m):
    print(f"[bisect] {m}", flush=True)


def main():
    dev = jax.devices()[0]
    ND = jax.device_count()
    log(f"device={dev} cc={getattr(dev, 'compute_capability', '?')}")

    cfg = Config.from_yaml(CONFIG or f"{REPO}/training/joe/configs/X16.yaml")
    object.__setattr__(cfg, "pool_size", POOL)
    if ENVS:
        object.__setattr__(cfg, "num_envs", ENVS)
    jax.config.update("jax_default_matmul_precision", "tensorfloat32")
    N = cfg.num_envs
    bundle = get_network_bundle(cfg.network)
    augment_fn = bundle["augment_obs"]
    init_obs = bundle["init_obs_state"]

    env = make_competition_env(preset_min_generals_distance(), None,
                               cfg.pool_size)
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

    obs_gb = TLONG * 2 * N * 39 * cfg.pad_to * cfg.pad_to * 2 / 2**30
    log(f"obs_aug stack at T={TLONG}: {obs_gb:.2f} GiB (N={N})")

    TWIN = int(single.opponent_army_history.shape[0])

    def build(T, do_obs, do_net, do_stack, do_augment=True):
        C, P = 39, cfg.pad_to

        def _r(prm, st, key, osp, pl):
            def body(carry, _):
                states, key, o = carry
                if do_obs and do_augment:
                    _, obs_arr, cost, mm, bm = _observe_both(states)
                    oa, no = jax.vmap(augment_fn)(obs_arr, cost, o)
                    oa = oa.astype(jnp.bfloat16)
                    tp = jnp.stack([no.opponent_army_history,
                                    no.opponent_land_history], axis=1)
                elif do_obs:
                    # Observations run; only augment_obs is skipped. Build
                    # obs_aug by concatenation so every element of obs_arr
                    # and cost is read -- a scalar tie would let XLA delete
                    # obs_to_array and build_cost_grid, which is exactly the
                    # artifact that made the earlier T4 drill-down wrong.
                    _, obs_arr, cost, mm, bm = _observe_both(states)
                    pad = jnp.zeros(
                        (2 * N, C - obs_arr.shape[1] - 1, P, P), jnp.float32)
                    oa = jnp.concatenate(
                        [obs_arr, cost[:, None], pad], axis=1).astype(jnp.bfloat16)
                    tp = jnp.zeros((2 * N, 2, TWIN), jnp.float32)
                    no = o
                else:
                    tie = states.armies[0, 0, 0].astype(jnp.bfloat16)
                    oa = jnp.full((2 * N, C, P, P), tie, jnp.bfloat16)
                    mm = jnp.zeros((2 * N, P, P, 4), jnp.bool_)
                    bm = jnp.zeros((2 * N, P, P), jnp.bool_)
                    tp = jnp.zeros((2 * N, 2, TWIN), jnp.float32)
                    no = o
                if do_net:
                    ks = jrandom.split(key, 2 * N + 1)
                    key = ks[0]
                    net_ = eqx.combine(prm, static)
                    acts, vals, lps, _, _, _ = jax.vmap(
                        net_, in_axes=(0, 0, 0, 0, 0, None))(
                        oa, mm, bm, tp, ks[1:], None)
                else:
                    acts = jnp.tile(jnp.array([1, 0, 0, 0, 0], jnp.int32),
                                    (2 * N, 1))
                    vals = jnp.zeros((2 * N,), jnp.float32)
                    lps = jnp.zeros((2 * N,), jnp.float32)
                ea = jnp.stack([acts[:N], acts[N:]], axis=1)
                ts, ns = jax.vmap(lambda s, a: env.step(s, a, pl))(states, ea)
                d = ts.terminated | ts.truncated
                o2 = reset_done_envs(no, jnp.concatenate([d, d]))
                if do_stack:
                    data = (oa, mm, bm, tp, acts, lps, vals,
                            jnp.concatenate([ts.terminated, ts.terminated]))
                else:
                    data = jnp.float32(0)
                return (ns, key, o2), data
            (s, k, o), data = jax.lax.scan(body, (st, key, osp), None, length=T)
            return s, data
        return jax.pmap(_r)

    # (name, T, do_obs, do_net, do_stack, do_augment)
    arms = [
        (f"full_T{TLONG}",    TLONG,  True,  True,  True,  True),
        (f"noaug_T{TLONG}",   TLONG,  True,  True,  True,  False),
        (f"nostack_T{TLONG}", TLONG,  True,  True,  False, True),
        (f"envonly_T{TLONG}", TLONG,  False, False, False, True),
    ]

    results, compile_s, failed = {}, {}, {}
    fns = {}
    for name, T, do_obs, do_net, do_stack, do_aug in arms:
        log(f"compiling {name} ...")
        t0 = time.perf_counter()
        try:
            f = build(T, do_obs, do_net, do_stack, do_aug)
            out = f(p_params, states0, keys, osp0, pool_rep)
            jax.block_until_ready(out[0].armies)
            del out
            fns[name] = f
            compile_s[name] = round(time.perf_counter() - t0, 1)
            log(f"  {name} compiled+ran {compile_s[name]}s")
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {str(e)[:200]}"
            log(f"  {name} FAILED {failed[name]}")

    times = {a: [] for a in fns}
    for rnd in range(ROUNDS):
        for name, f in fns.items():
            t0 = time.perf_counter()
            for _ in range(REPS):
                out = f(p_params, states0, keys, osp0, pool_rep)
                jax.block_until_ready(out[0].armies)
            dt = (time.perf_counter() - t0) / REPS
            del out
            times[name].append(round(dt, 4))
            log(f"round {rnd} {name:<14} {dt:.3f}s")

    best = {a: min(v) for a, v in times.items()}
    log("=" * 62)
    log("ROLLOUT BISECTION (best of rounds)")
    for a, v in best.items():
        log(f"  {a:<14} {v:8.3f}s")
    def have(*ks):
        return all(k in best for k in ks)

    fT, nT, aT, eT = (f"full_T{TLONG}", f"nostack_T{TLONG}",
                      f"noaug_T{TLONG}", f"envonly_T{TLONG}")
    if have(fT, nT):
        st = best[fT] - best[nT]
        log(f"  output stacking      : {st:+.3f}s "
            f"({100 * st / best[fT]:+.1f}% of the rollout)")
    if have(fT, aT):
        au = best[fT] - best[aT]
        log(f"  augment_obs (ours)   : {au:.3f}s "
            f"({100 * au / best[fT]:.1f}% of the rollout)")
    if have(aT, eT):
        ob = best[aT] - best[eT]
        log(f"  observations+masks   : {ob:.3f}s "
            f"({100 * ob / best[fT]:.1f}% of the rollout)")
        log("    (get_observation x2, build_cost_grid x2, "
            "compute_valid_move_mask, obs_to_array, compute_build_mask)")
    if eT in best:
        log(f"  env.step             : {best[eT]:.3f}s "
            f"({100 * best[eT] / best[fT]:.1f}% of the rollout)")
    log("=" * 62)

    res = {"device": str(dev), "cc": str(getattr(dev, "compute_capability", "?")),
           "num_envs": N, "obs_stack_gib": round(obs_gb, 2),
           "times_s": times, "best_s": best, "compile_s": compile_s,
           "failed": failed,
           "peak_device_gib": round((dev.memory_stats() or {}).get(
               "peak_bytes_in_use", 0) / 2**30, 2)}
    json.dump(res, open(OUT, "w"), indent=2)
    print("BISECT_JSON_BEGIN", flush=True)
    print(json.dumps(res, indent=2), flush=True)
    print("BISECT_JSON_END", flush=True)


if __name__ == "__main__":
    main()
