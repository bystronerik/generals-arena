#!/usr/bin/env python3
"""Model-size ceiling for the joe nets under the 150 ms move budget (Modal x86 CPU).

Extends the Phase 2 benchmark (scripts/joe_modal_cpu_bench.py) upward: how big
can a HistoryTransformer get before p99 ms/move crosses 150 ms on one x86 core,
float32 jax-CPU — the deployment stack from the competition plan section 5?

The size ladder is derived from the released AverageJoe configs
(average-joe/configs/{S,M,L}.yaml):

    S: depth 4,  embed 352, ff 2
    M: depth 5,  embed 384, ff 3
    L: depth 11, embed 448, ff 3      (M -> L: depth x2.2, embed x7/6)

XL applies the M -> L multiplier to L: depth 24 (11 x 2.2 = 24.2), embed 528
(448 x 7/6 = 522.7, rounded up to the next multiple of n_head = 8). L16 and
XL20 are half-steps between L and XL that bracket the expected 150 ms
crossing. M anchors the run to the Phase 2 measurement, and M7F4 is the
currently trained net (training/joe/configs/M7F4.yaml).

Every tier runs interleaved (game-level round robin) inside ONE container:
Modal per-core speed varies ~2.5x across hosts, so only same-host contrasts
count. Two containers run in parallel: the strict 1-core hard quota mirrors
the competition limits (its p99 carries the known ~500 ms CFS throttle
artifact — read its p90 as the steady state), and the burst-4 control gives
the true-compute p99 (Phase 2, section 2).

Usage (never pipe through tail/head — redirect to a file, AGENTS.md):

    modal run scripts/joe_modal_size_ceiling.py > /tmp/joe_size_ceiling.log 2>&1 &
    modal app list          # then check startup a few minutes in
    modal app logs <app-id>

All jax/equinox code lives inside the remote function: Modal re-imports this
file in the container, and the local process must not need the deps.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "jax==0.11.0",          # CPU jax — deployment stack, plan section 5
        "equinox",
        "jaxtyping",
    )
    .add_local_dir(
        str(REPO / "competition-module"),
        remote_path="/root/competition-module",
        copy=True,
    )
    .run_commands("pip install -e /root/competition-module --no-deps")
    .add_local_dir(
        str(REPO / "training" / "joe"),
        remote_path="/root/training/joe",
        copy=True,
    )
    .env({
        "PYTHONPATH": "/root",
        # Pin every thread pool to one worker BEFORE jax imports.
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
)

app = modal.App("joe-size-ceiling")

# Ordered smallest to largest. Cost model: depth * embed^2 * (4 + 2 * ff),
# normalized to M = 1.0 in the "scale" comment.
TIERS = {
    "M":    dict(depth=5,  embed_dim=384, n_head=8, ff_factor=3),  # 1.0x, Phase 2 anchor
    "M7F4": dict(depth=7,  embed_dim=384, n_head=8, ff_factor=4),  # 1.68x, current run
    "L":    dict(depth=11, embed_dim=448, n_head=8, ff_factor=3),  # 3.0x, released L
    "L16":  dict(depth=16, embed_dim=480, n_head=8, ff_factor=3),  # 5.0x, half-step
    "XL20": dict(depth=20, embed_dim=512, n_head=8, ff_factor=3),  # 7.1x, half-step
    "XL24": dict(depth=24, embed_dim=528, n_head=8, ff_factor=3),  # 9.1x, L x (M->L)
}

GAME_LEN = 1200   # competition truncation
N_GAMES = 3


def _bench_ladder_impl(label: str) -> dict:
    import gc
    import platform
    import time

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpy as np

    from generals.core.action import compute_valid_move_mask
    from training.joe.networks import (
        HistoryTransformer,
        augment_obs,
        decode_action,
        init_obs_state,
    )
    from training.joe.networks.common import (
        _BUILD_BASE_COST,
        _BUILD_PROXIMITY_DECAY,
        _BUILD_PROXIMITY_PENALTY,
        _BUILD_RADIUS,
    )

    assert platform.machine() == "x86_64", platform.machine()
    P = 21

    def log(msg):
        print(f"[size-ceiling:{label}] {msg}", flush=True)

    def rss_mb():
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS"):
                        return int(line.split()[1]) // 1024
        except OSError:
            pass
        return -1

    cpu_model = ""
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    log(f"machine={platform.machine()} cpu={cpu_model} "
        f"jax={jax.__version__} devices={jax.devices()}")

    def build_cost_from_channels(raw):
        """build_cost_from_obs on the 14-channel array (channels 1, 2, 5)."""
        structures = (((raw[2] > 0) | (raw[1] > 0)) & (raw[5] > 0)).astype(jnp.int32)
        padded = jnp.pad(structures, _BUILD_RADIUS)
        cost = jnp.full((P, P), _BUILD_BASE_COST, dtype=jnp.int32)
        R = _BUILD_RADIUS
        for di in range(-R, R + 1):
            for dj in range(-R, R + 1):
                s = _BUILD_PROXIMITY_PENALTY - _BUILD_PROXIMITY_DECAY * (abs(di) + abs(dj))
                if s > 0:
                    cost = cost + s * padded[R + di:R + di + P, R + dj:R + dj + P]
        return cost

    @eqx.filter_jit
    def step(net, raw, st):
        """Full per-move path: cost, augment, masks, forward, greedy decode."""
        cost = build_cost_from_channels(raw)
        aug, st = augment_obs(raw, cost, st)
        armies = raw[0]
        owned = raw[5] > 0
        move = compute_valid_move_mask(armies, owned, raw[3] > 0)
        plain = ~(raw[1] > 0) & ~(raw[2] > 0)
        build = owned & plain & (armies >= cost)
        td = jnp.stack([st.opponent_army_history, st.opponent_land_history])
        logits, value, _ = net._forward(aug, move, build, td)
        action = decode_action(jnp.argmax(logits), P)
        return action, value, st

    rng = np.random.default_rng(7)

    def random_raw(t):
        """Random but plausibly scaled 14-channel obs (values don't affect FLOPs)."""
        raw = np.zeros((14, P, P), dtype=np.float32)
        raw[0] = rng.poisson(8.0, (P, P))                    # armies
        for ch in (1, 2, 3, 4, 5, 6, 7, 8):
            raw[ch] = rng.random((P, P)) < 0.2
        raw[9:13] = rng.integers(1, 200, (4, 1, 1))
        raw[13] = t
        return raw

    # Build and warm every tier before any timing (same-host, one process).
    nets, meta = {}, {}
    for tier, hp in TIERS.items():
        net = HistoryTransformer(
            grid_size=P, pad_to=P, patch_size=3, use_bf16=False,
            **hp, key=jrandom.PRNGKey(0))
        n_params = sum(
            x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_array)))
        st = init_obs_state(P)
        t0 = time.perf_counter()
        action, value, st = step(net, jnp.asarray(random_raw(0)), st)
        np.asarray(action)
        compile_s = time.perf_counter() - t0
        nets[tier] = net
        meta[tier] = dict(params_m=round(n_params / 1e6, 3),
                          compile_s=round(compile_s, 1))
        log(f"net {tier}: {n_params / 1e6:.2f}M params, "
            f"compile+first call {compile_s:.1f}s, rss {rss_mb()} MB")

    gc.collect()
    gc.freeze()
    gc.disable()

    times_ms = {tier: [] for tier in TIERS}
    for game in range(N_GAMES):
        for tier, net in nets.items():
            st = init_obs_state(P)
            for t in range(GAME_LEN):
                raw = jnp.asarray(random_raw(t))
                t0 = time.perf_counter()
                action, value, st = step(net, raw, st)
                np.asarray(action)
                float(value)
                times_ms[tier].append((time.perf_counter() - t0) * 1e3)
            log(f"game {game + 1}/{N_GAMES} {tier}: "
                f"running p50 {np.percentile(times_ms[tier], 50):.1f} ms")

    gc.enable()

    out = {}
    for tier, ms in times_ms.items():
        arr = np.array(ms)
        hp = TIERS[tier]
        slow_idx = np.nonzero(arr > 150.0)[0]
        out[tier] = {
            "tier": tier,
            "label": label,
            **hp,
            **meta[tier],
            "cpu_model": cpu_model,
            "dtype": "float32",
            "games": N_GAMES,
            "moves": len(ms),
            "moves_over_150ms": int(slow_idx.size),
            "slow_move_ms": [round(float(x), 1) for x in arr[slow_idx][:40]],
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p90_ms": round(float(np.percentile(arr, 90)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "max_ms": round(float(arr.max()), 2),
            "mean_ms": round(float(arr.mean()), 2),
        }
        log(json.dumps(out[tier]))
    out["_rss_mb"] = rss_mb()
    return out


@app.function(image=IMAGE, cpu=(1.0, 1.0), memory=(2048, 2048), timeout=3600)
def bench_strict() -> dict:
    """The competition shape: hard 1-core quota, 2 GB."""
    return _bench_ladder_impl("strict")


@app.function(image=IMAGE, cpu=(1.0, 4.0), memory=(4096, 4096), timeout=3600)
def bench_burst() -> dict:
    """Control: burst to 4 cores allowed — true compute, no CFS throttle tail."""
    return _bench_ladder_impl("burst")


@app.local_entrypoint()
def main():
    calls = [("strict", bench_strict.spawn()), ("burst", bench_burst.spawn())]
    out = {}
    for name, call in calls:
        out[name] = call.get()
        print(f"===== {name} done =====", flush=True)
    print("RESULTS_JSON_BEGIN", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print("RESULTS_JSON_END", flush=True)
