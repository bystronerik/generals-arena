#!/usr/bin/env python3
"""Phase 2 x86 CPU latency benchmark for the joe networks (Modal, no GPU).

Measures ms/move on ONE x86 CPU core for randomly initialized S and M
HistoryTransformer nets (training/joe/networks/), float32 jax-CPU — the
deployment stack from plan section 5. Random weights are fine: latency does
not depend on what the weights are.

The timed step is the full per-move inference path the deployed bot will run
after stdio parsing: build cost from the observation, augment to 39 channels,
move + build masks, forward pass, greedy argmax, decode, host transfer.
Reported as p50/p90/p99/max over game-length step counts (competition games
truncate at 1200 turns).

The container requests AND limits cpu to 1.0 (no burst) and caps memory at
2 GB, mirroring the competition match limits. XLA/BLAS thread pools are
pinned to 1 in the image env.

Usage (never pipe through tail/head — redirect to a file, AGENTS.md):

    modal run scripts/joe_modal_cpu_bench.py > /tmp/joe_cpu_bench.log 2>&1 &
    modal app list          # then check startup a few minutes in
    modal app logs <app-id>

Results land in docs/research/measurements/joe-phase2-cpu-latency.md. The
verdict picks the single tier we train (plan section 8, item 10).

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

app = modal.App("joe-cpu-bench")

# Net sizes mirror training/joe/configs/{S,M}.yaml (Phase 0 freeze).
TIERS = {
    "S": dict(depth=4, embed_dim=352, n_head=8, ff_factor=2),
    "M": dict(depth=5, embed_dim=384, n_head=8, ff_factor=3),
}

GAME_LEN = 1200   # competition truncation
N_GAMES = 3


def _bench_impl(tier: str, label: str) -> dict:
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
        print(f"[cpu-bench:{label}] {msg}", flush=True)

    def cpu_stat():
        """cgroup v2 throttle counters; attributes tail spikes to the quota."""
        stats = {}
        try:
            with open("/sys/fs/cgroup/cpu.stat") as f:
                for line in f:
                    k, v = line.split()
                    stats[k] = int(v)
        except OSError:
            pass
        return stats

    cpu_model = ""
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break

    log(f"machine={platform.machine()} cpu={cpu_model} "
        f"jax={jax.__version__} devices={jax.devices()}")

    net = HistoryTransformer(
        grid_size=P, pad_to=P, patch_size=3, use_bf16=False,
        **TIERS[tier], key=jrandom.PRNGKey(0))
    n_params = sum(
        x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_array)))
    log(f"net {tier}: {n_params / 1e6:.2f}M params, float32")

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

    # Warmup / compile (the 10 s first-move grace absorbs this in a match)
    st = init_obs_state(P)
    t0 = time.perf_counter()
    action, value, st = step(net, jnp.asarray(random_raw(0)), st)
    np.asarray(action)
    compile_s = time.perf_counter() - t0
    log(f"compile+first call: {compile_s:.1f}s")

    # Rule out the Python collector as the source of tail spikes.
    gc.collect()
    gc.freeze()
    gc.disable()
    stat_before = cpu_stat()

    times_ms = []
    for game in range(N_GAMES):
        st = init_obs_state(P)
        for t in range(GAME_LEN):
            raw = jnp.asarray(random_raw(t))
            t0 = time.perf_counter()
            action, value, st = step(net, raw, st)
            np.asarray(action)
            float(value)
            times_ms.append((time.perf_counter() - t0) * 1e3)
        log(f"game {game + 1}/{N_GAMES} done "
            f"(running p50 {np.percentile(times_ms, 50):.1f} ms)")

    stat_after = cpu_stat()
    gc.enable()

    arr = np.array(times_ms)
    slow_idx = np.nonzero(arr > 50.0)[0]
    throttle = {
        k: stat_after.get(k, 0) - stat_before.get(k, 0)
        for k in ("nr_periods", "nr_throttled", "throttled_usec")
        if stat_after
    }
    result = {
        "tier": tier,
        "label": label,
        "slow_moves_over_50ms": int(slow_idx.size),
        "slow_move_positions": slow_idx[:40].tolist(),
        "slow_move_ms": [round(float(x), 1) for x in arr[slow_idx][:40]],
        "throttle_delta": throttle,
        "params_m": round(n_params / 1e6, 3),
        "cpu_model": cpu_model,
        "jax": jax.__version__,
        "dtype": "float32",
        "games": N_GAMES,
        "moves": len(times_ms),
        "compile_s": round(compile_s, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(arr.max()), 2),
        "mean_ms": round(float(arr.mean()), 2),
    }
    log(json.dumps(result))
    return result


@app.function(image=IMAGE, cpu=(1.0, 1.0), memory=(2048, 2048), timeout=1800)
def bench_tier(tier: str) -> dict:
    """The competition shape: hard 1-core quota, 2 GB."""
    return _bench_impl(tier, f"{tier}-strict")


@app.function(image=IMAGE, cpu=(1.0, 4.0), memory=(2048, 2048), timeout=1800)
def bench_tier_burst(tier: str) -> dict:
    """Control: same box, burst to 4 cores allowed. If the tail spikes vanish
    here, they are cgroup quota throttling, not net compute."""
    return _bench_impl(tier, f"{tier}-burst")


@app.local_entrypoint()
def main(tiers: str = "S,M", burst: bool = False):
    wanted = [t.strip().upper() for t in tiers.split(",") if t.strip()]
    calls = [(t, bench_tier.spawn(t)) for t in wanted]
    if burst:
        calls += [(f"{t}-burst", bench_tier_burst.spawn(t)) for t in wanted]
    out = {}
    for t, call in calls:
        out[t] = call.get()
        print(f"===== {t} done =====", flush=True)
    print("RESULTS_JSON_BEGIN", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print("RESULTS_JSON_END", flush=True)
