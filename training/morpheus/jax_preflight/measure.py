"""Run Modal/local JAX preflight measurements on the current device."""
from __future__ import annotations

import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from training.morpheus.jax_preflight.fixtures import all_parity_fixtures
from training.morpheus.jax_preflight.parity import (
    snapshot_to_jsonable,
    state_info_snapshot,
)
from training.morpheus.jax_preflight.transition import (
    DEFAULT_NUM_ENVS,
    DEFAULT_POOL_SIZE,
    DEFAULT_SCAN_STEPS,
    compile_scan_transition,
    make_action_sequence,
    make_competition_env,
    make_pool_and_states,
    step_one,
)


def _device_description() -> dict[str, Any]:
    devices = jax.devices()
    device = devices[0]
    kind = device.device_kind if hasattr(device, "device_kind") else str(device)
    platform = jax.default_backend()
    info: dict[str, Any] = {
        "platform": platform,
        "device_str": str(device),
        "device_kind": kind,
        "device_count": len(devices),
    }
    mem = getattr(device, "memory_stats", None)
    if callable(mem):
        try:
            stats = mem()
            info["memory_stats"] = {
                k: int(v) for k, v in stats.items() if isinstance(v, (int, float))
            }
        except Exception as exc:  # noqa: BLE001 — probe only
            info["memory_stats_error"] = str(exc)
    return info


def _package_versions() -> dict[str, Any]:
    import jaxlib

    versions: dict[str, Any] = {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
    }
    try:
        backend = jax.lib.xla_bridge.get_backend()
        versions["xla_platform"] = backend.platform
        versions["xla_platform_version"] = getattr(
            backend, "platform_version", None
        )
    except Exception as exc:  # noqa: BLE001
        versions["xla_error"] = str(exc)
    return versions


def _nvidia_versions() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        line = smi.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            out["gpu_name"] = parts[0]
            out["memory_total"] = parts[1]
            out["driver_version"] = parts[2]
            out["cuda_version"] = parts[3]
        else:
            out["nvidia_smi_raw"] = line
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError) as exc:
        out["nvidia_smi_error"] = str(exc)
    try:
        nvcc = subprocess.run(
            ["nvcc", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        out["nvcc"] = nvcc.stdout.strip() or nvcc.stderr.strip()
    except FileNotFoundError:
        out["nvcc"] = None
    return out


def _measure_transfer(states) -> dict[str, float]:
    """Host↔device round-trip time for one batched GameState tree."""
    leaves = jax.tree.leaves(states)
    host = jax.device_get(states)
    # force a known host tree before timing H2D
    _ = jax.tree.leaves(host)

    t0 = time.perf_counter()
    on_device = jax.device_put(host)
    jax.block_until_ready(jax.tree.leaves(on_device))
    h2d = time.perf_counter() - t0

    t0 = time.perf_counter()
    back = jax.device_get(on_device)
    _ = jax.tree.leaves(back)
    d2h = time.perf_counter() - t0

    nbytes = sum(int(np.asarray(x).nbytes) for x in leaves)
    return {
        "host_to_device_s": h2d,
        "device_to_host_s": d2h,
        "payload_bytes": nbytes,
    }


def _run_parity_fixtures(env, pool) -> list[dict[str, Any]]:
    results = []
    for fixture in all_parity_fixtures():
        last_state, info = step_one(env, pool, fixture.state, fixture.actions)
        snap = state_info_snapshot(last_state, info)
        results.append(
            {
                "name": fixture.name,
                "snapshot": snapshot_to_jsonable(snap),
                # keep raw arrays for in-process compare when both seats share memory
                "_raw": snap,
            }
        )
    return results


def run_preflight_on_device(
    *,
    role: str,
    seed: int = 0,
    num_envs: int = DEFAULT_NUM_ENVS,
    scan_steps: int = DEFAULT_SCAN_STEPS,
    pool_size: int = DEFAULT_POOL_SIZE,
    warm_reps: int = 3,
) -> dict[str, Any]:
    """Measure compile, warm throughput, memory, transfer, and fixture snapshots.

    `role` is `"cpu"` or `"gpu"` and is recorded for the merged report.
    """
    wall0 = time.perf_counter()
    device = _device_description()
    versions = _package_versions()
    nvidia = _nvidia_versions()

    env = make_competition_env(pool_size=pool_size)
    modifiers = {
        "mode": env.mode,
        "build_castles": env.build_castles,
        "deathtouch_turn": env.deathtouch_turn,
        "truncation": env.truncation,
        "pad_to": env.pad_to,
        "min_grid_size": env.min_grid_size,
        "max_grid_size": env.max_grid_size,
        "perfect_info": env.perfect_info,
        "pool_size": env.pool_size,
    }

    t_pool0 = time.perf_counter()
    pool, states = make_pool_and_states(env, seed=seed, num_envs=num_envs)
    jax.block_until_ready(jax.tree.leaves(pool))
    jax.block_until_ready(jax.tree.leaves(states))
    pool_gen_s = time.perf_counter() - t_pool0

    actions_seq = make_action_sequence(
        seed=seed + 1, num_envs=num_envs, num_steps=scan_steps
    )
    scan_fn = compile_scan_transition(env, pool)

    # Cold compile
    t0 = time.perf_counter()
    final, infos = scan_fn(states, actions_seq)
    jax.block_until_ready(jax.tree.leaves(final))
    jax.block_until_ready(jax.tree.leaves(infos))
    cold_compile_s = time.perf_counter() - t0

    # Warm throughput
    warm_times: list[float] = []
    for _ in range(warm_reps):
        t0 = time.perf_counter()
        final, infos = scan_fn(states, actions_seq)
        jax.block_until_ready(jax.tree.leaves(final))
        jax.block_until_ready(jax.tree.leaves(infos))
        warm_times.append(time.perf_counter() - t0)

    steps_total = num_envs * scan_steps
    warm_mean_s = float(np.mean(warm_times))
    warm_steps_per_s = steps_total / warm_mean_s if warm_mean_s > 0 else 0.0

    transfer = _measure_transfer(states)
    parity = _run_parity_fixtures(env, pool)

    # Drop in-process raw arrays from the serializable payload.
    parity_serializable = [
        {"name": p["name"], "snapshot": p["snapshot"]} for p in parity
    ]

    peak_bytes = None
    mem_stats = device.get("memory_stats") or {}
    if "peak_bytes_in_use" in mem_stats:
        peak_bytes = mem_stats["peak_bytes_in_use"]
    elif "bytes_in_use" in mem_stats:
        peak_bytes = mem_stats["bytes_in_use"]

    return {
        "role": role,
        "device": device,
        "versions": versions,
        "nvidia": nvidia,
        "modifiers": modifiers,
        "config": {
            "seed": seed,
            "num_envs": num_envs,
            "scan_steps": scan_steps,
            "pool_size": pool_size,
            "warm_reps": warm_reps,
        },
        "pool_generation_s": pool_gen_s,
        "cold_compile_s": cold_compile_s,
        "warm_mean_s": warm_mean_s,
        "warm_steps_per_s": warm_steps_per_s,
        "steps_per_warm_call": steps_total,
        "transfer": transfer,
        "device_memory_peak_bytes": peak_bytes,
        "parity_fixtures": parity_serializable,
        "compiled_ok": True,
        "device_wall_s": time.perf_counter() - wall0,
    }
