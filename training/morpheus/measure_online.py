"""Coupled online-runtime qualification for Morpheus Part 09.

Measures complete turns (tensor, belief, root, search, backup, reply) on one
CPU core, selects one joint deployment configuration, and writes
``scripts/configs/morpheus/online-runtime.json`` plus ``bots/morpheus/deployment.json``.
"""
from __future__ import annotations

import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_BOT = _REPO / "bots" / "morpheus"
for entry in (_REPO, _REPO / "bots", _BOT):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)

from deployment import (  # noqa: E402
    DeploymentConfig,
    P99_ESTIMATOR_TYPE,
    P99_WARMUP_RULE,
    save_deployment,
)
from evaluator import NetworkEvaluator  # noqa: E402
from inference import load_default_session  # noqa: E402
from observe import emit_observation  # noqa: E402
from runtime import (  # noqa: E402
    COST_COMPONENTS,
    RuntimeConfig,
    RuntimeController,
    nearest_rank_p99,
)
from state import create_initial_state  # noqa: E402
from transition import PASS_ACTION, transition  # noqa: E402

DEFAULT_SWEEP = _REPO / "scripts" / "configs" / "morpheus" / "online-sweep.json"
DEFAULT_RUNTIME_OUT = _REPO / "scripts" / "configs" / "morpheus" / "online-runtime.json"
DEFAULT_BOT_DEPLOYMENT = _BOT / "deployment.json"
MEMORY_CAP_BYTES = 2 * 1024**3
JUDGE_FIRST_REPLY_S = 10.0
JUDGE_NORMAL_REPLY_S = 0.150


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(usage)
    return int(usage) * 1024


def _cpu_identity() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    brand = ""
    try:
        import subprocess

        if platform.system() == "Darwin":
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        elif Path("/proc/cpuinfo").is_file():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    brand = line.split(":", 1)[1].strip()
                    break
    except Exception as exc:  # noqa: BLE001 — record and continue
        info["cpu_brand_error"] = str(exc)
    info["cpu_brand"] = brand
    return info


def pin_single_core(core: int = 0) -> dict[str, Any]:
    """Pin process and Torch to one core when the OS allows it."""
    import torch

    torch.set_num_threads(1)
    state: dict[str, Any] = {
        "requested_core": int(core),
        "torch_num_threads": 1,
        "affinity": None,
        "affinity_supported": hasattr(os, "sched_setaffinity"),
    }
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {int(core)})
            state["affinity"] = sorted(os.sched_getaffinity(0))
        except OSError as exc:
            state["affinity_error"] = str(exc)
    else:
        state["scheduler_note"] = (
            "sched_setaffinity unavailable; Torch threads set to 1 only"
        )
    return state


def load_sweep(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("online-sweep root must be an object")
    return data


def _part04_widths(sweep: dict[str, Any]) -> list[int]:
    widths = [int(x) for x in sweep.get("part04_widths", [64])]
    if not widths:
        raise ValueError("part04_widths must come from Part 04 capacity results")
    return widths


def _available_runtimes(sweep: dict[str, Any], artifact_runtime: str) -> list[str]:
    """Keep only sandbox candidates that match the loaded artifact runtime."""
    candidates = list(sweep.get("part00b_runtimes", []))
    matched = []
    for name in candidates:
        # Artifact encodes engine in quantization; match by substring.
        if "qnnpack" in artifact_runtime and "qnnpack" in str(name):
            matched.append(str(name))
        elif "fbgemm" in artifact_runtime and "fbgemm" in str(name):
            matched.append(str(name))
        elif "x86" in artifact_runtime and "x86" in str(name) and "fbgemm" not in str(name):
            matched.append(str(name))
    if not matched:
        matched = [artifact_runtime]
    return matched


def _grid_positions(side: int) -> np.ndarray:
    grid = np.zeros((side, side), dtype=np.int32)
    grid[0, 0] = 1
    grid[side - 1, side - 1] = 2
    # A few mountains so legal masks are non-trivial.
    if side >= 18:
        grid[side // 2, side // 2] = -2
        grid[3, side - 4] = -2
    return grid


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    if q >= 0.99:
        return nearest_rank_p99(samples)
    ordered = sorted(float(x) for x in samples)
    rank = max(1, int(math.ceil(q * len(ordered))))
    return ordered[rank - 1]


def _make_controller(
    *,
    H: int,
    W: int,
    deployment: DeploymentConfig,
    session,
    seed: int,
) -> RuntimeController:
    evaluator = NetworkEvaluator(session)
    cfg = deployment.to_runtime_config()
    return RuntimeController(
        seat=0,
        H=H,
        W=W,
        evaluator=evaluator,
        config=cfg,
        rng=np.random.default_rng(seed),
        proposal_policy=evaluator.policy_logits,
    )


def _run_scenario(
    *,
    session,
    deployment: DeploymentConfig,
    side: int,
    turns: int,
    seed: int,
) -> dict[str, Any]:
    state = create_initial_state(_grid_positions(side))
    ctl = _make_controller(
        H=side, W=side, deployment=deployment, session=session, seed=seed
    )
    first_ms: list[float] = []
    normal_ms: list[float] = []
    normal_sims: list[int] = []
    components: dict[str, list[float]] = {name: [] for name in COST_COMPONENTS}
    belief_root_ok = 0
    belief_root_n = 0
    recovery_n = 0
    collapsed_n = 0
    deadline_faults = 0
    min_sims_misses = 0
    normal_idx = 0
    turn = -1

    for turn in range(turns):
        obs = emit_observation(state, 0)
        t0 = time.perf_counter()
        action = ctl.decide(obs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        first = turn == 0
        limit = (
            deployment.first_move_limit_ms if first else deployment.normal_deadline_ms
        )
        if elapsed_ms > limit:
            deadline_faults += 1
        if first:
            first_ms.append(elapsed_ms)
        else:
            normal_ms.append(elapsed_ms)
            normal_sims.append(int(ctl.completed_simulations))
            belief_root_n += 1
            belief_root_ok += int(ctl.belief_plus_root_ok)
            recovery_n += int(ctl.recovery)
            if ctl.belief is not None and (ctl.belief.collapsed or ctl.belief.n == 0):
                collapsed_n += 1
            normal_idx += 1
            # Warm-up for the min-sims gate is a few normal turns, not the
            # full p99 window (window can exceed the sample length).
            sims_warmup = int(min(2, max(0, turns - 2)))
            if (
                normal_idx > sims_warmup
                and state.winner < 0
                and int(ctl.completed_simulations) < int(deployment.min_simulations)
            ):
                min_sims_misses += 1

        # Advance with pass vs pass so the game stays nonterminal for a while.
        actions = np.zeros((2, 5), dtype=np.int32)
        actions[0] = np.asarray(action, dtype=np.int32)
        actions[1] = PASS_ACTION
        state, _info = transition(state, actions)
        if state.winner >= 0:
            break

    # Per-call component samples from the rolling estimators (not turn sums).
    for name, est in ctl._estimators.items():
        components[name] = [float(x) for x in est._samples]

    return {
        "side": side,
        "turns_played": turn + 1,
        "first_move_ms": first_ms,
        "normal_move_ms": normal_ms,
        "normal_sims": normal_sims,
        "components": components,
        "belief_plus_root_ok_rate": (
            belief_root_ok / belief_root_n if belief_root_n else 0.0
        ),
        "recovery_rate": recovery_n / belief_root_n if belief_root_n else 0.0,
        "collapsed_rate": collapsed_n / belief_root_n if belief_root_n else 0.0,
        "deadline_faults": deadline_faults,
        "min_sims_misses": min_sims_misses,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _summarize_scenario(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = [ms for row in rows for ms in row["first_move_ms"]]
    normal = [ms for row in rows for ms in row["normal_move_ms"]]
    sims = [s for row in rows for s in row["normal_sims"]]
    components: dict[str, list[float]] = {name: [] for name in COST_COMPONENTS}
    for row in rows:
        for name, values in row["components"].items():
            components.setdefault(name, []).extend(values)
    faults = sum(int(row["deadline_faults"]) for row in rows)
    min_miss = sum(int(row["min_sims_misses"]) for row in rows)
    br_rates = [float(row["belief_plus_root_ok_rate"]) for row in rows]
    recovery = [float(row["recovery_rate"]) for row in rows]
    collapsed = [float(row["collapsed_rate"]) for row in rows]
    peak = max(int(row["peak_rss_bytes"]) for row in rows)
    offline = {
        name: (
            nearest_rank_p99(values)
            if values
            else 0.0
        )
        for name, values in components.items()
    }
    # Keep every component finite for strict JSON and admission calibration.
    offline = {
        name: (0.0 if value != value else float(value))
        for name, value in offline.items()
    }
    return {
        "first_p50_ms": _percentile(first, 0.50),
        "first_p99_ms": _percentile(first, 0.99),
        "normal_p50_ms": _percentile(normal, 0.50),
        "normal_p99_ms": _percentile(normal, 0.99),
        "normal_samples": len(normal),
        "mean_completed_simulations": float(np.mean(sims)) if sims else 0.0,
        "min_completed_simulations": int(min(sims)) if sims else 0,
        "deadline_faults": faults,
        "min_sims_misses": min_miss,
        "belief_plus_root_ok_rate": float(np.mean(br_rates)) if br_rates else 0.0,
        "recovery_rate": float(np.mean(recovery)) if recovery else 0.0,
        "collapsed_rate": float(np.mean(collapsed)) if collapsed else 0.0,
        "peak_rss_bytes": peak,
        "offline_p99_ms": offline,
        "hides_belief_collapse": bool(
            float(np.mean(collapsed) if collapsed else 0.0) > 0.05
            and _percentile(normal, 0.50) < 40.0
        ),
    }


def _calibrate_offline_p99(
    *,
    session,
    deployment: DeploymentConfig,
    side: int = 18,
    seed: int = 0,
) -> dict[str, float]:
    """Seed offline p99 from a short untimed rehearsal so admission is honest."""
    dep = replace(
        deployment,
        # Generous deadline so rehearsal completes every component once.
        normal_deadline_ms=10_000.0,
        first_move_limit_ms=10_000.0,
        admission_guard_ms=0.0,
        offline_p99_ms={name: 0.0 for name in COST_COMPONENTS},
        target_simulations=max(4, int(deployment.min_simulations)),
        p99_window=max(8, int(deployment.p99_window)),
    )
    # Cold-start rehearsal: discard estimator samples after a short warm pass.
    warm = _run_scenario(
        session=session,
        deployment=dep,
        side=side,
        turns=4,
        seed=seed,
    )
    del warm
    row = _run_scenario(
        session=session,
        deployment=dep,
        side=side,
        turns=max(8, dep.min_simulations + 4),
        seed=seed + 1,
    )
    offline: dict[str, float] = {}
    for name in COST_COMPONENTS:
        samples = row["components"].get(name, [])
        if name == "belief_proposal" and len(samples) >= 6:
            # Early turns keep many unique enemy infos; that spike must not
            # lock admission forever. Seed offline p99 from the warmer half.
            samples = list(samples[len(samples) // 2 :])
        if len(samples) >= 3:
            # Drop the coldest sample, then take nearest-rank p99 of the rest.
            warm_s = sorted(float(x) for x in samples)[1:]
            offline[name] = float(nearest_rank_p99(warm_s))
        elif samples:
            offline[name] = float(max(samples))
        else:
            offline[name] = float(deployment.offline_p99_ms.get(name, 1.0))
        if offline[name] != offline[name]:  # NaN guard
            offline[name] = float(deployment.offline_p99_ms.get(name, 1.0))
        offline[name] = max(0.0, float(offline[name]))
    return offline


def _candidate_from_sweep(
    sweep: dict[str, Any],
    *,
    width: int,
    runtime_name: str,
    n_particles: int,
    target_simulations: int,
    pending_leaf_batch: int,
    max_proposal_batch: int,
    artifact_manifest: dict[str, Any],
) -> DeploymentConfig:
    quant = artifact_manifest.get("quantization", {})
    return DeploymentConfig(
        inference_runtime=str(
            artifact_manifest.get("runtime", runtime_name)
        ),
        quantization_format=str(quant.get("format", "")),
        quantization_engine=str(quant.get("engine", "qnnpack")),
        trunk_channels=int(width),
        network_width=int(width),
        n_particles=int(n_particles),
        target_simulations=int(target_simulations),
        min_simulations=int(sweep.get("min_simulations", 8)),
        pending_leaf_batch=int(pending_leaf_batch),
        max_proposal_batch=int(max_proposal_batch),
        max_forward_equivalents=int(sweep.get("max_forward_equivalents", 113)),
        max_tree_nodes=int(sweep.get("max_tree_nodes", 4096)),
        search_depth=int(sweep.get("search_depth_default", 8)),
        normal_deadline_ms=float(sweep.get("normal_deadline_ms", 125.0)),
        reserve_ms=float(sweep.get("reserve_ms", 25.0)),
        first_move_limit_ms=float(sweep.get("first_move_limit_ms", 8500.0)),
        admission_guard_ms=float(sweep.get("admission_guard_ms_default", 10.0)),
        resident_memory_target_mb=float(
            sweep.get("resident_memory_target_mb_default", 256.0)
        ),
        p99_estimator_type=P99_ESTIMATOR_TYPE,
        p99_warmup_rule=P99_WARMUP_RULE,
        p99_window=int(sweep.get("p99_window_default", 64)),
        offline_p99_ms={name: 1.0 for name in COST_COMPONENTS},
        warmup_batch_shapes=tuple(
            int(x) for x in sweep.get("warmup_batch_shapes", [1, 4, 64])
        ),
    )


def _passes_gates(
    summary: dict[str, Any],
    deployment: DeploymentConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary["deadline_faults"] != 0:
        reasons.append("deadline_faults")
    if summary["first_p99_ms"] > deployment.first_move_limit_ms:
        reasons.append("first_internal_deadline")
    if summary["first_p99_ms"] > JUDGE_FIRST_REPLY_S * 1000.0:
        reasons.append("first_judge_deadline")
    if summary["normal_p99_ms"] > deployment.normal_deadline_ms:
        reasons.append("normal_internal_deadline")
    if summary["peak_rss_bytes"] >= MEMORY_CAP_BYTES:
        reasons.append("memory_cap")
    if summary["belief_plus_root_ok_rate"] < 1.0:
        reasons.append("belief_plus_root_incomplete")
    if summary["min_sims_misses"] != 0:
        reasons.append("min_sims_misses")
    if summary["mean_completed_simulations"] < deployment.min_simulations:
        reasons.append("mean_sims_below_minimum")
    if summary.get("hides_belief_collapse"):
        reasons.append("hides_belief_collapse")
    # Soft judge normal check recorded but not a hard local fault here —
    # Part 08 submission enforces 150 ms externally.
    return len(reasons) == 0, reasons


def _select_window_and_guard(
    *,
    component_samples: dict[str, list[float]],
    window_candidates: list[int],
    guard_candidates: list[float],
    normal_deadline_ms: float,
    normal_move_samples: list[float],
) -> tuple[int, float, dict[str, float], dict[str, Any]]:
    """Pick p99 window and admission guard from zero-fault measured boundary."""
    notes: dict[str, Any] = {"rejected_windows": [], "guard_trials": []}
    # Reject windows that are too small to remember a measured slow path:
    # require the window p99 to be at least the global p99 of the same samples
    # when enough samples exist.
    selected_window = window_candidates[-1]
    for window in window_candidates:
        ok = True
        for name, samples in component_samples.items():
            if len(samples) < max(window, 2):
                continue
            global_p99 = nearest_rank_p99(samples)
            # Simulate rolling: if the slow sample is only at the start and
            # window forgets it, reject.
            rolled = samples[-window:]
            rolled_p99 = nearest_rank_p99(rolled)
            if global_p99 > 0 and rolled_p99 < 0.5 * global_p99 and global_p99 > 5.0:
                ok = False
                notes["rejected_windows"].append(
                    {
                        "window": window,
                        "component": name,
                        "reason": "forgets_measured_slow_path",
                        "global_p99": global_p99,
                        "rolled_p99": rolled_p99,
                    }
                )
                break
        if ok:
            selected_window = window
            break

    offline = {
        name: (nearest_rank_p99(vals) if vals else 1.0)
        for name, vals in component_samples.items()
    }
    # Admission guard: smallest guard such that measured normal p99 + guard
    # stays inside the internal deadline under a scheduler slack assumption.
    selected_guard = guard_candidates[-1]
    normal_p99 = nearest_rank_p99(normal_move_samples) if normal_move_samples else 0.0
    for guard in sorted(guard_candidates):
        # Require room for the guard after the measured complete-turn p99.
        fits = normal_p99 + guard <= normal_deadline_ms
        notes["guard_trials"].append(
            {
                "guard_ms": guard,
                "normal_p99_ms": normal_p99,
                "fits": fits,
            }
        )
        if fits:
            selected_guard = guard
            break
    return selected_window, float(selected_guard), offline, notes


def measure_online_runtime(
    sweep_path: Path,
    *,
    single_core: bool = True,
    core: int = 0,
    write_runtime: bool = True,
    runtime_out: Optional[Path] = None,
    bot_deployment_out: Optional[Path] = None,
) -> dict[str, Any]:
    import hashlib
    import subprocess

    sweep_path = Path(sweep_path)
    sweep = load_sweep(sweep_path)
    sweep_bytes = sweep_path.read_bytes()
    sweep_digest = hashlib.sha256(sweep_bytes).hexdigest()
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO,
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        commit_hash = ""
    cpu = _cpu_identity()
    scheduler = pin_single_core(core) if single_core else {"single_core": False}
    session = load_default_session()
    # Warm proposal / leaf batch shapes so cold FX quantize is not the offline p99.
    try:
        import torch
        from network import BOARD, IN_CHANNELS

        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        for batch in (1, 4, 8, 16, 32, 64):
            x = torch.randn(int(batch), IN_CHANNELS, BOARD, BOARD, generator=gen)
            session.forward_policy(x)
            session.forward_policy_wdl(x)
    except Exception as exc:  # noqa: BLE001
        scheduler = dict(scheduler)
        scheduler["warmup_error"] = str(exc)
    manifest = session.manifest
    artifact_runtime = str(manifest.get("runtime", ""))
    widths = _part04_widths(sweep)
    runtimes = _available_runtimes(sweep, artifact_runtime)
    particles = [int(x) for x in sweep.get("particle_counts", [32, 64, 128])]
    targets = [int(x) for x in sweep.get("target_simulations", [8, 16, 32])]
    leaf_batches = [int(x) for x in sweep.get("pending_leaf_batches", [4])]
    proposal_batches = [int(x) for x in sweep.get("max_proposal_batches", [16, 32, 64])]
    board_sizes = [int(x) for x in sweep.get("board_sizes", [18, 21])]
    turns = int(sweep.get("turns_per_scenario", 12))
    seeds = [int(x) for x in sweep.get("seeds", [0, 1])]
    deadlines = [float(x) for x in sweep.get("normal_deadline_ms_candidates", [125.0])]
    depths = [int(x) for x in sweep.get("search_depths", [sweep.get("search_depth_default", 8)])]

    trials: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []
    calib_cache: dict[tuple[int, int, int], dict[str, float]] = {}

    for width in widths:
        arch = manifest.get("architecture", {})
        if int(arch.get("trunk_channels", width)) != int(width):
            trials.append(
                {
                    "skip": True,
                    "reason": "width_not_in_loaded_artifact",
                    "width": width,
                }
            )
            continue
        for runtime_name in runtimes:
            for n_particles in particles:
                for target in targets:
                    for leaf_batch in leaf_batches:
                        for prop_batch in proposal_batches:
                            for depth in depths:
                                for deadline_ms in deadlines:
                                    dep = _candidate_from_sweep(
                                        sweep,
                                        width=width,
                                        runtime_name=runtime_name,
                                        n_particles=n_particles,
                                        target_simulations=target,
                                        pending_leaf_batch=leaf_batch,
                                        max_proposal_batch=prop_batch,
                                        artifact_manifest=manifest,
                                    )
                                    dep.search_depth = int(depth)
                                    dep.normal_deadline_ms = float(deadline_ms)
                                    dep.reserve_ms = max(0.0, 150.0 - float(deadline_ms))
                                    dep.qualification_host = platform.platform()
                                    dep.qualification_cpu = str(cpu.get("cpu_brand", ""))
                                    cache_key = (
                                        int(n_particles),
                                        int(prop_batch),
                                        int(depth),
                                    )
                                    if cache_key not in calib_cache:
                                        calib_cache[cache_key] = _calibrate_offline_p99(
                                            session=session,
                                            deployment=dep,
                                            side=board_sizes[0],
                                            seed=0,
                                        )
                                    dep.offline_p99_ms = dict(calib_cache[cache_key])
                                    belief_root = (
                                        float(dep.offline_p99_ms.get("belief_proposal", 0.0))
                                        + float(
                                            dep.offline_p99_ms.get(
                                                "particle_transitions", 0.0
                                            )
                                        )
                                        + float(dep.offline_p99_ms.get("root_inference", 0.0))
                                        + float(dep.admission_guard_ms)
                                    )
                                    cfg_key = {
                                        "width": width,
                                        "runtime": runtime_name,
                                        "n_particles": n_particles,
                                        "target_simulations": target,
                                        "pending_leaf_batch": leaf_batch,
                                        "max_proposal_batch": prop_batch,
                                        "search_depth": depth,
                                        "normal_deadline_ms": deadline_ms,
                                    }
                                    # Skip only configs whose calibrated belief+root
                                    # is far past the deadline (2x). Early-turn
                                    # uniqueness spikes inflate p99; those configs
                                    # still need a full scenario measurement.
                                    if belief_root > 2.0 * dep.normal_deadline_ms:
                                        trials.append(
                                            {
                                                "config": cfg_key,
                                                "summary": {
                                                    "belief_plus_root_forecast_ms": belief_root,
                                                    "deadline_faults": 1,
                                                    "belief_plus_root_ok_rate": 0.0,
                                                    "min_sims_misses": 1,
                                                    "normal_p50_ms": belief_root,
                                                    "normal_p99_ms": belief_root,
                                                    "first_p50_ms": 0.0,
                                                    "first_p99_ms": 0.0,
                                                    "peak_rss_bytes": _peak_rss_bytes(),
                                                    "offline_p99_ms": dict(dep.offline_p99_ms),
                                                    "hides_belief_collapse": False,
                                                    "recovery_rate": 0.0,
                                                    "collapsed_rate": 0.0,
                                                    "mean_completed_simulations": 0.0,
                                                    "min_completed_simulations": 0,
                                                    "normal_samples": 0,
                                                },
                                                "accepted": False,
                                                "reject_reasons": [
                                                    "belief_plus_root_cannot_fit"
                                                ],
                                            }
                                        )
                                        continue
                                    rows = [
                                        _run_scenario(
                                            session=session,
                                            deployment=dep,
                                            side=side,
                                            turns=turns,
                                            seed=seed + side,
                                        )
                                        for side in board_sizes
                                        for seed in seeds
                                    ]
                                    summary = _summarize_scenario(rows)
                                    summary["belief_plus_root_forecast_ms"] = belief_root
                                    ok, reasons = _passes_gates(summary, dep)
                                    trial = {
                                        "config": cfg_key,
                                        "summary": summary,
                                        "accepted": ok,
                                        "reject_reasons": reasons,
                                    }
                                    trials.append(trial)
                                    if ok:
                                        survivors.append(
                                            {
                                                "deployment": dep,
                                                "summary": summary,
                                                "trial": trial,
                                            }
                                        )

    # Prefer highest target simulations, then particles, then proposal batch.
    survivors.sort(
        key=lambda s: (
            s["deployment"].target_simulations,
            s["deployment"].n_particles,
            s["deployment"].max_proposal_batch,
            -s["summary"]["normal_p99_ms"],
        ),
        reverse=True,
    )

    selected: Optional[DeploymentConfig] = None
    estimator_notes: dict[str, Any] = {}
    best_effort: Optional[dict[str, Any]] = None
    # Best effort among trials that completed belief+root, even if min-sims fails.
    effort_trials = [
        t
        for t in trials
        if t.get("summary", {}).get("belief_plus_root_ok_rate", 0) >= 1.0
        and t.get("summary", {}).get("normal_samples", 0) > 0
    ]
    if effort_trials:
        effort_trials.sort(
            key=lambda t: (
                float(t["summary"].get("mean_completed_simulations", 0.0)),
                -float(t["summary"].get("normal_p99_ms", 1e9)),
            ),
            reverse=True,
        )
        best_effort = effort_trials[0]

    if survivors:
        best = survivors[0]
        selected = best["deployment"]
        # Collect component samples from a fresh run for estimator selection.
        rows = [
            _run_scenario(
                session=session,
                deployment=selected,
                side=side,
                turns=max(turns, int(selected.p99_window) + 4),
                seed=100 + side,
            )
            for side in board_sizes
        ]
        summary = _summarize_scenario(rows)
        component_samples = {
            name: [ms for row in rows for ms in row["components"].get(name, [])]
            for name in COST_COMPONENTS
        }
        normal_moves = [ms for row in rows for ms in row["normal_move_ms"]]
        window, guard, offline, estimator_notes = _select_window_and_guard(
            component_samples=component_samples,
            window_candidates=[int(x) for x in sweep.get("p99_windows", [32, 64, 128])],
            guard_candidates=[
                float(x) for x in sweep.get("admission_guard_ms_candidates", [5, 10, 15, 20])
            ],
            normal_deadline_ms=selected.normal_deadline_ms,
            normal_move_samples=normal_moves,
        )
        peak = int(summary["peak_rss_bytes"])
        # Resident target: peak RSS in MB plus margin below 2 GB.
        peak_mb = peak / (1024 * 1024)
        resident_target = min(
            max(peak_mb * 1.25, peak_mb + 32.0),
            (MEMORY_CAP_BYTES / (1024 * 1024)) * 0.5,
        )
        selected = replace(
            selected,
            p99_window=int(window),
            admission_guard_ms=float(guard),
            offline_p99_ms={k: float(v) for k, v in offline.items()},
            resident_memory_target_mb=float(resident_target),
            belief_quality_threshold=None,
            qualification_host=platform.platform(),
            qualification_cpu=str(cpu.get("cpu_brand", "")),
        )

    verdict = "yes" if selected is not None else "no"
    reasons_no: list[str] = []
    if selected is None:
        if any(
            t.get("summary", {}).get("belief_plus_root_ok_rate", 1) < 1
            for t in trials
            if "summary" in t
        ):
            reasons_no.append("belief_plus_root_cannot_fit")
        if any(
            t.get("summary", {}).get("mean_completed_simulations", 0)
            < float(sweep.get("min_simulations", 8))
            for t in trials
            if t.get("summary", {}).get("normal_samples", 0) > 0
        ):
            reasons_no.append("min_search_target_missed")
        if not any(t.get("accepted") for t in trials):
            reasons_no.append("no_zero_fault_configuration")

    runtime_path = Path(runtime_out) if runtime_out else DEFAULT_RUNTIME_OUT
    bot_path = Path(bot_deployment_out) if bot_deployment_out else DEFAULT_BOT_DEPLOYMENT

    written: Optional[DeploymentConfig] = selected
    if written is None and best_effort is not None and write_runtime:
        # Rebuild a playable deployment from the best measured trial config.
        cfg = best_effort["config"]
        written = _candidate_from_sweep(
            sweep,
            width=int(cfg["width"]),
            runtime_name=str(cfg["runtime"]),
            n_particles=int(cfg["n_particles"]),
            target_simulations=int(cfg["target_simulations"]),
            pending_leaf_batch=int(cfg["pending_leaf_batch"]),
            max_proposal_batch=int(cfg["max_proposal_batch"]),
            artifact_manifest=manifest,
        )
        written.search_depth = int(cfg.get("search_depth", written.search_depth))
        written.normal_deadline_ms = float(
            cfg.get("normal_deadline_ms", written.normal_deadline_ms)
        )
        written.reserve_ms = max(0.0, 150.0 - written.normal_deadline_ms)
        offline = best_effort["summary"].get("offline_p99_ms", {})
        written.offline_p99_ms = {
            name: (
                float(offline[name])
                if name in offline
                and offline[name] == offline[name]  # not NaN
                else 1.0
            )
            for name in COST_COMPONENTS
        }
        peak = int(best_effort["summary"].get("peak_rss_bytes", 0))
        peak_mb = peak / (1024 * 1024) if peak else written.resident_memory_target_mb
        written.resident_memory_target_mb = float(
            min(max(peak_mb * 1.25, peak_mb + 32.0), 1024.0)
        )
        written.qualification_host = platform.platform()
        written.qualification_cpu = str(cpu.get("cpu_brand", ""))
        written.belief_limitation_note = (
            "Qualification verdict is no on this host: the measured stack does "
            "not finish 8 simulations on every warm normal move. This file is "
            "the best belief+root-complete trial for local play and further "
            "measurement, not an accepted deployment."
        )

    if written is not None and write_runtime:
        save_deployment(written, runtime_path)
        save_deployment(written, bot_path)

    report = {
        "part": "09-online-qualification",
        "verdict": verdict,
        "reasons_no": reasons_no,
        "sweep_config_path": str(sweep_path),
        "sweep_config_digest": sweep_digest,
        "commit_hash": commit_hash,
        "cpu": cpu,
        "scheduler": scheduler,
        "artifact_runtime": artifact_runtime,
        "artifact_quantization": manifest.get("quantization", {}),
        "part04_widths": widths,
        "runtimes_swept": runtimes,
        "trial_count": len(trials),
        "survivor_count": len(survivors),
        "trials": trials,
        "estimator_notes": estimator_notes,
        "selected": selected.to_dict() if selected is not None else None,
        "best_effort": best_effort,
        "written_deployment": written.to_dict() if written is not None else None,
        "runtime_config_path": str(runtime_path) if written is not None else None,
        "bot_deployment_path": str(bot_path) if written is not None else None,
        "judge_limits": {
            "first_reply_s": JUDGE_FIRST_REPLY_S,
            "normal_reply_s": JUDGE_NORMAL_REPLY_S,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
        },
        "specification_gaps": {
            "judge_cpu_model": "unpublished; local latency conditional on recorded host",
            "belief_quality_threshold": None,
            "acceptable_quantization_error": "not defined; artifact MAE recorded only",
            "p99_estimator": {
                "type": P99_ESTIMATOR_TYPE,
                "warmup_rule": P99_WARMUP_RULE,
                "selected_window": (
                    written.p99_window if written is not None else None
                ),
                "admission_guard_ms": (
                    written.admission_guard_ms if written is not None else None
                ),
            },
        },
        "belief_limitation_note": (
            written.belief_limitation_note
            if written is not None
            else DeploymentConfig().belief_limitation_note
        ),
    }
    return report
