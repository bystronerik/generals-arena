#!/usr/bin/env python3
"""Per-phase wall-clock profile of the joe PPO loop (Modal, small GPU).

Runs the REAL ``training/joe`` training loop (``main.run`` ->
``train.ppo.train``) for a few iterations at a small config, and times every
part of it: pool generation, JIT compile, rollout, GAE, the host-sync
diagnostics, the PPO update, logging, EMA, eval, and checkpointing.

Nothing in ``training/joe`` is modified. The timer attaches to
``ppo.train``'s code object through ``sys.monitoring`` local LINE events
(PEP 669), so every source line of the loop is clocked and untraced code
pays nothing. Each line is attributed to the iteration it ran in, which
separates iteration 0 (cold JIT compile) from the steady state.

Caveat that matters when you read the numbers: JAX dispatch is async, so a
line's wall clock is the time until the *next* blocking point. The loop
blocks explicitly after the rollout and after the PPO step, and every
``float(...)`` in the diagnostics block is an implicit barrier, so the
phases below are separated by real barriers and the attribution holds.

Usage (never pipe through tail/head — redirect to a file, AGENTS.md):

    modal run scripts/joe_modal_profile.py > /tmp/joe_profile.log 2>&1 &
    modal app list          # then check startup a few minutes in
    modal app logs <app-id>

All repo imports live inside the remote functions: Modal re-imports this
file in the container, and the local process must not need GPU deps.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
if modal.is_local():
    sys.path.insert(0, str(REPO))

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "jax[cuda12]==0.11.0",
        "equinox",
        "optax",
        "pyyaml",
    )
    .add_local_dir(
        str(REPO / "competition-module"),
        remote_path="/root/competition-module",
        copy=True,
    )
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root"})
    .add_local_file(str(REPO / "training" / "__init__.py"),
                    remote_path="/root/training/__init__.py")
    .add_local_dir(str(REPO / "training" / "joe"),
                   remote_path="/root/training/joe",
                   ignore=["**/__pycache__", "tests/**"])
)

app = modal.App("joe-profile")


# ---------------------------------------------------------------------------
# Line clock (stdlib only; runs in the container)
# ---------------------------------------------------------------------------


class LineClock:
    """Per-line, per-iteration wall clock for a single Python function.

    ``boundary_lineno`` is the first statement of the training for-loop; a
    LINE event there closes one iteration and opens the next. Lines seen
    before the first boundary land in iteration -1 (the setup block).
    """

    TOOL_ID = 2  # sys.monitoring.PROFILER_ID

    def __init__(self, code, boundary_lineno: int):
        self.code = code
        self.boundary = boundary_lineno
        self.iter_idx = -1
        self.times: dict[tuple[int, int], float] = {}
        self.hits: dict[tuple[int, int], int] = {}
        self.mode = "none"
        self._last_line = None
        self._last_t = 0.0

    # -- recording ---------------------------------------------------------

    def _tick(self, lineno: int) -> None:
        now = time.perf_counter()
        if self._last_line is not None:
            key = (self.iter_idx, self._last_line)
            self.times[key] = self.times.get(key, 0.0) + (now - self._last_t)
            self.hits[key] = self.hits.get(key, 0) + 1
        if lineno == self.boundary:
            self.iter_idx += 1
        self._last_line = lineno
        self._last_t = now

    def _finish(self) -> None:
        if self._last_line is not None:
            key = (self.iter_idx, self._last_line)
            self.times[key] = self.times.get(key, 0.0) + (
                time.perf_counter() - self._last_t)
            self.hits[key] = self.hits.get(key, 0) + 1
            self._last_line = None

    # -- sys.monitoring backend -------------------------------------------

    def _on_line(self, code, line_number):
        self._tick(line_number)

    def _on_return(self, code, offset, retval):
        self._finish()

    # -- sys.settrace backend ---------------------------------------------

    def _global_trace(self, frame, event, arg):
        if event == "call" and frame.f_code is self.code:
            return self._local_trace
        return None

    def _local_trace(self, frame, event, arg):
        if event == "line":
            self._tick(frame.f_lineno)
        elif event == "return":
            self._finish()
        return self._local_trace

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> str:
        mon = getattr(sys, "monitoring", None)
        if mon is not None and hasattr(mon, "set_local_events"):
            try:
                events = mon.events
                mon.use_tool_id(self.TOOL_ID, "joe-line-clock")
                mon.register_callback(self.TOOL_ID, events.LINE, self._on_line)
                mon.register_callback(self.TOOL_ID, events.PY_RETURN,
                                      self._on_return)
                mon.set_local_events(self.TOOL_ID, self.code,
                                     events.LINE | events.PY_RETURN)
                self.mode = "sys.monitoring"
                return self.mode
            except Exception as exc:  # pragma: no cover - backend fallback
                print(f"sys.monitoring unavailable ({exc}); using settrace",
                      flush=True)
        sys.settrace(self._global_trace)
        self.mode = "sys.settrace"
        return self.mode

    def stop(self) -> None:
        if self.mode == "sys.monitoring":
            mon = sys.monitoring
            mon.set_local_events(self.TOOL_ID, self.code, 0)
            mon.free_tool_id(self.TOOL_ID)
        elif self.mode == "sys.settrace":
            sys.settrace(None)
        self._finish()


# Phase anchors, in loop order. Each entry is (phase name, a substring that
# appears on the line where the phase starts). Matching is sequential, so a
# repeated line (``network = _get_network()``) resolves to the right one.
LOOP_ANCHORS = [
    ("01 eval (vs random)", "network = _get_network()"),
    ("02 curriculum check", "t0 = time.time()"),
    ("03 pool refresh", "# Periodic pool refresh"),
    ("04 rollout (self-play)", "# Collect rollout"),
    ("05 unpack rollout", "# Shapes: (D, num_steps"),
    ("06 GAE + adv normalize", "advs = p_gae("),
    ("07 diagnostics (host sync)", "# Rollout-level diagnostics"),
    ("08 diagnostics MC returns", "mc_rets, mc_valid = p_mc_returns"),
    ("09 PPO update", "# PPO update"),
    ("10 metrics host sync", "m = jax.tree.map(lambda x: x[0], metrics)"),
    ("11 episode stats", "elapsed = time.time() - t0"),
    ("12 progress print", "wall = int(time.time() - train_start)"),
    ("13 metrics log", "log_metrics = {"),
    ("14 EMA update", "# Update EMA params"),
    ("15 checkpoint", "network = _get_network()"),
    ("16 free arrays", "# Free large arrays"),
]

SETUP_ANCHORS = [
    ("00a setup: config + curriculum print", "def train("),
    ("00b setup: env + pool generation", "t0 = time.time()"),
    ("00c setup: partition/replicate params",
     "params, static = eqx.partition(network, eqx.is_array)"),
    ("00d setup: init envs + obs state", "states = p_init_envs("),
    ("00e setup: pmap definitions", "def _compute_gae("),
    ("00f setup: EMA init", "ema_decay = cfg.ema_decay"),
]


def build_line_phase_map(src_lines, first_lineno, loop_lineno):
    """lineno -> phase name for every source line of ``train``.

    Anchors are consumed in order, so a line that appears twice in the
    function (``network = _get_network()``) resolves to the occurrence its
    anchor's position implies.
    """
    phase_of = {}
    setup = list(SETUP_ANCHORS)
    loop = list(LOOP_ANCHORS)
    current = SETUP_ANCHORS[0][0]
    for offset, text in enumerate(src_lines):
        lineno = first_lineno + offset
        pending = setup if lineno < loop_lineno else loop
        if pending and pending[0][1] in text:
            current = pending.pop(0)[0]
        phase_of[lineno] = current
    unmatched = [name for name, _ in setup] + [name for name, _ in loop]
    return phase_of, unmatched


def _fmt(seconds: float) -> str:
    return f"{seconds:8.3f}"


def _table(title, rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [f"\n### {title}", line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Remote profile
# ---------------------------------------------------------------------------


def _profile_impl(gpu_label: str, cfg_dict: dict, engine_sha: str) -> dict:
    """Runs in the container: instrument ppo.train, run it, summarize."""
    import inspect
    import os
    import shutil

    import jax

    from training.joe.config import Config
    from training.joe.main import run
    from training.joe.train import ppo

    dev = jax.devices()[0]
    print(f"[profile:{gpu_label}] device={dev} jax={jax.__version__}",
          flush=True)

    src_lines, first_lineno = inspect.getsourcelines(ppo.train)
    loop_lineno = None
    body_lineno = None
    for offset, text in enumerate(src_lines):
        if "for it in range(start_step, cfg.num_iters):" in text:
            loop_lineno = first_lineno + offset
            body_lineno = first_lineno + offset + 2  # skip the comment line
            break
    if loop_lineno is None:
        raise RuntimeError("could not find the training for-loop in ppo.train")
    # The boundary is the first *statement* of the loop body.
    for offset, text in enumerate(src_lines):
        lineno = first_lineno + offset
        if lineno > loop_lineno and text.strip() and not text.strip().startswith("#"):
            body_lineno = lineno
            break
    print(f"[profile:{gpu_label}] ppo.train lines {first_lineno}.."
          f"{first_lineno + len(src_lines) - 1}, loop at {loop_lineno}, "
          f"iteration boundary at {body_lineno}", flush=True)

    phase_of, unmatched = build_line_phase_map(
        src_lines, first_lineno, loop_lineno)
    if unmatched:
        print(f"WARNING: phase anchors never matched: {unmatched}", flush=True)

    cfg = Config.from_dict(cfg_dict, source="profile")
    # Always cold: a leftover state.json would resume the run and skip the
    # very iterations (and the first-call JIT compiles) we are measuring.
    ckpt_dir = f"/tmp/joe-profile/{cfg.run_name}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    clock = LineClock(ppo.train.__code__, body_lineno)
    mode = clock.start()
    print(f"[profile:{gpu_label}] line clock backend: {mode}", flush=True)
    wall0 = time.perf_counter()
    try:
        run(cfg, ckpt_dir, engine_sha=engine_sha)
    finally:
        clock.stop()
    total_wall = time.perf_counter() - wall0

    if not clock.times:
        raise RuntimeError("line clock recorded nothing — backend failed")

    # ---- aggregate -------------------------------------------------------
    iters = sorted({i for i, _ in clock.times})
    train_iters = [i for i in iters if i >= 0]
    steady = [i for i in train_iters if i > 0]

    def phase_totals(iteration_filter):
        out = {}
        for (i, lineno), secs in clock.times.items():
            if not iteration_filter(i):
                continue
            out[phase_of.get(lineno, "?")] = out.get(
                phase_of.get(lineno, "?"), 0.0) + secs
        return out

    setup_tot = phase_totals(lambda i: i < 0)
    cold_tot = phase_totals(lambda i: i == 0)
    steady_tot = phase_totals(lambda i: i > 0)
    all_loop_tot = phase_totals(lambda i: i >= 0)
    n_steady = max(len(steady), 1)

    per_iter = {}
    for (i, lineno), secs in clock.times.items():
        per_iter[i] = per_iter.get(i, 0.0) + secs

    # Per-line detail, steady state only.
    line_steady = {}
    for (i, lineno), secs in clock.times.items():
        if i > 0:
            line_steady[lineno] = line_steady.get(lineno, 0.0) + secs

    # Per-iteration x per-phase. The plain steady-state mean is misleading:
    # a few iterations carry a second wave of JIT compiles, and the eval,
    # pool-refresh and checkpoint phases fire on their own cadences. This
    # table is what separates a clean iteration from those.
    per_iter_phase = {}
    for (i, lineno), secs in clock.times.items():
        if i < 0:
            continue
        ph = phase_of.get(lineno, "?")
        per_iter_phase.setdefault(i, {})[ph] = round(
            per_iter_phase.setdefault(i, {}).get(ph, 0.0) + secs, 4)

    src_by_lineno = {first_lineno + o: t.rstrip()
                     for o, t in enumerate(src_lines)}

    clocked = sum(clock.times.values())
    outside = total_wall - clocked

    mem = dev.memory_stats() or {}
    result = {
        "gpu": gpu_label,
        "device": str(dev),
        "jax": jax.__version__,
        "engine_sha": engine_sha,
        "config": cfg.to_dict(),
        "total_wall_s": round(total_wall, 2),
        "clock_backend": mode,
        "iterations": len(train_iters),
        "setup_phases": {k: round(v, 3) for k, v in sorted(setup_tot.items())},
        "cold_iter0_phases": {k: round(v, 3) for k, v in sorted(cold_tot.items())},
        "steady_phases_total": {k: round(v, 3) for k, v in sorted(steady_tot.items())},
        "steady_phases_per_iter": {
            k: round(v / n_steady, 4) for k, v in sorted(steady_tot.items())},
        "per_iteration_wall_s": {str(k): round(v, 3)
                                 for k, v in sorted(per_iter.items())},
        "steady_top_lines": [
            {"lineno": ln, "s_per_iter": round(s / n_steady, 4),
             "src": src_by_lineno.get(ln, "")}
            for ln, s in sorted(line_steady.items(), key=lambda kv: -kv[1])[:25]
        ],
        "peak_device_gib": round(mem.get("peak_bytes_in_use", 0) / 2**30, 2),
        "n_steady_iters": len(steady),
        "clocked_in_train_s": round(clocked, 2),
        "outside_train_s": round(outside, 2),
        "per_iteration_phases": {str(k): v
                                 for k, v in sorted(per_iter_phase.items())},
    }

    # ---- print report ----------------------------------------------------
    print(f"\n{'=' * 78}\nPROFILE REPORT — {gpu_label}\n{'=' * 78}", flush=True)
    print(f"config: depth={cfg.depth} embed={cfg.embed_dim} ff={cfg.ff_factor} "
          f"envs={cfg.num_envs} steps={cfg.num_steps} mb={cfg.minibatch_size} "
          f"pool={cfg.pool_size} iters={cfg.num_iters} bf16={cfg.use_bf16}")
    print(f"total wall {total_wall:.1f}s | inside ppo.train {clocked:.1f}s | "
          f"outside (import, net build, manifest, final save) {outside:.1f}s | "
          f"peak device mem {result['peak_device_gib']} GiB")

    setup_sum = sum(setup_tot.values())
    rows = [(k, _fmt(v), f"{100 * v / max(setup_sum, 1e-9):5.1f}%")
            for k, v in sorted(setup_tot.items(), key=lambda kv: -kv[1])]
    rows.append(("TOTAL setup (before iter 0)", _fmt(setup_sum), "100.0%"))
    print(_table("Setup, once per run", rows, ["phase", "sec", "share"]))

    cold_sum = sum(cold_tot.values())
    rows = [(k, _fmt(v), f"{100 * v / max(cold_sum, 1e-9):5.1f}%")
            for k, v in sorted(cold_tot.items(), key=lambda kv: -kv[1])]
    rows.append(("TOTAL iteration 0", _fmt(cold_sum), "100.0%"))
    print(_table("Iteration 0 — includes all cold JIT compiles", rows,
                 ["phase", "sec", "share"]))

    steady_sum = sum(steady_tot.values())
    rows = []
    for k, v in sorted(steady_tot.items(), key=lambda kv: -kv[1]):
        rows.append((k, _fmt(v / n_steady), _fmt(v),
                     f"{100 * v / max(steady_sum, 1e-9):5.1f}%"))
    rows.append((f"TOTAL steady state ({len(steady)} iters)",
                 _fmt(steady_sum / n_steady), _fmt(steady_sum), "100.0%"))
    print(_table(f"Steady state — iterations 1..{max(train_iters)}", rows,
                 ["phase", "s/iter", "sec total", "share"]))

    rows = [(str(i), _fmt(per_iter[i])) for i in sorted(per_iter)]
    print(_table("Wall clock per iteration (-1 = setup)", rows,
                 ["iter", "sec"]))

    rows = [(str(d["lineno"]), _fmt(d["s_per_iter"]), d["src"][:88])
            for d in result["steady_top_lines"] if d["s_per_iter"] > 1e-4]
    print(_table("Hottest source lines, steady state (ppo.py)", rows,
                 ["line", "s/iter", "source"]))

    # Per-iteration x phase, for the phases that actually move.
    big = [p for p in sorted(all_loop_tot) if all_loop_tot[p] > 0.01]
    short = {p: p.split(" ", 1)[0] for p in big}
    rows = []
    for i in sorted(per_iter_phase):
        cells = [str(i)]
        for ph in big:
            cells.append(f"{per_iter_phase[i].get(ph, 0.0):.2f}")
        cells.append(f"{sum(per_iter_phase[i].values()):.2f}")
        rows.append(tuple(cells))
    print(_table("Per-iteration x phase (seconds); columns are phase numbers",
                 rows, ["it"] + [short[p] for p in big] + ["total"]))
    print("  " + " | ".join(f"{short[p]}={p.split(' ', 1)[1]}" for p in big))

    print("\nRESULTS_JSON_BEGIN", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)
    print("RESULTS_JSON_END", flush=True)
    return result


@app.function(image=IMAGE, gpu="T4", timeout=3600)
def profile_t4(cfg_dict: dict, engine_sha: str) -> dict:
    return _profile_impl("T4", cfg_dict, engine_sha)


@app.function(image=IMAGE, gpu="L4", timeout=3600)
def profile_l4(cfg_dict: dict, engine_sha: str) -> dict:
    return _profile_impl("L4", cfg_dict, engine_sha)


@app.function(image=IMAGE, gpu="A10G", timeout=3600)
def profile_a10g(cfg_dict: dict, engine_sha: str) -> dict:
    return _profile_impl("A10G", cfg_dict, engine_sha)


# ---------------------------------------------------------------------------
# Rollout drill-down: split the one big compiled region into its stages
# ---------------------------------------------------------------------------


def _drilldown_impl(gpu_label: str, cfg_dict: dict) -> dict:
    """Cumulative-stack timing of the rollout's four stages.

    ``collect_rollout`` is a single fused XLA program, so a phase profile
    can only report it as one number. This builds the same pipeline in
    four nested versions and reports the differences:

        S1  env.step only
        S2  S1 + observations, build cost, move/build masks
        S3  S2 + observation augmentation (39-channel obs + temporal)
        S4  the real collect_rollout (S3 + network forward + sampling)

    Every stage carries real state forward (env states, then the augmented
    obs state), so nothing is hoisted out of the scan or dead-coded. S1-S3
    drive the env with a fixed pass action instead of network actions; the
    env step cost is action-shape invariant, but the game trajectories
    differ from S4's, so read the differences as stage costs, not as an
    exact decomposition of S4 (XLA also fuses across stage boundaries).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from training.joe.config import Config
    from training.joe.env import make_competition_env, preset_min_generals_distance
    from training.joe.networks import build_network, get_network_bundle
    from training.joe.train.rollout_selfplay import _observe_both, collect_rollout

    cfg = Config.from_dict(cfg_dict, source="drilldown")
    bundle = get_network_bundle(cfg.network)
    augment_fn = bundle["augment_obs"]
    init_obs_state_fn = bundle["init_obs_state"]

    dev = jax.devices()[0]
    N, T = cfg.num_envs, cfg.num_steps
    print(f"[drill:{gpu_label}] device={dev} envs={N} steps={T}", flush=True)

    env = make_competition_env(
        preset_min_generals_distance(), None, cfg.pool_size)
    t0 = time.perf_counter()
    pool, _ = env.reset(jrandom.PRNGKey(0))
    jax.block_until_ready(pool.armies)
    pool_s = time.perf_counter() - t0
    print(f"[drill:{gpu_label}] pool {cfg.pool_size} in {pool_s:.1f}s", flush=True)

    network = build_network(cfg, jrandom.PRNGKey(1))
    states = jax.vmap(env.init_state)(jrandom.split(jrandom.PRNGKey(2), N))
    single = init_obs_state_fn(cfg.pad_to, cfg.pad_to)
    osp = jax.tree.map(lambda x: jnp.tile(x, (2 * N, *([1] * x.ndim))), single)
    osp0 = jax.tree.map(lambda x: jnp.tile(x, (N, *([1] * x.ndim))), single)
    # [pass, r, c, dir, is_half] -- a legal no-op for both seats
    pass_act = jnp.tile(jnp.array([1, 0, 0, 0, 0], jnp.int32), (N, 2, 1))

    def _step(s, acts, pool_):
        return jax.vmap(lambda st, a: env.step(st, a, pool_))(s, acts)

    @jax.jit
    def s1(states, pool_, acts):
        def body(s, _):
            _, s2 = _step(s, acts, pool_)
            return s2, None
        return jax.lax.scan(body, states, None, length=T)[0]

    @jax.jit
    def s2(states, pool_, acts):
        def body(carry, _):
            s, acc = carry
            _, obs_arr, cost, move, build = _observe_both(s)
            acc = (acc + obs_arr.sum() + cost.sum()
                   + move.sum() + build.sum())
            _, s2_ = _step(s, acts, pool_)
            return (s2_, acc), None
        return jax.lax.scan(body, (states, jnp.float32(0)), None, length=T)[0]

    @jax.jit
    def s3(states, pool_, acts, osp_):
        def body(carry, _):
            s, o, acc = carry
            _, obs_arr, cost, move, build = _observe_both(s)
            obs_aug, new_o = jax.vmap(augment_fn)(obs_arr, cost, o)
            obs_aug = obs_aug.astype(jnp.bfloat16)
            temporal = jnp.stack([new_o.opponent_army_history,
                                  new_o.opponent_land_history], axis=1)
            acc = (acc + obs_aug.sum().astype(jnp.float32) + temporal.sum()
                   + move.sum() + build.sum())
            _, s2_ = _step(s, acts, pool_)
            return (s2_, new_o, acc), None
        return jax.lax.scan(
            body, (states, osp_, jnp.float32(0)), None, length=T)[0]

    @eqx.filter_jit
    def s4(states, network, key, osp_a, osp_b, pool_):
        return collect_rollout(states, env, network, key, T,
                               osp_a, osp_b, augment_fn, pool_)

    def bench(label, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        compile_s = time.perf_counter() - t0
        reps, t0 = 3, time.perf_counter()
        for _ in range(reps):
            out = fn(*args)
        jax.block_until_ready(out)
        run_s = (time.perf_counter() - t0) / reps
        del out
        print(f"[drill:{gpu_label}] {label}: {run_s:.3f}s "
              f"(first call incl. compile {compile_s:.1f}s)", flush=True)
        return {"stage": label, "s": round(run_s, 4),
                "first_call_s": round(compile_s, 2)}

    rows = [
        bench("S1 env.step only", s1, states, pool, pass_act),
        bench("S2 + observations & masks", s2, states, pool, pass_act),
        bench("S3 + augmentation", s3, states, pool, pass_act, osp),
        bench("S4 + network forward (real collect_rollout)", s4,
              states, network, jrandom.PRNGKey(3), osp0, osp0, pool),
    ]
    t = {r["stage"]: r["s"] for r in rows}
    deltas = {
        "env.step": round(rows[0]["s"], 4),
        "observations + masks": round(rows[1]["s"] - rows[0]["s"], 4),
        "augmentation": round(rows[2]["s"] - rows[1]["s"], 4),
        "network forward + sampling": round(rows[3]["s"] - rows[2]["s"], 4),
    }
    total = rows[3]["s"]
    print(f"\n[drill:{gpu_label}] rollout stage split "
          f"(total {total:.3f}s for {T} steps x {2 * N} seats):", flush=True)
    for k, v in deltas.items():
        print(f"    {k:<32} {v:7.3f}s  {100 * v / max(total, 1e-9):5.1f}%",
              flush=True)

    mem = dev.memory_stats() or {}
    result = {
        "gpu": gpu_label, "device": str(dev), "num_envs": N, "num_steps": T,
        "pool_size": cfg.pool_size, "pool_gen_cold_s": round(pool_s, 2),
        "stages": rows, "stage_deltas_s": deltas,
        "rollout_total_s": round(total, 4),
        "peak_device_gib": round(mem.get("peak_bytes_in_use", 0) / 2**30, 2),
    }
    print("\nDRILL_JSON_BEGIN", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)
    print("DRILL_JSON_END", flush=True)
    return result


@app.function(image=IMAGE, gpu="T4", timeout=3600)
def drilldown_t4(cfg_dict: dict) -> dict:
    return _drilldown_impl("T4", cfg_dict)


def _bf16_ab_impl(gpu_label: str, cfg_dict: dict) -> dict:
    """Same-host A/B of use_bf16 on the real rollout, plus a raw GEMM probe.

    Turing (sm_75, the T4) has no native bfloat16: cuBLAS bf16 GEMM needs
    sm_80+. Ampere and later (A10G, L4, A100, H100) do. This times the
    identical ``collect_rollout`` with ``use_bf16`` true and false,
    interleaved in one container so the contrast is same-host, and adds a
    bare matmul at the network's hidden shape as a hardware control.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from training.joe.config import Config
    from training.joe.env import make_competition_env, preset_min_generals_distance
    from training.joe.networks import build_network, get_network_bundle
    from training.joe.train.rollout_selfplay import collect_rollout

    cfg_true = Config.from_dict({**cfg_dict, "use_bf16": True}, source="ab")
    cfg_false = Config.from_dict({**cfg_dict, "use_bf16": False}, source="ab")
    bundle = get_network_bundle(cfg_true.network)
    augment_fn = bundle["augment_obs"]
    init_obs_state_fn = bundle["init_obs_state"]

    dev = jax.devices()[0]
    N, T = cfg_true.num_envs, cfg_true.num_steps
    print(f"[ab:{gpu_label}] device={dev} envs={N} steps={T}", flush=True)

    env = make_competition_env(
        preset_min_generals_distance(), None, cfg_true.pool_size)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    jax.block_until_ready(pool.armies)
    states = jax.vmap(env.init_state)(jrandom.split(jrandom.PRNGKey(2), N))
    single = init_obs_state_fn(cfg_true.pad_to, cfg_true.pad_to)
    osp0 = jax.tree.map(lambda x: jnp.tile(x, (N, *([1] * x.ndim))), single)

    # Same PRNG key for both nets: identical weights, only the static
    # use_bf16 flag differs.
    net_bf16 = build_network(cfg_true, jrandom.PRNGKey(1))
    net_f32 = build_network(cfg_false, jrandom.PRNGKey(1))

    @eqx.filter_jit
    def roll(states, network, key, osp_a, osp_b, pool_):
        return collect_rollout(states, env, network, key, T,
                               osp_a, osp_b, augment_fn, pool_)

    def timed(network, reps=2):
        args = (states, network, jrandom.PRNGKey(3), osp0, osp0, pool)
        out = roll(*args)
        jax.block_until_ready(out)
        t0 = time.perf_counter()
        for _ in range(reps):
            out = roll(*args)
        jax.block_until_ready(out)
        dt = (time.perf_counter() - t0) / reps
        del out
        return dt

    # Interleaved A/B/A/B so any drift hits both arms equally.
    rows = []
    for rnd in range(2):
        t_bf16 = timed(net_bf16)
        t_f32 = timed(net_f32)
        rows.append({"round": rnd, "bf16_s": round(t_bf16, 3),
                     "f32_s": round(t_f32, 3),
                     "bf16_over_f32": round(t_bf16 / max(t_f32, 1e-9), 2)})
        print(f"[ab:{gpu_label}] round {rnd}: rollout bf16 {t_bf16:.3f}s vs "
              f"f32 {t_f32:.3f}s  ({t_bf16 / max(t_f32, 1e-9):.2f}x)", flush=True)

    # Hardware control: a bare GEMM at the network's hidden width.
    gemm = {}
    d = cfg_true.embed_dim
    a32 = jrandom.normal(jrandom.PRNGKey(7), (4096, d), dtype=jnp.float32)
    b32 = jrandom.normal(jrandom.PRNGKey(8), (d, 3 * d), dtype=jnp.float32)
    for name, dt in (("f32", jnp.float32), ("bf16", jnp.bfloat16),
                     ("f16", jnp.float16)):
        a, b = a32.astype(dt), b32.astype(dt)
        f = jax.jit(lambda x, y: (x @ y).sum())
        jax.block_until_ready(f(a, b))
        t0 = time.perf_counter()
        for _ in range(50):
            o = f(a, b)
        jax.block_until_ready(o)
        gemm[name] = round((time.perf_counter() - t0) / 50 * 1e3, 3)
        print(f"[ab:{gpu_label}] gemm 4096x{d} @ {d}x{3 * d} {name}: "
              f"{gemm[name]:.3f} ms", flush=True)

    result = {"gpu": gpu_label, "device": str(dev), "rounds": rows,
              "gemm_ms": gemm, "num_envs": N, "num_steps": T,
              "embed_dim": d}
    print("\nAB_JSON_BEGIN", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)
    print("AB_JSON_END", flush=True)
    return result


@app.function(image=IMAGE, gpu="T4", timeout=3600)
def bf16_ab_t4(cfg_dict: dict) -> dict:
    return _bf16_ab_impl("T4", cfg_dict)


@app.function(image=IMAGE, gpu="L4", timeout=3600)
def profile_l4_full(cfg_dict: dict, engine_sha: str) -> dict:
    return _profile_impl("L4", cfg_dict, engine_sha)


def _local_engine_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO / "competition-module"), "rev-parse", "HEAD"],
        text=True).strip()


# Small-but-real shape. Architecture is the frozen S tier; only the batch,
# the pool, and the cadences shrink so a T4 finishes a handful of iterations.
SMALL_OVERRIDES = dict(
    run_name="joe-profile",
    # Architecture stays the frozen S tier (5.1 M params). Only the batch
    # and the cadences shrink, so a T4 finishes ten iterations in minutes.
    num_envs=256,
    num_steps=64,
    minibatch_size=1024,
    num_iters=10,
    # Production pool size on purpose: the cold generation and the warm
    # refresh are two of the phases being measured, and both scale with it.
    pool_size=200_000,
    # Cadences chosen so the periodic phases land on different iterations:
    # eval at 0/3/7, pool refresh at 5, checkpoint at 8. Iterations
    # 1/2/4/6/9 are then clean steady state.
    reset_pool_every=5,
    eval_every=4,
    eval_games=64,
    ckpt_every=9,
    save_every=9,
    curriculum=None,   # single stage at the competition preset (distance 17+)
)


@app.local_entrypoint()
def main(gpu: str = "T4", tier: str = "S", overrides: str = "",
         mode: str = "phases"):
    """mode=phases: per-phase profile of the whole training loop.

    mode=drilldown: split the rollout into env / observation / augment /
    network stages. The drill-down needs only a valid pool, so it uses a
    small one and skips the two-minute production pool generation.
    """
    import yaml

    with open(REPO / "training" / "joe" / "configs" / f"{tier}.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.update(SMALL_OVERRIDES)
    cfg_dict["run_name"] = f"joe-profile-{gpu}"
    if mode in ("drilldown", "bf16ab"):
        cfg_dict["pool_size"] = 8192
    if overrides:
        cfg_dict.update(json.loads(overrides))

    if mode == "drilldown":
        print(f"Rollout drill-down on {gpu} (tier {tier})", flush=True)
        result = drilldown_t4.remote(cfg_dict)
        out = REPO / "data" / "joe" / f"profile-{gpu}-drilldown.json"
    elif mode == "bf16ab":
        print(f"use_bf16 A/B on {gpu} (tier {tier})", flush=True)
        result = bf16_ab_t4.remote(cfg_dict)
        out = REPO / "data" / "joe" / f"profile-{gpu}-bf16ab.json"
    else:
        fns = {"T4": profile_t4, "L4": profile_l4, "A10G": profile_a10g}
        engine_sha = _local_engine_sha()
        print(f"Profiling {cfg_dict['run_name']} on {gpu} "
              f"(tier {tier}, engine {engine_sha[:12]})", flush=True)
        result = fns[gpu.upper()].remote(cfg_dict, engine_sha)
        out = REPO / "data" / "joe" / f"profile-{gpu}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"DONE: wrote {out}", flush=True)
