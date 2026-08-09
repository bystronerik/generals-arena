#!/usr/bin/env python3
"""M7 qualification: the feasibility gate on one x86 core.

    modal run scripts/morpheus_rs_modal_m7.py --config n8-s32-b4-d8 --games 20
    modal run scripts/morpheus_rs_modal_m7.py --config n8-s32-b4-d8,n8-s16-b4-d2

Milestone M7 of docs/bots/morpheus-rs/rewrite-plan.md. §8's exit gate is
measured **here**, not on the laptop: "a `deployment.json` whose measured p99.9
move time on the x86 reference host is < 150 ms with zero overruns over ≥20
games". An arm64 laptop has no deployment authority — the competition host is
one x86 core running Linux, and M3 already found one bug (the missing hardware
FMA) that existed only there.

Runs `scripts/morpheus_rs_m7.py sweep` inside `@app.function(cpu=1)`, which is
the same code path the local sweep uses, and returns its summary. Nothing is
written back but the report: the traces themselves die with the container, and
the numbers that matter are the aggregates.

Two things the container forces, both the same as the M0 baseline's:

- **No git**, so nothing registers a lineage step. Every run here is unrated by
  construction anyway — the sweep drives `matchup.py` directly.
- **The Rust toolchain is installed at image-build time**, where network is
  allowed, pinned to the sandbox's 1.97 stable (rewrite-plan §1).

The proxy caveat from the CPU probe still applies: this is *a* one-core x86
Linux container, not the judge's sandbox, which cannot be measured directly.

**Do not run this while a local sweep is running.** `morpheus_rs_m7.py sweep`
overwrites `bots/morpheus-rs/deployment.json` for the duration of each config,
and Modal hashes the local directory as it copies it into the image — so a
concurrent sweep aborts the build with "deployment.json was modified during
build process". Serialize them; the failure is loud but the cause is not.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("bash", "curl", "build-essential", "util-linux")
    .pip_install("numpy==2.4.6", "jax==0.11.0", "jaxlib==0.11.0")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain 1.97.1"
    )
    .add_local_dir(str(REPO / "competition-module"), "/root/competition-module", copy=True)
    .add_local_dir(str(REPO / "arena"), "/root/arena", copy=True)
    .add_local_dir(str(REPO / "bots"), "/root/bots", copy=True,
                   ignore=["**/target", "**/__pycache__"])
    .add_local_dir(str(REPO / "scripts"), "/root/scripts", copy=True,
                   ignore=["**/__pycache__"])
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({
        "PYTHONPATH": "/root",
        "PYTHONUNBUFFERED": "1",
        "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
    })
)

app = modal.App("morpheus-rs-m7-qualify")


@app.function(image=IMAGE, cpu=1, memory=4096, timeout=60 * 60 * 5)
def qualify(configs: list[str], games: int, seed: int, guard_ms: float) -> dict:
    """Build the binary, sweep the configs, return the summaries."""
    import os
    import subprocess
    import sys
    import time

    sys.path.insert(0, "/root")
    out: dict = {"online_cpus": os.cpu_count(), "cpuinfo_model": ""}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                out["cpuinfo_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    # `cwd`, not just `--manifest-path`: cargo reads `.cargo/config.toml` from
    # the working directory, and that file carries `target-cpu=x86-64-v3`.
    # Without it a forward costs 277 ms instead of 5 — M3's most expensive bug.
    t0 = time.perf_counter()
    built = subprocess.run(
        ["cargo", "build", "--release"],
        cwd="/root/bots/morpheus-rs", capture_output=True, text=True,
    )
    out["rust_build_seconds"] = round(time.perf_counter() - t0, 1)
    out["rust_build_ok"] = built.returncode == 0
    if built.returncode != 0:
        out["rust_build_stderr"] = built.stderr[-4000:]
        return out
    out["rustc_version"] = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True
    ).stdout.strip()

    argv = [
        sys.executable, "/root/scripts/morpheus_rs_m7.py", "sweep",
        "--config", *configs,
        "--games", str(games),
        "--seed", str(seed),
        "--out", "/root/m7",
    ]
    if guard_ms is not None:
        argv += ["--extra", json.dumps({"admission_guard_ms": guard_ms})]
    t0 = time.perf_counter()
    sweep = subprocess.run(argv, capture_output=True, text=True)
    out["sweep_wall_s"] = round(time.perf_counter() - t0, 1)
    out["sweep_returncode"] = sweep.returncode
    out["sweep_stdout_tail"] = sweep.stdout[-3000:]
    out["sweep_stderr_tail"] = sweep.stderr[-3000:]

    report = subprocess.run(
        [sys.executable, "/root/scripts/morpheus_rs_m7.py", "report", "--out", "/root/m7"],
        capture_output=True, text=True,
    )
    out["report_stdout"] = report.stdout
    summary = Path("/root/m7/summary.json")
    if summary.is_file():
        out["summary"] = json.loads(summary.read_text())
    return out


def _markdown(payload: dict, games: int, guard_ms: float) -> str:
    summary = payload.get("summary") or {}
    lines: list[str] = []
    lines.append("# Morpheus-rs M7 — qualification on one x86 core\n")
    lines.append(
        "Milestone M7 of [the rewrite plan](../../bots/morpheus-rs/rewrite-plan.md). "
        "The exit gate is measured here rather than on the laptop: the competition "
        "host is one x86 core running Linux, and an arm64 developer machine has no "
        "deployment authority over it.\n"
    )
    lines.append(f"- Container: `@app.function(cpu=1)`, {payload.get('online_cpus')} CPUs visible")
    lines.append(f"- CPU: {payload.get('cpuinfo_model') or 'unknown'}")
    lines.append(f"- Toolchain: {payload.get('rustc_version', 'unknown')}, "
                 f"built in {payload.get('rust_build_seconds')} s")
    lines.append(f"- Games per config: {games} · `admission_guard_ms` = {guard_ms}\n")
    lines.append("| config | turns | p50 | p99 | p99.9 | max | >150 ms | sims p50 | belief ok |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in sorted(summary):
        s = summary[name]
        m = s["move_ms"]
        lines.append(
            f"| `{name}` | {s['turns']} | {m['p50']:.0f} | {m['p99']:.0f} | "
            f"{m['p999']:.0f} | {m['max']:.0f} | {s['over_judge_limit']} | "
            f"{s['completed_simulations']['p50']:.0f} | "
            f"{s['belief_plus_root_ok_frac'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## Per-component p99, per call\n")
    lines.append("The unit that survives a knob change: the search components run "
                 "once per simulation, so a per-turn total moves when the simulation "
                 "count does and says nothing about the kernel.\n")
    for name in sorted(summary):
        s = summary[name]
        lines.append(f"### `{name}`\n")
        lines.append("| component | ms/call p99 | calls/turn | ms/turn p99 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for comp in sorted(s["per_call_p99_ms"]):
            lines.append(
                f"| {comp} | {s['per_call_p99_ms'][comp]:.4f} | "
                f"{s['calls_mean'].get(comp, 0):.2f} | "
                f"{s['per_turn_p99_ms'].get(comp, 0):.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(config: str = "n8-s32-b4-d8", games: int = 20, seed: int = 8000,
         guard_ms: float = 0.0, out: str = "morpheus-rs-m7-x86") -> None:
    configs = [c for c in config.split(",") if c]
    payload = qualify.remote(configs, games, seed, guard_ms)
    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    (MEASUREMENTS / f"{out}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (MEASUREMENTS / f"{out}.md").write_text(_markdown(payload, games, guard_ms))
    print(payload.get("report_stdout", ""))
    print(f"wrote {MEASUREMENTS / out}.json and .md")
