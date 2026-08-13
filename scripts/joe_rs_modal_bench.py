#!/usr/bin/env python3
"""joe-rs J3 latency gate on x86: full per-move path, one core, no GPU.

    modal run scripts/joe_rs_modal_bench.py > /tmp/joe_rs_bench.log 2>&1 &
    modal app list          # then check startup a few minutes in
    modal app logs <app-id>

Writes docs/research/measurements/joe-rs-latency-modal.json (the md sits
next to it, written by hand from the numbers).

Milestone J3 of docs/bots/joe-rs/port-plan.md: full per-move path (parse ->
obs -> forward -> reply) on one x86 core, p99 <= 50 ms against the 150 ms
move limit; R1's tripwire is p99 > 75 ms. The input is the recorded
synthetic-long wire log (1,320 turns of real frames), the same stream the
parity sequence gate replays, via the binary's own `bench` subcommand.

The crate is built from source in the container (target-cpu=x86-64-v3 from
`.cargo/config.toml`), so the figure is what a sandbox `cargo build
--release` would produce. Modal is a proxy, not the target — same caveat as
every CPU measurement before it (fleet generations differ run to run).
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"
IN_LOG = REPO / "data" / "joe" / "joe-rs-parity" / "games" / "synthetic-long.in.log"

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential")
    # Network is allowed at image-build time; the bench itself resolves
    # nothing. Toolchain tracks the sandbox's 1.97 stable (port-plan §1).
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain 1.97.1"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir(str(REPO / "bots" / "joe-rs"), "/root/joe-rs", copy=True,
                   ignore=["target", "__pycache__", "tests"])
    .add_local_file(str(IN_LOG), "/root/synthetic-long.in.log", copy=True)
)

app = modal.App("joe-rs-latency-bench")


@app.function(image=IMAGE, cpu=1, memory=4096, timeout=60 * 60)
def bench() -> dict:
    import os
    import platform
    import subprocess
    import time

    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
    ):
        os.environ[var] = "1"

    out: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "online_cpus": os.cpu_count(),
        "cpu_model": "",
    }
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            out["cpu_model"] = line.split(":", 1)[1].strip()
            break

    t0 = time.time()
    build = subprocess.run(
        ["cargo", "build", "--release"], cwd="/root/joe-rs",
        capture_output=True, text=True)
    out["build_s"] = round(time.time() - t0, 1)
    if build.returncode != 0:
        out["error"] = build.stderr[-3000:]
        return out

    env = dict(os.environ)
    env["JOE_RS_ARTIFACT"] = "/root/joe-rs/artifact"
    with open("/root/synthetic-long.in.log", "rb") as f:
        run = subprocess.run(
            ["/root/joe-rs/target/release/joe-rs", "bench"],
            stdin=f, capture_output=True, text=True, env=env)
    out["bench_stdout"] = run.stdout.strip()
    out["bench_rc"] = run.returncode
    if run.returncode != 0:
        out["bench_stderr"] = run.stderr[-3000:]
    return out


@app.local_entrypoint()
def main() -> None:
    result = bench.remote()
    print(json.dumps(result, indent=2))
    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    path = MEASUREMENTS / "joe-rs-latency-modal.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {path}")
