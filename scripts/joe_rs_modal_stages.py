#!/usr/bin/env python3
"""joe-net N0.1: split joe-rs's per-move path into stages, on one x86 core.

    modal run scripts/joe_rs_modal_stages.py > /tmp/joe_rs_stages.log 2>&1 &
    modal app list          # then check startup a few minutes in
    modal app logs <app-id>

Writes docs/research/measurements/joe-net-n0-latency-modal.json.

docs/bots/morpheus-rs/joe-net-plan.md N0.1 needs two numbers that the
whole-move figure in docs/bots/joe-rs/latency.md cannot give: the forward
pass alone (F), which sets how many simulations survive a 130 ms turn, and
augment_obs alone, which decides whether the ported search can advance
joe's history per node. `joe-rs bench --stages` reports both.

The plain `bench` runs first in the same container, so the total stays
comparable to the recorded 21.4 ms p50 and a fleet-generation difference
shows up as a shift in a number we already know rather than as a surprise
in one we do not. Modal is a proxy, not the target — the same caveat as
every CPU measurement before it.
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

app = modal.App("joe-rs-stage-bench")


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

    def run(args: list[str]) -> subprocess.CompletedProcess:
        with open("/root/synthetic-long.in.log", "rb") as f:
            return subprocess.run(
                ["/root/joe-rs/target/release/joe-rs", *args],
                stdin=f, capture_output=True, text=True, env=env)

    plain = run(["bench"])
    out["bench_rc"] = plain.returncode
    out["bench_stdout"] = plain.stdout.strip()
    if plain.returncode != 0:
        out["bench_stderr"] = plain.stderr[-3000:]
        return out

    staged = run(["bench", "--stages"])
    out["stages_rc"] = staged.returncode
    if staged.returncode != 0:
        out["stages_stderr"] = staged.stderr[-3000:]
        return out
    lines = [ln for ln in staged.stdout.strip().splitlines() if ln.strip()]
    out["stages_total_line"] = lines[0]
    out["stages"] = json.loads(lines[-1])
    # The per-turn value telemetry floods stderr; keep only the table.
    out["stages_table"] = "\n".join(
        ln for ln in staged.stderr.splitlines() if " value " not in ln)
    return out


@app.local_entrypoint()
def main() -> None:
    result = bench.remote()
    print(json.dumps(result, indent=2))
    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    path = MEASUREMENTS / "joe-net-n0-latency-modal.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {path}")
