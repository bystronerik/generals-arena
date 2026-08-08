#!/usr/bin/env python3
"""M3 shoot-out on x86: the Rust engine against TorchScript, one core, no GPU.

    modal run scripts/morpheus_rs_modal_inference_bench.py

Writes docs/research/measurements/morpheus-rs-inference-bench-modal.{json,md}.

Milestone M3 of docs/bots/morpheus-rs/rewrite-plan.md. Its exit gate is worded
against **x86**, not against the development laptop, and for a specific
reason: TorchScript on x86 dispatches to oneDNN kernels that are tuned harder
than anything an M3 Pro's NEON path gets, so a Rust engine that wins on arm64
has not yet won the argument. The competition host is a one-core x86 Linux
container, and this is the closest thing to it that can be measured.

Both engines run in the same container, on the same deterministic input
stream, with every thread pool pinned to one. The Rust crate is built from
source here rather than shipped as a binary, so the figure is what the
sandbox's own `cargo build --release` would produce.

**Modal is a proxy, not the target** — the same caveat as the M0 CPU probe and
the M0.5 smoke test. It reports "one-core x86 Linux server container", and the
M0 probe already found that separate runs land on different fleet generations
(`morpheus-rs-cpu-probe.md`). Treat a single run as a shape, not a
qualification; M7 is where a deployment number gets accepted.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "util-linux")
    .pip_install("numpy==2.4.6")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    # Network is allowed at image-build time; the benchmark itself resolves
    # nothing. Version tracks the sandbox's 1.97 stable (rewrite-plan §1).
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "| sh -s -- -y --profile minimal --default-toolchain 1.97.1"
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir(str(REPO / "bots" / "morpheus-rs"), "/root/morpheus-rs", copy=True,
                   ignore=["target", "__pycache__"])
    .add_local_dir(str(REPO / "bots" / "morpheus"), "/root/morpheus", copy=True,
                   ignore=["__pycache__", "tests"])
)

app = modal.App("morpheus-rs-inference-bench")

BATCHES = (1, 4, 8)
ITERS = 200
# The LCG the Rust `bench` subcommand uses, so both engines see identical
# inputs down to the bit rather than each rolling its own "random".
LCG_SEED = 12345


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
        # Reserving one CPU does not hide the others; M0 saw 17 visible on a
        # one-core reservation. Recorded so a contended run is recognisable.
        "online_cpus": os.cpu_count(),
        "cpu_model": "",
        "flags": [],
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and not out["cpu_model"]:
                out["cpu_model"] = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not out["flags"]:
                out["flags"] = sorted(line.split(":", 1)[1].split())
    except OSError:
        pass
    out["avx2"] = "avx2" in out["flags"]
    out["avx512f"] = "avx512f" in out["flags"]

    # --- Rust ---------------------------------------------------------------
    build = time.perf_counter()
    # `cwd`, not just `--manifest-path`: cargo reads `.cargo/config.toml` from
    # the working directory upward, and that file sets `target-cpu=x86-64-v3`.
    # The first run of this script omitted the cwd and measured a baseline
    # x86-64 build at 277 ms per forward — `f32::mul_add` without a hardware
    # FMA becomes a libm call. The bench now records the target features it
    # actually compiled with so that mistake cannot be made silently twice.
    built = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", "/root/morpheus-rs/Cargo.toml"],
        cwd="/root/morpheus-rs",
        capture_output=True, text=True,
    )
    out["rust_build_seconds"] = time.perf_counter() - build
    out["rust_build_ok"] = built.returncode == 0
    if built.returncode != 0:
        out["rust_build_stderr"] = built.stderr[-4000:]
        return out
    out["rustc_version"] = subprocess.run(
        ["rustc", "--version"], capture_output=True, text=True
    ).stdout.strip()

    run = subprocess.run(
        ["/root/morpheus-rs/target/release/morpheus-rs", "bench", str(ITERS)],
        capture_output=True, text=True, cwd="/root/morpheus-rs",
    )
    if run.returncode != 0:
        out["rust_bench_stderr"] = run.stderr[-4000:]
        return out
    rust: dict = {}
    for line in run.stdout.splitlines():
        parts = line.split()
        if parts[:1] == ["load_ms"]:
            rust["load_ms"] = float(parts[1])
        elif parts[:1] == ["hardware_fma"]:
            rust["hardware_fma"] = parts[1] == "true"
        elif parts[:1] == ["warmup_ms"]:
            rust["warmup_ms"] = float(parts[1])
        elif parts[:1] == ["stage"]:
            rust.setdefault("stages", {})[parts[1]] = float(parts[2])
        elif len(parts) == 9 and parts[1] == "batch":
            rust[f"{parts[0]}_b{parts[2]}"] = {
                "p50": float(parts[4]), "p99": float(parts[6]), "min": float(parts[8])
            }
    out["rust"] = rust

    # --- TorchScript --------------------------------------------------------
    import torch

    torch.set_num_threads(1)
    art = Path("/root/morpheus/artifact")
    load = time.perf_counter()
    modules = {
        "policy": torch.jit.load(str(art / "model_policy.pt"), map_location="cpu").eval(),
        "policy_wdl": torch.jit.load(str(art / "model_policy_wdl.pt"), map_location="cpu").eval(),
        "full": torch.jit.load(str(art / "model.pt"), map_location="cpu").eval(),
    }
    torchscript: dict = {"load_ms": (time.perf_counter() - load) * 1e3,
                         "torch_version": torch.__version__}

    def lcg(batch: int):
        seed = LCG_SEED
        n = batch * 49 * 21 * 21
        values = torch.empty(n, dtype=torch.float32)
        for i in range(n):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            values[i] = ((seed >> 8) / 16777216.0) * 2.0 - 1.0
        return values.reshape(batch, 49, 21, 21)

    def pct(xs, q):
        xs = sorted(xs)
        return xs[max(1, min(len(xs), -(-int(q * len(xs) * 1000) // 1000)))- 1]

    for name, module in modules.items():
        for batch in BATCHES:
            x = lcg(batch)
            with torch.no_grad():
                for _ in range(10):
                    module(x)
                times = []
                for _ in range(ITERS):
                    t = time.perf_counter()
                    module(x)
                    times.append((time.perf_counter() - t) * 1e3)
            torchscript[f"{name}_b{batch}"] = {
                "p50": pct(times, 0.50), "p99": pct(times, 0.99), "min": min(times)
            }
    out["torchscript"] = torchscript
    return out


def _markdown(payload: dict) -> str:
    rust = payload.get("rust", {})
    ts = payload.get("torchscript", {})
    lines = [
        "# morpheus-rs inference on one x86 core (Modal)",
        "",
        "M3's exit gate, measured where it is written to apply. Generated by",
        "`scripts/morpheus_rs_modal_inference_bench.py`; the arm64 half of the",
        "shoot-out is in [`morpheus-rs-inference-bench.md`](morpheus-rs-inference-bench.md).",
        "",
        f"- Host: {payload.get('cpu_model') or 'unknown'} — "
        f"{payload.get('machine')}, {payload.get('online_cpus')} CPUs visible "
        "on a one-core reservation",
        f"- AVX2 {payload.get('avx2')}, AVX-512F {payload.get('avx512f')}",
        f"- {payload.get('rustc_version', 'rustc ?')}, "
        f"torch {ts.get('torch_version', '?')}",
        f"- Release build from source in {payload.get('rust_build_seconds', 0):.1f} s, "
        f"hardware FMA compiled in: **{rust.get('hardware_fma')}**",
        "",
        "## policy_wdl — the entry point root and leaf evaluation use",
        "",
        "| batch | TorchScript p99 (ms) | morpheus-rs p99 (ms) | speedup |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for batch in BATCHES:
        a = ts.get(f"policy_wdl_b{batch}", {}).get("p99")
        b = rust.get(f"policy_wdl_b{batch}", {}).get("p99")
        if a and b:
            lines.append(f"| {batch} | {a:.2f} | {b:.2f} | {a / b:.2f}× |")
    lines += [
        "",
        "## Stage breakdown (batch 1, morpheus-rs)",
        "",
        "| stage | ms |",
        "| --- | ---: |",
    ]
    for name, ms in (rust.get("stages") or {}).items():
        lines.append(f"| {name} | {ms:.3f} |")
    lines += [
        "",
        f"Load {rust.get('load_ms', 0):.1f} ms + warmup {rust.get('warmup_ms', 0):.1f} ms",
        f"against TorchScript's {ts.get('load_ms', 0):.0f} ms of module loading.",
        "",
        "**Modal is a proxy.** The generals.bot sandbox cannot be probed, and the",
        "M0 CPU probe found separate runs landing on different fleet generations.",
        "One run is a shape, not a qualification — M7 is where a deployment",
        "number gets accepted.",
        "",
    ]
    return "\n".join(lines)


@app.local_entrypoint()
def main() -> None:
    payload = bench.remote()
    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    out_json = MEASUREMENTS / "morpheus-rs-inference-bench-modal.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = MEASUREMENTS / "morpheus-rs-inference-bench-modal.md"
    out_md.write_text(_markdown(payload), encoding="utf-8")
    print(f"wrote {out_json.relative_to(REPO)}")
    print(f"wrote {out_md.relative_to(REPO)}")
    if not payload.get("rust_build_ok"):
        raise SystemExit("rust build failed in the container")
    # The first run of this script measured a build with no hardware FMA and
    # reported a 49x slowdown as if it were the engine's speed. Refusing to
    # publish that quietly is cheaper than noticing it a second time.
    if payload.get("rust", {}).get("hardware_fma") is False:
        raise SystemExit(
            "built without hardware FMA — every f32::mul_add is a libm call. "
            "Check that cargo ran with bots/morpheus-rs as its working "
            "directory so .cargo/config.toml applies."
        )
