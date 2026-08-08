#!/usr/bin/env python3
"""M3 shoot-out, local half: the Rust engine against TorchScript on this host.

    python bots/morpheus-rs/tools/bench_inference.py
    python bots/morpheus-rs/tools/bench_inference.py --iters 500

Writes docs/research/measurements/morpheus-rs-inference-bench.{json,md}.

The candle and tract figures in that report come from the spikes under
`tools/spikes/`, which need network to build and are therefore not run here;
their numbers are passed in with `--candle` / `--tract` or carried forward from
the committed JSON. Everything this script measures, it measures itself.

The x86 half — which is the half M3's exit gate is actually written against —
is `scripts/morpheus_rs_modal_inference_bench.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
REPO = BOT_DIR.parents[1]
MEASUREMENTS = REPO / "docs" / "research" / "measurements"
BINARY = BOT_DIR / "target" / "release" / "morpheus-rs"

BATCHES = (1, 4, 8)
LCG_SEED = 12345

# Multiply-accumulates in one forward, counted from the graph: stem
# 64·49·9·441, twelve blocks of (128·64 + 128·9 + 64·128)·441, and the eleven
# heads. Used only to turn a millisecond into a GFLOP/s, which is the number
# that says whether a kernel is slow or merely small.
MAC_PER_FORWARD = (
    64 * 49 * 9 * 441
    + 12 * (128 * 64 + 128 * 9 + 64 * 128) * 441
    + (9 + 1 + 16 + 1) * 64 * 441
    + (1 + 3 + 4) * 128
)


def pin_threads() -> None:
    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
    ):
        os.environ[var] = "1"


def lcg_input(batch: int):
    """The exact stream the Rust `bench` subcommand generates."""
    import torch

    seed = LCG_SEED
    n = batch * 49 * 21 * 21
    values = torch.empty(n, dtype=torch.float32)
    for i in range(n):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        values[i] = ((seed >> 8) / 16777216.0) * 2.0 - 1.0
    return values.reshape(batch, 49, 21, 21)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank, matching the Rust bench and the Python telemetry."""
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-int(q * len(ordered) * 1000) // 1000)))
    return ordered[rank - 1]


def run_rust(iters: int) -> dict:
    if not BINARY.is_file():
        raise SystemExit(
            f"no release binary at {BINARY}; run `cargo build --release "
            f"--manifest-path {BOT_DIR / 'Cargo.toml'}`"
        )
    result = subprocess.run(
        [str(BINARY), "bench", str(iters)],
        capture_output=True, text=True, cwd=str(BOT_DIR),
    )
    if result.returncode != 0:
        raise SystemExit(f"bench failed:\n{result.stderr[-2000:]}")
    out: dict = {"stages": {}}
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts[:1] == ["load_ms"]:
            out["load_ms"] = float(parts[1])
        elif parts[:1] == ["warmup_ms"]:
            out["warmup_ms"] = float(parts[1])
        elif parts[:1] == ["stage"]:
            out["stages"][parts[1]] = float(parts[2])
        elif len(parts) == 9 and parts[1] == "batch":
            out[f"{parts[0]}_b{parts[2]}"] = {
                "p50": float(parts[4]), "p99": float(parts[6]), "min": float(parts[8])
            }
    return out


def run_torchscript(iters: int) -> dict:
    import torch

    torch.set_num_threads(1)
    art = REPO / "bots" / "morpheus" / "artifact"
    load = time.perf_counter()
    modules = {
        "policy": torch.jit.load(str(art / "model_policy.pt"), map_location="cpu").eval(),
        "policy_wdl": torch.jit.load(str(art / "model_policy_wdl.pt"), map_location="cpu").eval(),
        "full": torch.jit.load(str(art / "model.pt"), map_location="cpu").eval(),
    }
    out = {"load_ms": (time.perf_counter() - load) * 1e3, "torch_version": torch.__version__}
    for name, module in modules.items():
        for batch in BATCHES:
            x = lcg_input(batch)
            with torch.no_grad():
                for _ in range(10):
                    module(x)
                times = []
                for _ in range(iters):
                    t = time.perf_counter()
                    module(x)
                    times.append((time.perf_counter() - t) * 1e3)
            out[f"{name}_b{batch}"] = {
                "p50": percentile(times, 0.50),
                "p99": percentile(times, 0.99),
                "min": min(times),
            }
    return out


def gflops(ms: float, batch: int) -> float:
    return (2 * MAC_PER_FORWARD * batch) / (ms * 1e-3) / 1e9


def markdown(payload: dict) -> str:
    rust, ts = payload["rust"], payload["torchscript"]
    spikes = payload.get("spikes", {})
    lines = [
        "# morpheus-rs inference: the M3 engine shoot-out",
        "",
        "Which inference engine the Rust bot ships, and why it is not the one",
        "the plan expected. Generated by `bots/morpheus-rs/tools/bench_inference.py`;",
        "the x86 half, which is what M3's exit gate is written against, comes",
        "from `scripts/morpheus_rs_modal_inference_bench.py`.",
        "",
        f"- Host: {payload['platform']}, {payload['machine']}",
        f"- Every thread pool pinned to one; torch {ts.get('torch_version')}",
        f"- {payload['iters']} timed forwards per cell, nearest-rank percentiles,",
        "  identical LCG inputs for every engine",
        f"- {MAC_PER_FORWARD / 1e6:.1f} MMAC per forward ({2 * MAC_PER_FORWARD / 1e6:.0f} MFLOP)",
        "",
        "## The shoot-out",
        "",
        "`policy_wdl`, the entry point root and leaf evaluation use. Lower is",
        "better; the ratio is against TorchScript, so above 1.00 is a loss.",
        "",
        "| engine | b1 p99 (ms) | ×TS | b4 p99 (ms) | ×TS | b8 p99 (ms) | ×TS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def row(label: str, table: dict, key: str) -> str:
        cells = []
        for batch in BATCHES:
            entry = table.get(f"{key}{batch}") if table else None
            base = ts.get(f"policy_wdl_b{batch}", {}).get("p99")
            if entry and base:
                cells.append(f"{entry['p99']:.2f} | {entry['p99'] / base:.2f}×")
            else:
                cells.append("— | —")
        return f"| {label} | " + " | ".join(cells) + " |"

    lines.append(row("TorchScript (the oracle)", ts, "policy_wdl_b"))
    lines.append(row("**morpheus-rs (shipped)**", rust, "policy_wdl_b"))
    for name in ("candle", "tract"):
        if name in spikes:
            lines.append(row(name, spikes[name], "b"))
    lines += [
        "",
        "## Where the time goes in the shipped engine",
        "",
        "| stage | ms (batch 1) | share |",
        "| --- | ---: | ---: |",
    ]
    stages = rust.get("stages", {})
    total = sum(stages.values()) or 1.0
    for name, ms in stages.items():
        lines.append(f"| {name} | {ms:.3f} | {100 * ms / total:.0f}% |")
    b1 = rust.get("policy_wdl_b1", {}).get("p50")
    if b1:
        lines += [
            "",
            f"Batch-1 p50 is {b1:.2f} ms, about {gflops(b1, 1):.0f} GFLOP/s on one core.",
            f"Load {rust.get('load_ms', 0):.1f} ms plus warmup {rust.get('warmup_ms', 0):.1f} ms,",
            f"against TorchScript's {ts.get('load_ms', 0):.0f} ms of module loading —"
            " which matters for the first-move grace window, not for a turn.",
        ]
    lines += [
        "",
        "## What these numbers are and are not",
        "",
        "Batch is a **loop** in the Rust engine, not a tensor dimension: its",
        "batch-4 figure is four sequential forwards timed together, which is",
        "what a leaf batch actually costs. TorchScript's batch-4 is one batched",
        "call. That is the honest comparison — the search wants a batch of leaves",
        "evaluated, and does not care how.",
        "",
        "This host is arm64. M3's exit gate is written against x86 because that",
        "is where the competition runs and where TorchScript dispatches to its",
        "best-tuned kernels. A win here is necessary and not sufficient.",
        "",
        "Interpretation, and why candle was dropped despite the plan ranking it",
        "first: [`docs/bots/morpheus-rs/inference.md`](../../bots/morpheus-rs/inference.md).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--skip-torchscript", action="store_true")
    parser.add_argument(
        "--spikes",
        type=Path,
        default=None,
        help="JSON of {engine: {b1: {...}}} from tools/spikes, merged into the report",
    )
    args = parser.parse_args(argv)
    pin_threads()

    out_json = MEASUREMENTS / "morpheus-rs-inference-bench.json"
    previous = json.loads(out_json.read_text()) if out_json.is_file() else {}

    payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "iters": args.iters,
        "mac_per_forward": MAC_PER_FORWARD,
        "rust": run_rust(args.iters),
        "torchscript": (
            previous.get("torchscript", {}) if args.skip_torchscript
            else run_torchscript(args.iters)
        ),
        "spikes": (
            json.loads(args.spikes.read_text()) if args.spikes
            else previous.get("spikes", {})
        ),
    }

    MEASUREMENTS.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    out_md = MEASUREMENTS / "morpheus-rs-inference-bench.md"
    out_md.write_text(markdown(payload))
    print(f"wrote {out_json.relative_to(REPO)}")
    print(f"wrote {out_md.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
