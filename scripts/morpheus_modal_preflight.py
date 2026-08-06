#!/usr/bin/env python3
"""Modal entry point for Part 00: competition JAX preflight on A100 (40GB).

Usage:
    modal run scripts/morpheus_modal_preflight.py

Writes:
    docs/research/measurements/morpheus-jax-preflight.{json,md}

Requires a Modal token (`modal token new`) and network access. The GPU seat
uses CUDA JAX; the CPU seat uses the CPU JAX wheel. Both seats run the same
seeds, action sequences, and parity fixtures.
"""
from __future__ import annotations

import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# Pin JAX to the sandbox major line; CUDA vs CPU wheels differ by extra.
JAX_PIN = "jax==0.11.0"
JAXLIB_PIN = "jaxlib==0.11.0"
NUMPY_PIN = "numpy==2.4.6"

_COMMON = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(NUMPY_PIN)
    .add_local_dir(
        str(REPO / "competition-module"),
        remote_path="/root/competition-module",
        copy=True,
    )
    .add_local_dir(
        str(REPO / "training"),
        remote_path="/root/training",
        copy=True,
    )
    # --no-deps: avoid pulling a second jax wheel over the CUDA/CPU pin below.
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root"})
)

# CUDA wheel on the GPU seat; CPU wheel on the CPU seat. Record exact versions
# in the measurement report — they are not the judge's CPU-only pin set.
GPU_IMAGE = _COMMON.pip_install("jax[cuda12]==0.11.0")
CPU_IMAGE = _COMMON.pip_install(JAX_PIN, JAXLIB_PIN)

app = modal.App("morpheus-jax-preflight")


@app.function(image=GPU_IMAGE, gpu="A100", timeout=60 * 30)
def run_gpu() -> dict:
    from training.morpheus.jax_preflight.measure import run_preflight_on_device

    return run_preflight_on_device(role="gpu")


@app.function(image=CPU_IMAGE, cpu=4, memory=8192, timeout=60 * 30)
def run_cpu() -> dict:
    from training.morpheus.jax_preflight.measure import run_preflight_on_device

    return run_preflight_on_device(role="cpu")


@app.local_entrypoint()
def main() -> None:
    from training.morpheus.jax_preflight.report import build_report, write_report

    t0 = time.perf_counter()
    # Fan out CPU and GPU seats. Charge A100 hours from the GPU seat's own
    # measured wall time, not the overlapped local wait.
    cpu_call = run_cpu.spawn()
    gpu_call = run_gpu.spawn()
    cpu_result = cpu_call.get()
    gpu_result = gpu_call.get()
    wall_s = time.perf_counter() - t0

    gpu_wall_s = float(gpu_result.get("device_wall_s") or wall_s)
    a100_hours = gpu_wall_s / 3600.0
    report = build_report(
        cpu_result,
        gpu_result,
        a100_hours=a100_hours,
        wall_s=wall_s,
    )
    json_path, md_path = write_report(report)
    verdict = report["decision"]["verdict"]
    print(f"verdict={verdict}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"a100_hours≈{a100_hours:.4f} wall_s={wall_s:.1f}")
