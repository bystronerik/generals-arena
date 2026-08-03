#!/usr/bin/env python3
"""Modal CPU entry for Part 00b: sandbox static 8-bit export preflight.

Usage:
    modal run scripts/morpheus_modal_export_preflight.py

Writes (on the local machine after the remote probe returns):
    docs/research/measurements/morpheus-export-preflight.{json,md}

The image mirrors the competition sandbox pins on Linux CPython 3.12 with the
CPU torch wheel. This is the seat that can exercise fbgemm/x86 when QNNPACK is
not the judge engine. It does not use a GPU and does not spend A100 budget.
"""
from __future__ import annotations

import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

TORCH_PIN = "torch==2.13.0"
NUMPY_PIN = "numpy==2.4.6"
SAFETENSORS_PIN = "safetensors==0.8.0"
# Remaining sandbox pins (torch installed separately from the CPU index).
_SANDBOX_REST = (
    NUMPY_PIN,
    "scipy==1.18.0",
    "pandas==3.0.5",
    "scikit-learn==1.9.0",
    "jax==0.11.0",
    "jaxlib==0.11.0",
    "numba==0.66.0",
    "networkx==3.6.1",
    SAFETENSORS_PIN,
    "gymnasium==1.3.0",
)

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    # Match requirements-sandbox.txt: CPU torch before the rest of the pin set.
    .pip_install(
        TORCH_PIN,
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(*_SANDBOX_REST)
    .env({"PYTHONPATH": "/root"})
    # copy=True so pin files and training code land in the image; keep these
    # steps last (Modal forbids later build steps after non-copy add_local_*).
    .add_local_dir(
        str(REPO / "training"),
        remote_path="/root/training",
        copy=True,
    )
    .add_local_file(
        str(REPO / "requirements-sandbox.txt"),
        remote_path="/root/requirements-sandbox.txt",
        copy=True,
    )
    .add_local_file(
        str(REPO / "competition-module" / "competition" / "requirements.txt"),
        remote_path="/root/competition-module/competition/requirements.txt",
        copy=True,
    )
)

app = modal.App("morpheus-export-preflight")


@app.function(image=IMAGE, cpu=4, memory=8192, timeout=60 * 30)
def run_sandbox_export_cpu(
    seed: int = 0,
    n_blocks: int = 12,
) -> dict:
    """Run the export probe inside the Modal Linux CPU image."""
    from training.morpheus.export_preflight.measure import run_export_preflight

    return run_export_preflight(
        requirements_path=Path("/root/requirements-sandbox.txt"),
        seed=seed,
        n_blocks=n_blocks,
        seat="modal-cpu",
    )


@app.local_entrypoint()
def main(seed: int = 0, n_blocks: int = 12) -> None:
    from training.morpheus.export_preflight.report import build_report, write_report

    t0 = time.perf_counter()
    result = run_sandbox_export_cpu.remote(seed=seed, n_blocks=n_blocks)
    wall_s = time.perf_counter() - t0
    result.setdefault("host", {})
    result["host"]["modal_wall_s"] = wall_s

    report = build_report(result)
    json_path, md_path = write_report(report)
    verdict = report["decision"]["verdict"]
    viable = report["decision"]["viable_candidates"]
    engines = (result.get("host") or {}).get("supported_qengines")
    print(f"verdict={verdict}")
    print(f"viable={viable}")
    print(f"supported_qengines={engines}")
    print(f"seat={(result.get('host') or {}).get('seat')}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wall_s={wall_s:.1f}")
