#!/usr/bin/env python3
"""Unified Modal entry points for Morpheus training gates.

Part 12:
    modal run scripts/morpheus_modal.py::ablate_objective \\
      --config training/morpheus/configs/objective-ablation.json

Later parts add train / resume / inspect on this app. A100 time from
``ablate_objective`` charges to Part 13.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

VOLUME = modal.Volume.from_name("morpheus-training", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "torch==2.6.0",
    )
    .add_local_dir(
        str(REPO / "competition-module"),
        remote_path="/root/competition-module",
        copy=True,
    )
    .add_local_dir(str(REPO / "arena"), remote_path="/root/arena", copy=True)
    .add_local_dir(str(REPO / "bots"), remote_path="/root/bots", copy=True)
    .add_local_dir(str(REPO / "training"), remote_path="/root/training", copy=True)
    .add_local_dir(
        str(REPO / "docs" / "research" / "measurements"),
        remote_path="/root/docs/research/measurements",
        copy=True,
    )
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root"})
)

app = modal.App("morpheus-training")


@app.function(
    image=IMAGE,
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/vol": VOLUME},
)
def ablate_objective_remote(config_json: str, run_id: str) -> dict:
    """Run Part 12 objective ablation; charge A100 time to Part 13."""
    from training.morpheus.objective.ablate import run_ablation

    t0 = time.perf_counter()
    cfg_path = Path("/tmp/objective-ablation.json")
    cfg_path.write_text(config_json, encoding="utf-8")
    out_dir = Path("/vol/morpheus/objective") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Also write the repo measurement paths when present.
    repo_json = Path("/root/docs/research/measurements/morpheus-objective-ablation.json")
    repo_md = Path("/root/docs/research/measurements/morpheus-objective-ablation.md")
    wall_s = 0.0
    report = run_ablation(
        cfg_path,
        json_path=repo_json if repo_json.parent.is_dir() else out_dir / "report.json",
        md_path=repo_md if repo_md.parent.is_dir() else out_dir / "report.md",
        a100_hours=0.0,  # filled below from wall clock
    )
    wall_s = time.perf_counter() - t0
    a100_hours = wall_s / 3600.0
    report["accounting"]["a100_hours"] = a100_hours
    report["accounting"]["wall_s"] = wall_s
    report["accounting"]["run_id"] = run_id
    # Persist updated accounting.
    vol_json = out_dir / "morpheus-objective-ablation.json"
    vol_md = out_dir / "morpheus-objective-ablation.md"
    from training.morpheus.objective.ablate import write_report

    write_report(report, vol_json, vol_md)
    if repo_json.parent.is_dir():
        write_report(report, repo_json, repo_md)
    VOLUME.commit()
    return {
        "ok": bool(report["decision"]["pass"]),
        "verdict": report["decision"]["verdict"],
        "selected_candidate": report["decision"]["selected_candidate"],
        "a100_hours": a100_hours,
        "output": str(out_dir),
        "decision": report["decision"],
    }


@app.local_entrypoint()
def ablate_objective(
    config: str = "training/morpheus/configs/objective-ablation.json",
    run_id: str = "",
) -> None:
    cfg_path = Path(config)
    if not cfg_path.is_file():
        cfg_path = REPO / config
    payload = cfg_path.read_text(encoding="utf-8")
    rid = run_id or time.strftime("objective-%Y%m%d-%H%M%S")
    result = ablate_objective_remote.remote(payload, rid)
    print(json.dumps(result, indent=2))
    # Mirror report into the local measurements tree when the remote wrote it
    # only on the volume: local ablation path stays available without Modal.
    local_fallback = REPO / "docs/research/measurements"
    if not (local_fallback / "morpheus-objective-ablation.json").is_file():
        from training.morpheus.objective.ablate import run_ablation

        run_ablation(
            cfg_path,
            a100_hours=float(result.get("a100_hours") or 0.0),
        )
