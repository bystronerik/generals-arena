#!/usr/bin/env python3
"""Unified Modal entry points for Morpheus training gates.

Part 12:
    modal run scripts/morpheus_modal.py::ablate_objective \\
      --config training/morpheus/configs/objective-ablation.json

Part 13:
    modal run scripts/morpheus_modal.py::qualify_compute \\
      --config training/morpheus/configs/modal-qualification.json

Pilot class-1 learn smoke:
    modal run scripts/morpheus_modal.py::pilot_learn \\
      --config training/morpheus/configs/pilot-class1-learn.json

Later parts add train / resume / inspect on this app. A100 time from
``ablate_objective``, compute qualification, and ``pilot_learn`` charges to
Part 13 accounting notes.
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
        "jax==0.11.0",
        "jaxlib==0.11.0",
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
    .add_local_dir(
        str(REPO / "scripts" / "configs"),
        remote_path="/root/scripts/configs",
        copy=True,
    )
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root"})
)

# Hybrid probe image. jax[cuda12]==0.11.0 wants nvidia-cudnn>=9.8 while
# torch==2.6.0 pins 9.1.0.70, so keep CPU jax here (same pins as IMAGE).
# Engine transitions still run; CUDA JAX can return after a compatible pin set.
GPU_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.4.6",
        "torch==2.6.0",
        "jax==0.11.0",
        "jaxlib==0.11.0",
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
    .add_local_dir(
        str(REPO / "scripts" / "configs"),
        remote_path="/root/scripts/configs",
        copy=True,
    )
    .run_commands("pip install -e /root/competition-module --no-deps")
    .env({"PYTHONPATH": "/root"})
)

# Trajectories are gitignored; bake local pilot shards into the image when present.
# Use CPU IMAGE (same as ablate_objective) to avoid jax[cuda12]/torch CUDA pin conflicts.
_PILOT_TRAJ = REPO / "data" / "trajectories"
PILOT_IMAGE = (
    IMAGE.add_local_dir(
        str(_PILOT_TRAJ),
        remote_path="/root/data/trajectories",
        copy=True,
    )
    if _PILOT_TRAJ.is_dir()
    else IMAGE
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


@app.function(
    image=IMAGE,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    volumes={"/vol": VOLUME},
)
def measure_layout_remote(
    config_json: str, layout_json: str, run_id: str, engine_version: str = ""
) -> dict:
    """CPU throughput probe for one Part 13 layout."""
    from training.morpheus.compute.measure import measure_layout
    from training.morpheus.self_play.driver import DriverConfig

    cfg = json.loads(config_json)
    layout = json.loads(layout_json)
    driver = DriverConfig.from_dict(cfg.get("self_play") or {})
    name = str(layout.get("name") or "layout")
    out = Path("/vol/morpheus/compute_gate") / run_id / name
    t0 = time.perf_counter()
    row = measure_layout(
        layout,
        base_config=driver,
        output=out,
        repo_root=Path("/root"),
        engine_version=engine_version or None,
    )
    row["remote_wall_s"] = time.perf_counter() - t0
    VOLUME.commit()
    return row


@app.function(
    image=GPU_IMAGE,
    gpu="A100-80GB",
    timeout=60 * 30,
    volumes={"/vol": VOLUME},
)
def measure_hybrid_gpu_remote(
    config_json: str, layout_json: str, run_id: str, engine_version: str = ""
) -> dict:
    """Hybrid probe: full-stack games on an A100 host (JAX may use the GPU)."""
    from training.morpheus.compute.measure import measure_layout
    from training.morpheus.self_play.driver import DriverConfig

    cfg = json.loads(config_json)
    layout = json.loads(layout_json)
    layout = {**layout, "backend": "hybrid"}
    driver = DriverConfig.from_dict(cfg.get("self_play") or {})
    name = str(layout.get("name") or "hybrid")
    out = Path("/vol/morpheus/compute_gate") / run_id / name
    t0 = time.perf_counter()
    row = measure_layout(
        layout,
        base_config=driver,
        output=out,
        repo_root=Path("/root"),
        engine_version=engine_version or None,
    )
    wall_s = time.perf_counter() - t0
    row["remote_wall_s"] = wall_s
    row["a100_hours"] = wall_s / 3600.0
    VOLUME.commit()
    return row


@app.local_entrypoint()
def qualify_compute(
    config: str = "training/morpheus/configs/modal-qualification.json",
    run_id: str = "",
    local: bool = False,
) -> None:
    """Part 13 Modal compute gate. Writes morpheus-modal-qualification.{json,md}."""
    from training.morpheus.compute.qualify import run_qualification

    cfg_path = Path(config)
    if not cfg_path.is_file():
        cfg_path = REPO / config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    rid = run_id or time.strftime("qualify-%Y%m%d-%H%M%S")
    payload = cfg_path.read_text(encoding="utf-8")

    from arena.records.store import engine_version as current_engine_version

    engine_sha = current_engine_version()

    schedule = dict(cfg.get("schedule") or {})
    learning_curve_h = float(schedule.get("learning_curve_pilot_a100_hours") or 0.0)
    gpu_self_play_h = float(schedule.get("gpu_self_play_a100_hours") or 0.0)

    layout_results: list[dict] = []
    measured_gpu_h = 0.0

    if local:
        report = run_qualification(
            cfg_path,
            learning_curve_a100_hours=learning_curve_h,
            gpu_self_play_a100_hours=gpu_self_play_h,
        )
    else:
        hybrid_layouts = [
            lay
            for lay in (cfg.get("layouts") or [])
            if str(lay.get("backend") or "") == "hybrid"
        ]
        cpu_layouts = [
            lay
            for lay in (cfg.get("layouts") or [])
            if str(lay.get("backend") or "cpu") != "hybrid"
        ]
        # Fan out CPU layout probes; charge hybrid A100 wall separately.
        cpu_calls = [
            measure_layout_remote.spawn(payload, json.dumps(lay), rid, engine_sha)
            for lay in cpu_layouts
        ]
        hybrid_calls = [
            measure_hybrid_gpu_remote.spawn(payload, json.dumps(lay), rid, engine_sha)
            for lay in hybrid_layouts
        ]
        for call in cpu_calls:
            layout_results.append(call.get())
        for call in hybrid_calls:
            row = call.get()
            measured_gpu_h += float(row.get("a100_hours") or 0.0)
            layout_results.append(row)

        # Hybrid A100 wall time charges to throughput_qualification only.
        report = run_qualification(
            cfg_path,
            layout_results=layout_results,
            throughput_a100_hours=measured_gpu_h,
            learning_curve_a100_hours=learning_curve_h,
            gpu_self_play_a100_hours=gpu_self_play_h,
        )

    summary = {
        "verdict": report["decision"]["verdict"],
        "selected_layout": (report.get("selected_layout") or {}).get("name"),
        "backend": (report.get("selected_layout") or {}).get("backend"),
        "workers_per_a100": report.get("workers_per_a100"),
        "games_per_hour": (report.get("rates") or {}).get("aggregate_games_per_hour"),
        "positions_per_hour": (report.get("rates") or {}).get(
            "positions_supply_per_hour"
        ),
        "games_per_checkpoint": (report.get("rates") or {}).get("games_per_checkpoint"),
        "checkpoint_count": (report.get("rates") or {}).get("checkpoint_count"),
        "a100_total": (report.get("a100_accounting") or {}).get("total_a100_hours"),
        "fallback": (report["decision"].get("fallback") or {}).get("selected"),
        "run_id": rid,
        "local": local,
    }
    print(json.dumps(summary, indent=2))
    print(f"verdict={summary['verdict']}")


@app.function(
    image=PILOT_IMAGE,
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/vol": VOLUME},
)
def pilot_learn_remote(config_json: str, run_id: str) -> dict:
    """Thin class-1 learning smoke; A100 wall charges to learning_curve_pilot."""
    from training.morpheus.pilot.report import write_pilot_report
    from training.morpheus.pilot.run import run_pilot_learn

    print(f"[pilot_learn_remote] start run_id={run_id}", flush=True)
    cfg_path = Path("/tmp/pilot-class1-learn.json")
    cfg_path.write_text(config_json, encoding="utf-8")
    out_dir = Path("/vol/morpheus/pilot") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pilot_learn_remote] output_dir={out_dir}", flush=True)
    import torch as _torch

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(
        f"[pilot_learn_remote] torch_cuda={_torch.cuda.is_available()} "
        f"device={device} device_count={_torch.cuda.device_count()}",
        flush=True,
    )
    report = run_pilot_learn(
        cfg_path,
        output_dir=out_dir,
        repo_root=Path("/root"),
        device=device,
    )
    print(
        f"[pilot_learn_remote] write reports ok={report.get('ok')} "
        f"wall_s={report.get('wall_s')}",
        flush=True,
    )
    vol_json = out_dir / "morpheus-pilot-class1-learn.json"
    vol_md = out_dir / "morpheus-pilot-class1-learn.md"
    write_pilot_report(report, json_path=vol_json, md_path=vol_md)
    repo_meas = Path("/root/docs/research/measurements")
    if repo_meas.is_dir():
        write_pilot_report(
            report,
            json_path=repo_meas / "morpheus-pilot-class1-learn.json",
            md_path=repo_meas / "morpheus-pilot-class1-learn.md",
        )
    VOLUME.commit()
    print(f"[pilot_learn_remote] done run_id={run_id}", flush=True)
    return {
        "ok": bool(report.get("ok")),
        "loss_step0": report.get("loss_step0"),
        "loss_final": report.get("loss_final"),
        "loss_decreased": report.get("loss_decreased"),
        "checkpoint_reloadable": report.get("checkpoint_reloadable"),
        "a100_hours": report.get("a100_hours"),
        "output": str(out_dir),
        "report": report,
    }


@app.local_entrypoint()
def pilot_learn(
    config: str = "training/morpheus/configs/pilot-class1-learn.json",
    run_id: str = "",
    local: bool = False,
) -> None:
    """Scraped class-1 thin learning smoke (Modal A100 or local CPU)."""
    from training.morpheus.pilot.report import write_pilot_report
    from training.morpheus.pilot.run import run_pilot_learn

    cfg_path = Path(config)
    if not cfg_path.is_file():
        cfg_path = REPO / config
    rid = run_id or time.strftime("pilot-%Y%m%d-%H%M%S")
    local_meas = REPO / "docs/research/measurements"
    local_json = local_meas / "morpheus-pilot-class1-learn.json"
    local_md = local_meas / "morpheus-pilot-class1-learn.md"
    print(
        f"[pilot_learn] entry local={local} run_id={rid} config={cfg_path}",
        flush=True,
    )

    if local:
        out = REPO / "data" / "morpheus" / "pilot" / rid
        report = run_pilot_learn(cfg_path, output_dir=out, repo_root=REPO)
        write_pilot_report(report, json_path=local_json, md_path=local_md)
        summary = {
            "ok": report.get("ok"),
            "loss_step0": report.get("loss_step0"),
            "loss_final": report.get("loss_final"),
            "a100_hours": report.get("a100_hours"),
            "run_id": rid,
            "local": True,
            "checkpoint": report.get("checkpoint"),
        }
    else:
        print("[pilot_learn] dispatching pilot_learn_remote on Modal", flush=True)
        result = pilot_learn_remote.remote(cfg_path.read_text(encoding="utf-8"), rid)
        report = result.get("report") or {}
        if report:
            write_pilot_report(report, json_path=local_json, md_path=local_md)
        summary = {k: v for k, v in result.items() if k != "report"}
        summary["run_id"] = rid
        summary["local"] = False

    print(json.dumps(summary, indent=2), flush=True)
    print(f"ok={summary.get('ok')}", flush=True)
