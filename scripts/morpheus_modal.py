#!/usr/bin/env python3
"""Unified Modal entry points for Morpheus training gates.

Part 12:
    modal run scripts/morpheus_modal.py::ablate_objective \\
      --config training/morpheus/configs/objective-ablation.json

Part 13:
    modal run scripts/morpheus_modal.py::qualify_compute \\
      --config training/morpheus/configs/modal-qualification.json

Part 14:
    modal run scripts/morpheus_modal.py::train \\
      --config training/morpheus/configs/promotable-run.json
    modal run scripts/morpheus_modal.py::resume \\
      --config training/morpheus/configs/promotable-run.json --run-id <run_id>
    modal run scripts/morpheus_modal.py::inspect_run --run-id <run_id>
    modal run scripts/morpheus_modal.py::download \\
      --run-id <run_id> --checkpoint <checkpoint_id> --dest /tmp/ckpt

A100 time from ``ablate_objective``, compute qualification, and Part 14
``train`` / ``resume`` charges to Part 13 accounting notes.
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
    image=IMAGE,
    gpu="A100-80GB",
    timeout=60 * 60 * 6,
    volumes={"/vol": VOLUME},
)
def train_remote(config_json: str, run_id: str) -> dict:
    """Part 14 training loop. A100 wall charges to main_training."""
    from training.morpheus.trainer.config import load_train_run_config
    from training.morpheus.trainer.loop import run_training

    cfg_path = Path("/tmp/promotable-run.json")
    cfg_path.write_text(config_json, encoding="utf-8")
    cfg = load_train_run_config(cfg_path)
    # Force volume-backed run root on Modal.
    from dataclasses import replace

    cfg = replace(cfg, run_root="/vol/morpheus/runs", buffer_dir="/vol/morpheus/buffer")
    import torch as _torch

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(
        f"[train_remote] run_id={run_id} device={device} scope={cfg.scope} "
        f"promotable={cfg.promotable_main_run}",
        flush=True,
    )
    result = run_training(
        cfg,
        run_id=run_id,
        repo_root=Path("/root"),
        device=device,
    )
    VOLUME.commit()
    return {
        "ok": result.ok,
        "run_id": result.run_id,
        "global_step": result.global_step,
        "last_checkpoint": result.last_checkpoint,
        "a100_hours": result.a100_hours,
        "scope": result.scope,
        "promotable_main_run": result.promotable_main_run,
        "run_dir": str(result.run_dir),
        "calibration_ok": None
        if result.calibration is None
        else bool(result.calibration.get("ok")),
    }


@app.local_entrypoint()
def train(
    config: str = "training/morpheus/configs/promotable-run.json",
    run_id: str = "",
    local: bool = False,
) -> None:
    """Part 14 resumable trainer (Modal A100 or local CPU)."""
    from training.morpheus.trainer.config import load_train_run_config
    from training.morpheus.trainer.loop import run_training

    cfg_path = Path(config)
    if not cfg_path.is_file():
        cfg_path = REPO / config
    rid = run_id or time.strftime("train-%Y%m%d-%H%M%S")
    cfg = load_train_run_config(cfg_path)
    print(
        f"[train] entry local={local} run_id={rid} scope={cfg.scope} "
        f"promotable={cfg.promotable_main_run} part13={cfg.part13_verdict}",
        flush=True,
    )
    if local:
        result = run_training(cfg, run_id=rid, repo_root=REPO, device="cpu")
        summary = result.to_dict()
    else:
        summary = train_remote.remote(cfg_path.read_text(encoding="utf-8"), rid)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"ok={summary.get('ok')}", flush=True)


@app.function(
    image=IMAGE,
    gpu="A100-80GB",
    timeout=60 * 60 * 6,
    volumes={"/vol": VOLUME},
)
def resume_remote(
    config_json: str, run_id: str, checkpoint_id: str = ""
) -> dict:
    """Resume Part 14 training from an immutable checkpoint on the volume."""
    from dataclasses import replace

    from training.morpheus.trainer.config import load_train_run_config
    from training.morpheus.trainer.loop import resume_training

    cfg_path = Path("/tmp/promotable-run.json")
    cfg_path.write_text(config_json, encoding="utf-8")
    cfg = load_train_run_config(cfg_path)
    cfg = replace(cfg, run_root="/vol/morpheus/runs", buffer_dir="/vol/morpheus/buffer")
    import torch as _torch

    device = "cuda" if _torch.cuda.is_available() else "cpu"
    result = resume_training(
        cfg,
        run_id=run_id,
        checkpoint_id=(checkpoint_id or None),
        repo_root=Path("/root"),
        device=device,
    )
    VOLUME.commit()
    return {
        "ok": result.ok,
        "run_id": result.run_id,
        "global_step": result.global_step,
        "last_checkpoint": result.last_checkpoint,
        "a100_hours": result.a100_hours,
        "run_dir": str(result.run_dir),
    }


@app.local_entrypoint()
def resume(
    config: str = "training/morpheus/configs/promotable-run.json",
    run_id: str = "",
    checkpoint: str = "",
    local: bool = False,
) -> None:
    """Resume a Part 14 run from the latest or named checkpoint."""
    from training.morpheus.trainer.config import load_train_run_config
    from training.morpheus.trainer.loop import resume_training

    if not run_id:
        raise SystemExit("--run-id is required for resume")
    cfg_path = Path(config)
    if not cfg_path.is_file():
        cfg_path = REPO / config
    cfg = load_train_run_config(cfg_path)
    if local:
        result = resume_training(
            cfg,
            run_id=run_id,
            checkpoint_id=(checkpoint or None),
            repo_root=REPO,
            device="cpu",
        )
        summary = result.to_dict()
    else:
        summary = resume_remote.remote(
            cfg_path.read_text(encoding="utf-8"), run_id, checkpoint
        )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"ok={summary.get('ok')}", flush=True)


@app.function(
    image=IMAGE,
    cpu=2,
    memory=4096,
    timeout=60 * 10,
    volumes={"/vol": VOLUME},
)
def inspect_run_remote(run_id: str) -> dict:
    from training.morpheus.trainer.report import write_inspect_report

    run_dir = Path("/vol/morpheus/runs") / run_id
    report = write_inspect_report(run_dir)
    VOLUME.commit()
    return report


@app.local_entrypoint()
def inspect_run(run_id: str = "", local_dir: str = "") -> None:
    """Inspect an immutable run manifest and checkpoint list."""
    from training.morpheus.trainer.report import write_inspect_report

    if local_dir:
        report = write_inspect_report(Path(local_dir))
    else:
        if not run_id:
            raise SystemExit("--run-id or --local-dir is required")
        report = inspect_run_remote.remote(run_id)
    print(json.dumps(report, indent=2), flush=True)


@app.function(
    image=IMAGE,
    cpu=2,
    memory=8192,
    timeout=60 * 30,
    volumes={"/vol": VOLUME},
)
def download_remote(run_id: str, checkpoint_id: str) -> dict:
    """Read checkpoint bytes from the volume and return a file map for local write."""
    src = Path("/vol/morpheus/runs") / run_id / checkpoint_id
    if not src.is_dir():
        raise FileNotFoundError(f"checkpoint missing: {src}")
    files: dict[str, bytes] = {}
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = path.relative_to(src).as_posix()
        files[rel] = path.read_bytes()
    return {
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "files": {k: v.hex() for k, v in files.items()},
        "file_count": len(files),
    }


@app.local_entrypoint()
def download(
    run_id: str = "",
    checkpoint: str = "",
    dest: str = "",
    local_dir: str = "",
) -> None:
    """Download one immutable checkpoint to a local directory."""
    if not dest:
        raise SystemExit("--dest is required")
    dest_path = Path(dest)
    if dest_path.exists():
        raise SystemExit(f"refusing to overwrite {dest_path}")
    if local_dir:
        from training.morpheus.trainer.report import download_checkpoint

        if not checkpoint:
            raise SystemExit("--checkpoint is required with --local-dir")
        out = download_checkpoint(Path(local_dir), checkpoint, dest_path)
        print(json.dumps({"ok": True, "path": str(out)}, indent=2), flush=True)
        return
    if not run_id or not checkpoint:
        raise SystemExit("--run-id and --checkpoint are required")
    payload = download_remote.remote(run_id, checkpoint)
    dest_path.mkdir(parents=True)
    for rel, hex_bytes in payload["files"].items():
        target = dest_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes.fromhex(hex_bytes))
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(dest_path),
                "file_count": payload["file_count"],
                "run_id": run_id,
                "checkpoint_id": checkpoint,
            },
            indent=2,
        ),
        flush=True,
    )
