"""Resumable Modal training loop (Part 14).

Writes an immutable run manifest once, trains with deterministic RNG, publishes
immutable checkpoints, updates the league with training snapshots (not arena
acceptance), and optionally runs deployment-matched calibration.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from arena.records.store import engine_version as current_engine_version
from training.morpheus.self_play.league import (
    League,
    checkpoint_snapshot,
    smoke_league,
)
from training.morpheus.trainer.buffer import (
    BufferCursor,
    ReplayBuffer,
    load_replay_buffer,
)
from training.morpheus.trainer.calibrate import run_deployment_calibration
from training.morpheus.trainer.checkpoint import (
    CheckpointState,
    apply_checkpoint_to_modules,
    build_model_and_optimizer,
    capture_rng_state,
    latest_checkpoint_path,
    list_checkpoints,
    restore_rng_state,
    validate_checkpoint_against_run,
    write_checkpoint,
)
from training.morpheus.trainer.config import TrainRunConfig, load_train_run_config
from training.morpheus.trainer.manifest import (
    EVENT_SNAPSHOT_SAVED,
    RUN_MANIFEST_VERSION,
    RunManifest,
    load_run_manifest,
    write_run_manifest,
)
from training.morpheus.trainer.step import trainer_step

REPO = Path(__file__).resolve().parents[3]
MORPHEUS_BOT = REPO / "bots" / "morpheus"


def _ensure_bot_path() -> None:
    for entry in (REPO, REPO / "bots", MORPHEUS_BOT):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)


def _log(msg: str) -> None:
    print(f"[trainer] {msg}", flush=True)


def _schema_versions() -> tuple[str, str, str]:
    _ensure_bot_path()
    from schema import (  # type: ignore
        ACTION_SCHEMA_VERSION,
        ARCHITECTURE_VERSION,
        TENSOR_SCHEMA_VERSION,
    )

    return TENSOR_SCHEMA_VERSION, ACTION_SCHEMA_VERSION, ARCHITECTURE_VERSION


def _resolve_path(path: str | Path, *, repo_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root / p


@dataclass
class TrainResult:
    ok: bool
    run_id: str
    run_dir: Path
    global_step: int
    last_checkpoint: str | None
    loss_history: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    a100_hours: float = 0.0
    calibration: dict[str, Any] | None = None
    scope: str = ""
    promotable_main_run: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "global_step": self.global_step,
            "last_checkpoint": self.last_checkpoint,
            "loss_history": list(self.loss_history),
            "wall_s": self.wall_s,
            "a100_hours": self.a100_hours,
            "calibration": self.calibration,
            "scope": self.scope,
            "promotable_main_run": self.promotable_main_run,
            "notes": list(self.notes),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def build_run_manifest(
    cfg: TrainRunConfig,
    *,
    run_id: str,
    engine_era: str | None = None,
) -> RunManifest:
    tensor_schema, action_schema, architecture = _schema_versions()
    return RunManifest(
        run_id=run_id,
        schema_version=RUN_MANIFEST_VERSION,
        engine_era=engine_era or current_engine_version(),
        tensor_schema=tensor_schema,
        action_schema=action_schema,
        architecture_version=architecture,
        objective_name=cfg.objective.name,
        objective=cfg.objective.to_dict(),
        scope=cfg.scope,
        promotable_main_run=cfg.promotable_main_run,
        layout=dict(cfg.layout),
        trainer=cfg.trainer_dict(),
        part13_verdict=cfg.part13_verdict,
        part13_fallback=cfg.part13_fallback,
        deployment=dict(cfg.deployment),
        notes=cfg.notes,
    )


def _initial_curriculum_state(cfg: TrainRunConfig) -> dict[str, Any]:
    return {
        "active_classes": sorted(cfg.replay.class_balance.keys()),
        "class_balance": dict(cfg.replay.class_balance),
        "events": [],
    }


def _initial_budget_state(cfg: TrainRunConfig) -> dict[str, Any]:
    return {
        "prior_a100_hours": cfg.budget.prior_a100_hours,
        "main_training_a100_hours_cap": cfg.budget.main_training_a100_hours,
        "deployment_calibration_a100_hours_cap": (
            cfg.budget.deployment_calibration_a100_hours
        ),
        "budget_a100_hours": cfg.budget.budget_a100_hours,
        "main_training_a100_hours_spent": 0.0,
        "deployment_calibration_a100_hours_spent": 0.0,
        "steps": 0,
    }


def _update_league_with_snapshot(
    league: League,
    *,
    checkpoint_path: Path,
    global_step: int,
) -> League:
    """Open a new epoch and register the snapshot as recent (not arena accept)."""
    league.begin_epoch()
    # Replace learner path with the new snapshot digest when present.
    snap_id = f"recent-{global_step:08d}"
    recent = checkpoint_snapshot(
        snapshot_id=snap_id,
        role="recent",
        path=checkpoint_path,
        weight=1.0,
    )
    if snap_id in {s.snapshot_id for s in league.snapshots()}:
        league.replace(recent)
    else:
        # Cap recent slots: keep learner/best/exploiter; add this recent.
        league.register(recent)
    # Refresh learner digest to the new weights when learner is a checkpoint kind.
    try:
        learner = league.learner()
        if learner.kind == "checkpoint":
            updated = checkpoint_snapshot(
                snapshot_id=learner.snapshot_id,
                role="learner",
                path=checkpoint_path,
                weight=learner.weight,
            )
            league.replace(updated)
    except Exception:  # noqa: BLE001
        pass
    league.freeze_epoch()
    return league


def _state_from_live(
    *,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    league: League,
    curriculum: dict[str, Any],
    np_rng: np.random.Generator,
    consumed: BufferCursor,
    budget: dict[str, Any],
    manifest: RunManifest,
    n_blocks: int,
    seed: int,
) -> CheckpointState:
    return CheckpointState(
        global_step=global_step,
        model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        league=league.to_dict(),
        curriculum=dict(curriculum),
        rng=capture_rng_state(np_rng),
        consumed=consumed,
        budget=dict(budget),
        run_compat=manifest.compatibility_key(),
        n_blocks=n_blocks,
        seed=seed,
        meta={"event": EVENT_SNAPSHOT_SAVED},
    )


def _parameter_digest(model: torch.nn.Module) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in sorted(model.state_dict()):
        t = model.state_dict()[key].detach().cpu().contiguous().numpy().tobytes()
        h.update(key.encode("utf-8"))
        h.update(t)
    return h.hexdigest()


def run_training(
    config: TrainRunConfig | Path | str,
    *,
    run_id: str,
    repo_root: Path | None = None,
    device: str | None = None,
    buffer: ReplayBuffer | None = None,
    max_steps: int | None = None,
    resume_checkpoint: Path | None = None,
    run_calibration: bool | None = None,
    engine_era: str | None = None,
) -> TrainResult:
    """Train from scratch or resume. Deterministic under fixed seed + buffer.

    ``engine_era`` pins the competition-module SHA. Pass it from a host that
    has the submodule git metadata (e.g. Modal local entrypoint) when the
    remote image copies ``competition-module`` without ``.git``.
    """
    t_run = time.perf_counter()
    root = Path(repo_root) if repo_root is not None else REPO
    cfg = (
        config
        if isinstance(config, TrainRunConfig)
        else load_train_run_config(_resolve_path(config, repo_root=root))
    )
    if cfg.promotable_main_run and cfg.part13_verdict != "yes":
        raise RuntimeError(
            "refusing promotable_main_run: Part 13 verdict is not yes"
        )

    run_dir = _resolve_path(cfg.run_root, repo_root=root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _log(f"run_dir={run_dir} scope={cfg.scope} promotable={cfg.promotable_main_run}")

    manifest = build_run_manifest(cfg, run_id=run_id, engine_era=engine_era)
    write_run_manifest(manifest, run_dir)

    cuda_available = bool(torch.cuda.is_available())
    if device is None:
        device = "cuda" if cuda_available else "cpu"
    dev = torch.device(device)
    _log(f"device={dev}")

    if buffer is None:
        buffer_dir = _resolve_path(cfg.buffer_dir, repo_root=root)
        _log(f"buffer_load begin dir={buffer_dir}")
        buffer = load_replay_buffer(buffer_dir, window_size=cfg.replay.window_size)
        _log(f"buffer_load done samples={len(buffer)}")

    _log(f"model_init begin n_blocks={cfg.n_blocks}")
    model, optimizer, scheduler = build_model_and_optimizer(
        n_blocks=cfg.n_blocks,
        seed=cfg.seed,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        device=dev,
    )
    _log("model_init done")
    np_rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    league = smoke_league()
    curriculum = _initial_curriculum_state(cfg)
    budget = _initial_budget_state(cfg)
    consumed = BufferCursor(next_index=0, consumed_ids=(), epoch=0)
    global_step = 0
    history: list[dict[str, Any]] = []
    last_ckpt_name: str | None = None

    if resume_checkpoint is not None:
        ckpt_path = Path(resume_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = run_dir / ckpt_path
        state = validate_checkpoint_against_run(ckpt_path, run_dir)
        apply_checkpoint_to_modules(
            state, model=model, optimizer=optimizer, scheduler=scheduler
        )
        restore_rng_state(state.rng, np_rng)
        league = League.from_dict(state.league)
        curriculum = dict(state.curriculum)
        budget = dict(state.budget)
        consumed = state.consumed
        global_step = int(state.global_step)
        last_ckpt_name = ckpt_path.name
        _log(f"resumed step={global_step} checkpoint={last_ckpt_name}")

    steps_target = int(max_steps if max_steps is not None else cfg.cadence.max_steps)
    do_calibrate = (
        cfg.require_deployment_calibration
        if run_calibration is None
        else bool(run_calibration)
    )

    t_train = time.perf_counter()
    _log(f"train_loop begin steps_target={steps_target} batch_size={cfg.batch_size}")
    while global_step < steps_target:
        batch = buffer.sample_batch(
            np_rng,
            batch_size=cfg.batch_size,
            class_balance=cfg.replay.class_balance,
        )
        loss, terms = trainer_step(
            model=model,
            optimizer=optimizer,
            samples=batch,
            objective=cfg.objective,
            device=dev,
            scheduler=scheduler,
        )
        global_step += 1
        consumed = BufferCursor(
            next_index=consumed.next_index + len(batch),
            consumed_ids=consumed.consumed_ids
            + tuple(f"step{global_step}-i{i}" for i in range(len(batch))),
            epoch=consumed.epoch,
        )
        # Keep consumed_ids bounded.
        if len(consumed.consumed_ids) > 256:
            consumed = BufferCursor(
                next_index=consumed.next_index,
                consumed_ids=consumed.consumed_ids[-256:],
                epoch=consumed.epoch,
            )
        term_dict = terms.to_dict()["terms"]
        history.append(
            {
                "step": global_step,
                "loss": loss,
                "terms": term_dict,
            }
        )
        log_every = max(1, cfg.cadence.log_every_steps)
        if global_step % log_every == 0 or global_step == steps_target:
            term_bits = " ".join(
                f"{name}={float(value):.6g}" for name, value in sorted(term_dict.items())
            )
            _log(f"step={global_step} loss={loss:.6g} {term_bits}")
        if global_step % max(1, cfg.cadence.snapshot_every_steps) == 0 or global_step == steps_target:
            train_spent = (time.perf_counter() - t_train) / 3600.0
            budget["main_training_a100_hours_spent"] = float(
                budget.get("main_training_a100_hours_spent") or 0.0
            ) + train_spent
            budget["steps"] = global_step
            t_train = time.perf_counter()
            state = _state_from_live(
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                league=league,
                curriculum=curriculum,
                np_rng=np_rng,
                consumed=consumed,
                budget=budget,
                manifest=manifest,
                n_blocks=cfg.n_blocks,
                seed=cfg.seed,
            )
            ckpt_path = write_checkpoint(
                run_dir, state, event=EVENT_SNAPSHOT_SAVED
            )
            last_ckpt_name = ckpt_path.name
            # League updates happen after the immutable snapshot is published.
            # They never rewrite checkpoint bytes.
            league = _update_league_with_snapshot(
                league, checkpoint_path=ckpt_path, global_step=global_step
            )
            _log(f"snapshot_saved step={global_step} path={ckpt_path.name}")

        if cfg.cadence.stopping_rule == "budget_exhausted":
            spent = float(budget.get("main_training_a100_hours_spent") or 0.0)
            if spent >= cfg.budget.main_training_a100_hours:
                _log("stopping: budget_exhausted")
                break

    calibration_report: dict[str, Any] | None = None
    if do_calibrate and last_ckpt_name is not None:
        _log(f"deployment_calibration begin checkpoint={last_ckpt_name}")
        cal = run_deployment_calibration(
            run_dir=run_dir,
            checkpoint_id=last_ckpt_name,
            deployment=cfg.deployment,
            self_play=cfg.self_play,
            repo_root=root,
            engine_version=manifest.engine_era,
        )
        calibration_report = cal.to_dict()
        budget["deployment_calibration_a100_hours_spent"] = float(
            budget.get("deployment_calibration_a100_hours_spent") or 0.0
        ) + float(cal.a100_hours)
        # Refresh budget file beside the checkpoint for accounting inspect.
        budget_path = run_dir / last_ckpt_name / "budget.json"
        if budget_path.is_file():
            # Do not mutate the immutable checkpoint; write sibling accounting.
            sibling = run_dir / "budget_accounting.json"
            sibling.write_text(
                json.dumps(budget, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        _log(f"deployment_calibration ok={cal.ok} a100_hours={cal.a100_hours:.6f}")

    wall_s = time.perf_counter() - t_run
    a100_hours = wall_s / 3600.0
    ok = last_ckpt_name is not None and (
        not do_calibrate or bool((calibration_report or {}).get("ok"))
    )
    notes = [
        f"scope={cfg.scope}",
        f"part13_verdict={cfg.part13_verdict}",
        f"part13_fallback={cfg.part13_fallback}",
        "Training snapshots use event snapshot_saved (not arena_checkpoint_accept).",
        "Curriculum movement uses event curriculum_class_advance when recorded.",
    ]
    if not cfg.promotable_main_run:
        notes.append(
            "Research-scope run: not eligible for Part 15 freeze until Part 13 is yes "
            "and promotable_main_run is selected."
        )
    result = TrainResult(
        ok=bool(ok),
        run_id=run_id,
        run_dir=run_dir,
        global_step=global_step,
        last_checkpoint=last_ckpt_name,
        loss_history=history,
        wall_s=wall_s,
        a100_hours=a100_hours,
        calibration=calibration_report,
        scope=cfg.scope,
        promotable_main_run=cfg.promotable_main_run,
        notes=notes,
    )
    (run_dir / "train_report.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def resume_training(
    config: TrainRunConfig | Path | str,
    *,
    run_id: str,
    checkpoint_id: str | None = None,
    repo_root: Path | None = None,
    device: str | None = None,
    buffer: ReplayBuffer | None = None,
    max_steps: int | None = None,
    run_calibration: bool | None = None,
) -> TrainResult:
    """Resume from an immutable checkpoint under an existing run directory."""
    root = Path(repo_root) if repo_root is not None else REPO
    cfg = (
        config
        if isinstance(config, TrainRunConfig)
        else load_train_run_config(_resolve_path(config, repo_root=root))
    )
    run_dir = _resolve_path(cfg.run_root, repo_root=root) / run_id
    # Validate existing manifest against a freshly built one.
    stored = load_run_manifest(run_dir)
    presented = build_run_manifest(cfg, run_id=run_id, engine_era=stored.engine_era)
    from training.morpheus.trainer.manifest import assert_resume_compatible

    assert_resume_compatible(stored, presented)

    if checkpoint_id:
        ckpt = run_dir / checkpoint_id
    else:
        found = latest_checkpoint_path(run_dir)
        if found is None:
            raise FileNotFoundError(f"no checkpoint under {run_dir}")
        ckpt = found
    return run_training(
        cfg,
        run_id=run_id,
        repo_root=root,
        device=device,
        buffer=buffer,
        max_steps=max_steps,
        resume_checkpoint=ckpt,
        run_calibration=run_calibration,
        engine_era=stored.engine_era,
    )


def inspect_run(run_dir: Path) -> dict[str, Any]:
    """Summarize an existing run for the Modal inspect_run entry point."""
    run_dir = Path(run_dir)
    manifest = load_run_manifest(run_dir)
    ckpts = [p.name for p in list_checkpoints(run_dir)]
    latest = latest_checkpoint_path(run_dir)
    report_path = run_dir / "train_report.json"
    report = None
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    calibration_dirs = []
    cal_root = run_dir / "calibration"
    if cal_root.is_dir():
        calibration_dirs = sorted(p.name for p in cal_root.iterdir() if p.is_dir())
    return {
        "run_id": manifest.run_id,
        "scope": manifest.scope,
        "promotable_main_run": manifest.promotable_main_run,
        "part13_verdict": manifest.part13_verdict,
        "part13_fallback": manifest.part13_fallback,
        "engine_era": manifest.engine_era,
        "tensor_schema": manifest.tensor_schema,
        "action_schema": manifest.action_schema,
        "objective_name": manifest.objective_name,
        "checkpoints": ckpts,
        "latest_checkpoint": None if latest is None else latest.name,
        "calibration_checkpoints": calibration_dirs,
        "train_report": report,
        "event_vocabulary": dict(manifest.event_vocabulary),
        "layout": dict(manifest.layout),
        "budget": dict(manifest.trainer.get("budget") or {}),
    }


# Exported for tests that compare interrupted vs continuous runs.
parameter_digest = _parameter_digest
