"""Part 14 checkpoint resume: interrupted and continuous runs match."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.morpheus

from training.morpheus.objective.config import load_pilot_objective_bundle
from training.morpheus.objective.targets import build_seat_targets
from training.morpheus.self_play.schema import SparsePolicy
from training.morpheus.trainer.buffer import ReplayBuffer, write_sample
from training.morpheus.trainer.checkpoint import (
    CheckpointError,
    apply_checkpoint_to_modules,
    build_model_and_optimizer,
    latest_checkpoint_path,
    list_checkpoints,
    load_checkpoint,
    restore_rng_state,
)
from training.morpheus.trainer.config import (
    BudgetConfig,
    CadenceConfig,
    OptimizerConfig,
    ReplayConfig,
    SchedulerConfig,
    TrainRunConfig,
)
from training.morpheus.trainer.loop import (
    parameter_digest,
    resume_training,
    run_training,
)
from training.morpheus.trainer.manifest import (
    EVENT_ARENA_CHECKPOINT_ACCEPT,
    EVENT_CURRICULUM_CLASS_ADVANCE,
    EVENT_SNAPSHOT_SAVED,
    ManifestError,
    load_run_manifest,
)
from training.morpheus.trainer.sample import TrainSample
import torch

REPO = Path(__file__).resolve().parents[3]
OBJECTIVE = REPO / "training/morpheus/configs/pilot-objective.json"


def _tiny_sample(*, item_id: str = "tiny-0") -> TrainSample:
    H = W = 4
    ownership = np.zeros((2, H, W), dtype=bool)
    ownership[0, 0, 0] = True
    ownership[1, 3, 3] = True
    armies = np.zeros((H, W), dtype=np.int32)
    armies[0, 0] = 5
    armies[3, 3] = 8
    generals = np.zeros((H, W), dtype=bool)
    generals[0, 0] = True
    generals[3, 3] = True
    castles = np.zeros((H, W), dtype=bool)
    targets = build_seat_targets(
        winner="a",
        seat=0,
        policy=SparsePolicy(indices=(3969,), probs=(1.0,)),
        ownership=ownership,
        armies=armies,
        generals=generals,
        castles=castles,
        final_land=(2, 1),
        final_army=(5, 8),
        final_castles=(0, 0),
        current_turn=10,
        terminal_turn=40,
    )
    pad = targets.board_mask.shape[0]
    tensor = np.zeros((49, pad, pad), dtype=np.float32)
    tensor[0] = targets.board_mask
    legal = (targets.policy > 0).astype(bool)
    legal[3969] = True
    return TrainSample(
        item_id=item_id,
        sample_seat=0,
        tensor=tensor,
        legal_mask=legal,
        targets=targets,
        source_label="synthetic",
        outcome="a",
    )


def _cfg(
    tmp_path: Path,
    *,
    log_every_steps: int = 1,
    snapshot_every_steps: int = 1,
    max_steps: int = 4,
) -> TrainRunConfig:
    bundle = load_pilot_objective_bundle(OBJECTIVE)
    return TrainRunConfig(
        name="resume-unit",
        scope="narrower_non_promotable_research_scope",
        promotable_main_run=False,
        part13_verdict="no",
        part13_fallback="narrower_non_promotable_research_scope",
        seed=0,
        batch_size=1,
        n_blocks=2,
        objective=bundle["training"],
        rated_objective=bundle["rated"],
        optimizer=OptimizerConfig(
            name="adamw",
            learning_rate=1e-2,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
        ),
        scheduler=SchedulerConfig(name="constant", step_size=10_000, gamma=1.0),
        replay=ReplayConfig(
            window_size=8,
            class_balance={"1": 1.0},
            augment_symmetries=False,
        ),
        cadence=CadenceConfig(
            snapshot_every_steps=snapshot_every_steps,
            log_every_steps=log_every_steps,
            max_steps=max_steps,
            games_per_checkpoint=1,
            checkpoint_count=1,
            stopping_rule="max_steps",
        ),
        budget=BudgetConfig(
            main_training_a100_hours=1.0,
            deployment_calibration_a100_hours=0.1,
            prior_a100_hours=0.0,
            budget_a100_hours=48.0,
        ),
        layout={
            "backend": "cpu",
            "physical_cores_per_game": 1,
            "seat_search": "sequential",
            "workers_per_a100": 16,
        },
        deployment={
            "n_particles": 4,
            "target_simulations": 4,
            "min_simulations": 2,
            "search_depth": 2,
            "pending_leaf_batch": 2,
            "normal_deadline_ms": 60_000.0,
        },
        self_play={
            "seed": 0,
            "games": 1,
            "max_turns": 2,
            "league_weight": 1.0,
            "panel_weight": 0.0,
            "panel_members": [{"bot_id": "smoke", "role": "heuristic", "weight": 1.0}],
            "n_particles": 2,
            "target_simulations": 2,
            "min_simulations": 1,
            "search_depth": 2,
            "pending_leaf_batch": 1,
            "deadline_ms": 60_000.0,
            "seat_search": "sequential",
            "output": "data/morpheus/self_play/unit",
        },
        buffer_dir=str(tmp_path / "buf"),
        run_root=str(tmp_path / "runs"),
        require_deployment_calibration=False,
        notes=("resume unit",),
    )


def _buffer(tmp_path: Path) -> ReplayBuffer:
    sample = _tiny_sample()
    write_sample(tmp_path / "buf", sample, sample_id="tiny-0", class_id="1")
    return ReplayBuffer(
        samples=[sample],
        sample_ids=["tiny-0"],
        class_ids=["1"],
        window_size=8,
    )


def test_interrupt_resume_matches_continuous(tmp_path: Path):
    cfg = _cfg(tmp_path)
    buf = _buffer(tmp_path)

    continuous = run_training(
        cfg,
        run_id="continuous",
        repo_root=REPO,
        device="cpu",
        buffer=buf,
        max_steps=4,
        run_calibration=False,
    )
    assert continuous.ok
    cont_ckpt = latest_checkpoint_path(continuous.run_dir)
    assert cont_ckpt is not None
    cont_state = load_checkpoint(cont_ckpt)

    interrupted = run_training(
        cfg,
        run_id="interrupted",
        repo_root=REPO,
        device="cpu",
        buffer=buf,
        max_steps=2,
        run_calibration=False,
    )
    assert interrupted.global_step == 2
    mid = latest_checkpoint_path(interrupted.run_dir)
    assert mid is not None

    resumed = resume_training(
        cfg,
        run_id="interrupted",
        checkpoint_id=mid.name,
        repo_root=REPO,
        device="cpu",
        buffer=buf,
        max_steps=4,
        run_calibration=False,
    )
    assert resumed.global_step == 4
    res_ckpt = latest_checkpoint_path(resumed.run_dir)
    assert res_ckpt is not None
    res_state = load_checkpoint(res_ckpt)

    # Same final step identity and model content digest.
    assert cont_state.global_step == res_state.global_step == 4
    assert cont_state.content_digest() == res_state.content_digest()

    # Next-update parity: reload mid checkpoint, take one step, match continuous step-3.
    mid_state = load_checkpoint(mid)
    model, opt, sched = build_model_and_optimizer(
        n_blocks=cfg.n_blocks,
        seed=cfg.seed,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        device=torch.device("cpu"),
    )
    apply_checkpoint_to_modules(mid_state, model=model, optimizer=opt, scheduler=sched)
    rng = np.random.default_rng(0)
    restore_rng_state(mid_state.rng, rng)
    from training.morpheus.trainer.step import trainer_step

    batch = buf.sample_batch(rng, batch_size=1, class_balance=cfg.replay.class_balance)
    loss_resume, _ = trainer_step(
        model=model,
        optimizer=opt,
        samples=batch,
        objective=cfg.objective,
        device=torch.device("cpu"),
        scheduler=sched,
    )
    digest_resume = parameter_digest(model)

    # Continuous history loss at step 3 must match the resumed next update loss.
    cont_step3 = [h for h in continuous.loss_history if h["step"] == 3][0]
    res_step3 = [h for h in resumed.loss_history if h["step"] == 3][0]
    assert abs(cont_step3["loss"] - res_step3["loss"]) < 1e-6
    assert abs(loss_resume - cont_step3["loss"]) < 1e-6
    assert digest_resume == parameter_digest(
        # Rebuild continuous model at step 3 via its checkpoint list.
        _model_at_step(continuous.run_dir, 3, cfg)
    )


def test_train_loss_logged_every_log_every_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    cfg = _cfg(tmp_path, log_every_steps=2, snapshot_every_steps=4, max_steps=4)
    buf = _buffer(tmp_path)
    result = run_training(
        cfg,
        run_id="loss-log",
        repo_root=REPO,
        device="cpu",
        buffer=buf,
        max_steps=4,
        run_calibration=False,
    )
    assert result.ok
    assert len(result.loss_history) == 4
    out = capsys.readouterr().out
    loss_lines = [
        line
        for line in out.splitlines()
        if line.startswith("[trainer] step=") and " loss=" in line
    ]
    assert len(loss_lines) == 2
    assert "step=2 loss=" in loss_lines[0]
    assert "step=4 loss=" in loss_lines[1]
    assert "policy=" in loss_lines[0]
    assert not any("step=1 loss=" in line for line in loss_lines)
    assert not any("step=3 loss=" in line for line in loss_lines)


def _model_at_step(run_dir: Path, step: int, cfg: TrainRunConfig):
    matches = [p for p in list_checkpoints(run_dir) if f"ckpt-{step:08d}-" in p.name]
    assert matches, f"no checkpoint for step {step} under {run_dir}"
    state = load_checkpoint(matches[0])
    model, opt, sched = build_model_and_optimizer(
        n_blocks=cfg.n_blocks,
        seed=cfg.seed,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        device=torch.device("cpu"),
    )
    apply_checkpoint_to_modules(state, model=model, optimizer=opt, scheduler=sched)
    return model


def test_checkpoint_names_are_immutable(tmp_path: Path):
    cfg = _cfg(tmp_path)
    buf = _buffer(tmp_path)
    result = run_training(
        cfg,
        run_id="immut",
        repo_root=REPO,
        device="cpu",
        buffer=buf,
        max_steps=2,
        run_calibration=False,
    )
    paths = list_checkpoints(result.run_dir)
    assert len(paths) == 2
    names = {p.name for p in paths}
    # Re-running write of the same step must not create a mutable "latest" weight file.
    assert all(n.startswith("ckpt-") for n in names)
    meta = json.loads((paths[0] / "meta.json").read_text(encoding="utf-8"))
    assert meta["event"] == EVENT_SNAPSHOT_SAVED
    assert meta["event"] != EVENT_ARENA_CHECKPOINT_ACCEPT
    assert meta["event"] != EVENT_CURRICULUM_CLASS_ADVANCE

    manifest = load_run_manifest(result.run_dir)
    assert manifest.event_vocabulary["training_snapshot"] == EVENT_SNAPSHOT_SAVED
    assert (
        manifest.event_vocabulary["arena_acceptance"] == EVENT_ARENA_CHECKPOINT_ACCEPT
    )
    assert (
        manifest.event_vocabulary["curriculum_movement"]
        == EVENT_CURRICULUM_CLASS_ADVANCE
    )

    # Manifest is immutable.
    with pytest.raises(ManifestError):
        from training.morpheus.trainer.manifest import RunManifest, write_run_manifest

        mutated = RunManifest.from_dict({**manifest.to_dict(), "run_id": "other"})
        write_run_manifest(mutated, result.run_dir)


def test_resume_rejects_schema_drift(tmp_path: Path):
    cfg = _cfg(tmp_path)
    buf = _buffer(tmp_path)
    result = run_training(
        cfg,
        run_id="drift",
        repo_root=REPO,
        device="cpu",
        buffer=buf,
        max_steps=1,
        run_calibration=False,
    )
    ckpt = latest_checkpoint_path(result.run_dir)
    assert ckpt is not None
    meta_path = ckpt / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["run_compat"]["tensor_schema"] = "morpheus-tensor-v-WRONG"
    # Do not mutate the published checkpoint in place for the product path;
    # simulate a drifted copy.
    drift_dir = result.run_dir / "ckpt-drifted"
    import shutil

    shutil.copytree(ckpt, drift_dir)
    (drift_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from training.morpheus.trainer.checkpoint import validate_checkpoint_against_run

    with pytest.raises(CheckpointError):
        validate_checkpoint_against_run(drift_dir, result.run_dir)
