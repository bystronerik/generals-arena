"""Part 14 trainer step: overfit a tiny deterministic shard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.morpheus

from training.morpheus.objective.config import load_pilot_objective_bundle
from training.morpheus.objective.targets import build_seat_targets
from training.morpheus.self_play.schema import SparsePolicy
from training.morpheus.trainer.buffer import ReplayBuffer, write_sample
from training.morpheus.trainer.checkpoint import build_model_and_optimizer
from training.morpheus.trainer.config import (
    BudgetConfig,
    CadenceConfig,
    OptimizerConfig,
    ReplayConfig,
    SchedulerConfig,
    TrainRunConfig,
)
from training.morpheus.trainer.sample import TrainSample
from training.morpheus.trainer.step import trainer_step

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


def _research_config(*, run_root: Path, buffer_dir: Path) -> TrainRunConfig:
    bundle = load_pilot_objective_bundle(OBJECTIVE)
    return TrainRunConfig(
        name="unit-research",
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
            window_size=16,
            class_balance={"1": 1.0},
            augment_symmetries=False,
        ),
        cadence=CadenceConfig(
            snapshot_every_steps=1,
            log_every_steps=1,
            max_steps=8,
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
            "max_turns": 4,
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
        buffer_dir=str(buffer_dir),
        run_root=str(run_root),
        require_deployment_calibration=False,
        notes=("unit test config",),
    )


def test_trainer_step_overfits_tiny_shard(tmp_path: Path):
    cfg = _research_config(run_root=tmp_path / "runs", buffer_dir=tmp_path / "buf")
    sample = _tiny_sample()
    write_sample(tmp_path / "buf", sample, sample_id="tiny-0", class_id="1")
    model, optimizer, scheduler = build_model_and_optimizer(
        n_blocks=cfg.n_blocks,
        seed=cfg.seed,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        device=torch.device("cpu"),
    )
    losses = []
    for _ in range(20):
        loss, _ = trainer_step(
            model=model,
            optimizer=optimizer,
            samples=[sample],
            objective=cfg.objective,
            device=torch.device("cpu"),
            scheduler=scheduler,
        )
        losses.append(loss)
    assert np.isfinite(losses[0]) and np.isfinite(losses[-1])
    assert losses[-1] < losses[0]


def test_buffer_sample_round_trip(tmp_path: Path):
    from training.morpheus.trainer.buffer import read_sample

    sample = _tiny_sample()
    path = write_sample(tmp_path / "buf", sample, sample_id="tiny-0", class_id="1")
    loaded, sid, cid = read_sample(path)
    assert sid == "tiny-0"
    assert cid == "1"
    assert loaded.item_id == sample.item_id
    assert float(loaded.targets.value) == float(sample.targets.value)
    assert loaded.tensor.shape == sample.tensor.shape


def test_promotable_run_config_loads_research_scope():
    path = REPO / "training/morpheus/configs/promotable-run.json"
    from training.morpheus.trainer.config import load_train_run_config

    cfg = load_train_run_config(path)
    assert cfg.promotable_main_run is False
    assert cfg.part13_verdict == "no"
    assert cfg.scope == "narrower_non_promotable_research_scope"
    assert cfg.optimizer.name == "adamw"
    assert cfg.cadence.games_per_checkpoint == 54484
    assert cfg.cadence.log_every_steps == 10


def test_scraped_classes13_run_config_loads_log_every_steps():
    path = REPO / "training/morpheus/configs/scraped-classes13-run.json"
    from training.morpheus.trainer.config import load_train_run_config

    cfg = load_train_run_config(path)
    assert cfg.cadence.log_every_steps == 50
    assert cfg.cadence.snapshot_every_steps == 200
