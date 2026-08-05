"""Fail-closed Part 14 training-run configuration.

Optimizer, schedule, batch size, replay window, class balance, checkpoint
cadence, and stopping rule must be explicit. Part 13 layout and Part 12
objective values are required inputs — no silent trainer defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from training.morpheus.objective.config import ObjectiveConfig, load_objective_config


class TrainConfigError(ValueError):
    """Training run config is incomplete or invalid."""


def _require_finite(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainConfigError(f"{name} must be a finite number") from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise TrainConfigError(f"{name} must be a finite number, got {value!r}")
    return out


def _require_positive(name: str, value: Any) -> float:
    out = _require_finite(name, value)
    if out <= 0.0:
        raise TrainConfigError(f"{name} must be > 0, got {out}")
    return out


def _require_nonneg(name: str, value: Any) -> float:
    out = _require_finite(name, value)
    if out < 0.0:
        raise TrainConfigError(f"{name} must be >= 0, got {out}")
    return out


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainConfigError(f"{name} must be an integer") from exc
    if float(value) != float(out):
        raise TrainConfigError(f"{name} must be an integer, got {value!r}")
    if out < minimum:
        raise TrainConfigError(f"{name} must be >= {minimum}, got {out}")
    return out


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> OptimizerConfig:
        if data is None:
            raise TrainConfigError("optimizer is required (fail closed)")
        required = ("name", "learning_rate", "weight_decay", "betas", "eps")
        missing = [k for k in required if k not in data]
        if missing:
            raise TrainConfigError(f"optimizer missing keys: {missing}")
        betas = data["betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise TrainConfigError("optimizer.betas must be a length-2 list")
        name = str(data["name"]).lower()
        if name != "adamw":
            raise TrainConfigError(f"unsupported optimizer {name!r}; only adamw")
        return cls(
            name=name,
            learning_rate=_require_positive("learning_rate", data["learning_rate"]),
            weight_decay=_require_nonneg("weight_decay", data["weight_decay"]),
            betas=(
                _require_positive("betas[0]", betas[0]),
                _require_positive("betas[1]", betas[1]),
            ),
            eps=_require_positive("eps", data["eps"]),
        )


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    step_size: int
    gamma: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "step_size": self.step_size,
            "gamma": self.gamma,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> SchedulerConfig:
        if data is None:
            raise TrainConfigError("scheduler is required (fail closed)")
        required = ("name", "step_size", "gamma")
        missing = [k for k in required if k not in data]
        if missing:
            raise TrainConfigError(f"scheduler missing keys: {missing}")
        name = str(data["name"]).lower()
        if name not in ("constant", "step"):
            raise TrainConfigError(
                f"unsupported scheduler {name!r}; expected constant|step"
            )
        return cls(
            name=name,
            step_size=_require_int("step_size", data["step_size"], minimum=1),
            gamma=_require_positive("gamma", data["gamma"]),
        )


@dataclass(frozen=True)
class ReplayConfig:
    window_size: int
    class_balance: dict[str, float]
    augment_symmetries: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "class_balance": dict(self.class_balance),
            "augment_symmetries": self.augment_symmetries,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ReplayConfig:
        if data is None:
            raise TrainConfigError("replay is required (fail closed)")
        required = ("window_size", "class_balance", "augment_symmetries")
        missing = [k for k in required if k not in data]
        if missing:
            raise TrainConfigError(f"replay missing keys: {missing}")
        balance = data["class_balance"]
        if not isinstance(balance, Mapping) or not balance:
            raise TrainConfigError("replay.class_balance must be a non-empty object")
        parsed = {
            str(k): _require_nonneg(f"class_balance[{k}]", v) for k, v in balance.items()
        }
        if sum(parsed.values()) <= 0.0:
            raise TrainConfigError("replay.class_balance weights must sum to > 0")
        return cls(
            window_size=_require_int("window_size", data["window_size"], minimum=1),
            class_balance=parsed,
            augment_symmetries=bool(data["augment_symmetries"]),
        )


@dataclass(frozen=True)
class CadenceConfig:
    snapshot_every_steps: int
    max_steps: int
    games_per_checkpoint: int
    checkpoint_count: int
    stopping_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_every_steps": self.snapshot_every_steps,
            "max_steps": self.max_steps,
            "games_per_checkpoint": self.games_per_checkpoint,
            "checkpoint_count": self.checkpoint_count,
            "stopping_rule": self.stopping_rule,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> CadenceConfig:
        if data is None:
            raise TrainConfigError("cadence is required (fail closed)")
        required = (
            "snapshot_every_steps",
            "max_steps",
            "games_per_checkpoint",
            "checkpoint_count",
            "stopping_rule",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise TrainConfigError(f"cadence missing keys: {missing}")
        rule = str(data["stopping_rule"])
        if rule not in ("max_steps", "budget_exhausted"):
            raise TrainConfigError(
                f"unsupported stopping_rule {rule!r}; expected max_steps|budget_exhausted"
            )
        return cls(
            snapshot_every_steps=_require_int(
                "snapshot_every_steps", data["snapshot_every_steps"], minimum=1
            ),
            max_steps=_require_int("max_steps", data["max_steps"], minimum=1),
            games_per_checkpoint=_require_int(
                "games_per_checkpoint", data["games_per_checkpoint"], minimum=1
            ),
            checkpoint_count=_require_int(
                "checkpoint_count", data["checkpoint_count"], minimum=1
            ),
            stopping_rule=rule,
        )


@dataclass(frozen=True)
class BudgetConfig:
    main_training_a100_hours: float
    deployment_calibration_a100_hours: float
    prior_a100_hours: float
    budget_a100_hours: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_training_a100_hours": self.main_training_a100_hours,
            "deployment_calibration_a100_hours": self.deployment_calibration_a100_hours,
            "prior_a100_hours": self.prior_a100_hours,
            "budget_a100_hours": self.budget_a100_hours,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> BudgetConfig:
        if data is None:
            raise TrainConfigError("budget is required (fail closed)")
        required = (
            "main_training_a100_hours",
            "deployment_calibration_a100_hours",
            "prior_a100_hours",
            "budget_a100_hours",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise TrainConfigError(f"budget missing keys: {missing}")
        cfg = cls(
            main_training_a100_hours=_require_nonneg(
                "main_training_a100_hours", data["main_training_a100_hours"]
            ),
            deployment_calibration_a100_hours=_require_nonneg(
                "deployment_calibration_a100_hours",
                data["deployment_calibration_a100_hours"],
            ),
            prior_a100_hours=_require_nonneg(
                "prior_a100_hours", data["prior_a100_hours"]
            ),
            budget_a100_hours=_require_positive(
                "budget_a100_hours", data["budget_a100_hours"]
            ),
        )
        planned = (
            cfg.prior_a100_hours
            + cfg.main_training_a100_hours
            + cfg.deployment_calibration_a100_hours
        )
        if planned > cfg.budget_a100_hours + 1e-12:
            raise TrainConfigError(
                f"planned A100 hours {planned} exceed budget {cfg.budget_a100_hours}"
            )
        return cfg


@dataclass(frozen=True)
class TrainRunConfig:
    """Complete fail-closed config for ``train`` / ``resume``."""

    name: str
    scope: str
    promotable_main_run: bool
    part13_verdict: str
    part13_fallback: str | None
    seed: int
    batch_size: int
    n_blocks: int
    objective: ObjectiveConfig
    rated_objective: ObjectiveConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    replay: ReplayConfig
    cadence: CadenceConfig
    budget: BudgetConfig
    layout: dict[str, Any]
    deployment: dict[str, Any]
    self_play: dict[str, Any]
    buffer_dir: str
    run_root: str
    require_deployment_calibration: bool = True
    notes: tuple[str, ...] = ()
    objective_path: str | None = None

    def trainer_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "n_blocks": self.n_blocks,
            "optimizer": self.optimizer.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "replay": self.replay.to_dict(),
            "cadence": self.cadence.to_dict(),
            "budget": self.budget.to_dict(),
            "require_deployment_calibration": self.require_deployment_calibration,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope,
            "promotable_main_run": self.promotable_main_run,
            "part13_verdict": self.part13_verdict,
            "part13_fallback": self.part13_fallback,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "n_blocks": self.n_blocks,
            "objective": self.objective.to_dict(),
            "rated_objective": self.rated_objective.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "replay": self.replay.to_dict(),
            "cadence": self.cadence.to_dict(),
            "budget": self.budget.to_dict(),
            "layout": dict(self.layout),
            "deployment": dict(self.deployment),
            "self_play": dict(self.self_play),
            "buffer_dir": self.buffer_dir,
            "run_root": self.run_root,
            "require_deployment_calibration": self.require_deployment_calibration,
            "notes": list(self.notes),
            "objective_path": self.objective_path,
        }


def load_train_run_config(path: Path | str) -> TrainRunConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TrainConfigError("train run config must be an object")

    required = (
        "name",
        "scope",
        "promotable_main_run",
        "part13_verdict",
        "seed",
        "batch_size",
        "n_blocks",
        "optimizer",
        "scheduler",
        "replay",
        "cadence",
        "budget",
        "layout",
        "deployment",
        "self_play",
        "buffer_dir",
        "run_root",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise TrainConfigError(f"train run config missing keys: {missing}")

    objective_path = data.get("objective_path")
    if objective_path and not Path(objective_path).is_file():
        candidates = [
            path.parent / objective_path,
            path.resolve().parents[3] / objective_path,
        ]
        for alt in candidates:
            if alt.is_file():
                objective_path = str(alt)
                break

    if "objective" in data:
        objective = ObjectiveConfig.from_dict(data["objective"])
    elif objective_path:
        objective = load_objective_config(Path(objective_path))
    else:
        raise TrainConfigError("objective or objective_path is required")

    if "rated_objective" in data:
        rated = ObjectiveConfig.from_dict(data["rated_objective"])
    elif objective_path:
        bundle_path = Path(objective_path)
        raw = json.loads(bundle_path.read_text(encoding="utf-8"))
        if "rated_objective" not in raw:
            raise TrainConfigError(
                f"{bundle_path}: rated_objective required for Part 14"
            )
        rated = ObjectiveConfig.from_dict(raw["rated_objective"])
    else:
        raise TrainConfigError("rated_objective is required")

    if rated.mode != "rated":
        raise TrainConfigError("rated_objective.mode must be 'rated'")

    promotable = bool(data["promotable_main_run"])
    scope = str(data["scope"])
    verdict = str(data["part13_verdict"])
    if promotable and verdict != "yes":
        raise TrainConfigError(
            "promotable_main_run=true requires part13_verdict='yes'"
        )
    if promotable and scope != "promotable_main_run":
        raise TrainConfigError(
            "promotable_main_run=true requires scope='promotable_main_run'"
        )

    layout = data["layout"]
    if not isinstance(layout, Mapping) or not layout:
        raise TrainConfigError("layout must be a non-empty object")
    for key in (
        "backend",
        "physical_cores_per_game",
        "seat_search",
        "workers_per_a100",
    ):
        if key not in layout:
            raise TrainConfigError(f"layout missing required key {key!r}")

    deployment = data["deployment"]
    if not isinstance(deployment, Mapping) or not deployment:
        raise TrainConfigError("deployment must be a non-empty object")
    self_play = data["self_play"]
    if not isinstance(self_play, Mapping) or not self_play:
        raise TrainConfigError("self_play must be a non-empty object")

    return TrainRunConfig(
        name=str(data["name"]),
        scope=scope,
        promotable_main_run=promotable,
        part13_verdict=verdict,
        part13_fallback=(
            None
            if data.get("part13_fallback") is None
            else str(data["part13_fallback"])
        ),
        seed=_require_int("seed", data["seed"], minimum=0),
        batch_size=_require_int("batch_size", data["batch_size"], minimum=1),
        n_blocks=_require_int("n_blocks", data["n_blocks"], minimum=1),
        objective=objective,
        rated_objective=rated,
        optimizer=OptimizerConfig.from_dict(data["optimizer"]),
        scheduler=SchedulerConfig.from_dict(data["scheduler"]),
        replay=ReplayConfig.from_dict(data["replay"]),
        cadence=CadenceConfig.from_dict(data["cadence"]),
        budget=BudgetConfig.from_dict(data["budget"]),
        layout=dict(layout),
        deployment=dict(deployment),
        self_play=dict(self_play),
        buffer_dir=str(data["buffer_dir"]),
        run_root=str(data["run_root"]),
        require_deployment_calibration=bool(
            data.get("require_deployment_calibration", True)
        ),
        notes=tuple(str(x) for x in (data.get("notes") or ())),
        objective_path=str(objective_path) if objective_path else None,
    )
