"""Fail-closed training objective and exploration configuration.

Every loss weight and exploration field must be present. Rated play forces
root noise and action temperature off. There are no silent defaults for a
training run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# Required loss-weight keys (must all be present and finite).
LOSS_WEIGHT_KEYS: tuple[str, ...] = (
    "policy",
    "wdl",
    "hidden_owner",
    "enemy_army_bins",
    "enemy_general",
    "hidden_castle",
    "land_margin",
    "army_margin",
    "castle_margin",
    "turns_to_termination",
)

# Required exploration keys for a training run.
EXPLORATION_KEYS: tuple[str, ...] = (
    "root_noise_epsilon",
    "root_noise_alpha",
    "action_temperature",
    "deterministic_turn",
)

# Scalar-target normalization keys (explicit; no silent scale).
SCALAR_NORM_KEYS: tuple[str, ...] = (
    "land_margin_scale",
    "army_margin_scale",
    "castle_margin_scale",
    "turns_to_termination_scale",
)


class ObjectiveConfigError(ValueError):
    """Raised when a required objective field is missing or invalid."""


def _require_finite(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ObjectiveConfigError(f"{name} must be a finite number") from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise ObjectiveConfigError(f"{name} must be a finite number, got {value!r}")
    return out


def _require_nonneg(name: str, value: Any) -> float:
    out = _require_finite(name, value)
    if out < 0.0:
        raise ObjectiveConfigError(f"{name} must be >= 0, got {out}")
    return out


def _require_positive(name: str, value: Any) -> float:
    out = _require_finite(name, value)
    if out <= 0.0:
        raise ObjectiveConfigError(f"{name} must be > 0, got {out}")
    return out


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ObjectiveConfigError(f"{name} must be an integer") from exc
    if out != value and not isinstance(value, bool):
        # Reject 1.5 silently coerced; allow 1.0.
        if float(value) != float(out):
            raise ObjectiveConfigError(f"{name} must be an integer, got {value!r}")
    if out < minimum:
        raise ObjectiveConfigError(f"{name} must be >= {minimum}, got {out}")
    return out


@dataclass(frozen=True)
class LossWeights:
    """Explicit auxiliary and primary loss weights. No omitted keys."""

    policy: float
    wdl: float
    hidden_owner: float
    enemy_army_bins: float
    enemy_general: float
    hidden_castle: float
    land_margin: float
    army_margin: float
    castle_margin: float
    turns_to_termination: float

    def __post_init__(self) -> None:
        for key in LOSS_WEIGHT_KEYS:
            _require_nonneg(f"loss_weights.{key}", getattr(self, key))

    def to_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in LOSS_WEIGHT_KEYS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> LossWeights:
        if data is None:
            raise ObjectiveConfigError(
                "loss_weights is required; omit nothing (fail closed)"
            )
        missing = [k for k in LOSS_WEIGHT_KEYS if k not in data]
        if missing:
            raise ObjectiveConfigError(
                f"loss_weights missing required keys: {missing}"
            )
        extra = [k for k in data if k not in LOSS_WEIGHT_KEYS]
        if extra:
            raise ObjectiveConfigError(f"loss_weights has unknown keys: {extra}")
        return cls(**{k: _require_nonneg(f"loss_weights.{k}", data[k]) for k in LOSS_WEIGHT_KEYS})


@dataclass(frozen=True)
class ExplorationConfig:
    """Root noise and action temperature. Rated play forces all exploration off."""

    root_noise_epsilon: float
    root_noise_alpha: float
    action_temperature: float
    deterministic_turn: int

    def __post_init__(self) -> None:
        _require_nonneg("root_noise_epsilon", self.root_noise_epsilon)
        if self.root_noise_epsilon > 1.0:
            raise ObjectiveConfigError("root_noise_epsilon must be in [0, 1]")
        _require_positive("root_noise_alpha", self.root_noise_alpha)
        _require_positive("action_temperature", self.action_temperature)
        _require_int("deterministic_turn", self.deterministic_turn, minimum=0)

    @property
    def exploration_enabled(self) -> bool:
        return self.root_noise_epsilon > 0.0 or self.action_temperature != 1.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "root_noise_epsilon": float(self.root_noise_epsilon),
            "root_noise_alpha": float(self.root_noise_alpha),
            "action_temperature": float(self.action_temperature),
            "deterministic_turn": int(self.deterministic_turn),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ExplorationConfig:
        if data is None:
            raise ObjectiveConfigError(
                "exploration is required; omit nothing (fail closed)"
            )
        missing = [k for k in EXPLORATION_KEYS if k not in data]
        if missing:
            raise ObjectiveConfigError(f"exploration missing required keys: {missing}")
        extra = [k for k in data if k not in EXPLORATION_KEYS]
        if extra:
            raise ObjectiveConfigError(f"exploration has unknown keys: {extra}")
        return cls(
            root_noise_epsilon=_require_nonneg(
                "root_noise_epsilon", data["root_noise_epsilon"]
            ),
            root_noise_alpha=_require_positive(
                "root_noise_alpha", data["root_noise_alpha"]
            ),
            action_temperature=_require_positive(
                "action_temperature", data["action_temperature"]
            ),
            deterministic_turn=_require_int(
                "deterministic_turn", data["deterministic_turn"], minimum=0
            ),
        )

    @classmethod
    def rated(cls) -> ExplorationConfig:
        """Rated play: no root noise, temperature 1, immediate deterministic turn."""
        return cls(
            root_noise_epsilon=0.0,
            root_noise_alpha=1.0,  # unused when epsilon is 0; still explicit
            action_temperature=1.0,
            deterministic_turn=0,
        )


@dataclass(frozen=True)
class ScalarNormalization:
    """Explicit scales for margin and termination regression targets."""

    land_margin_scale: float
    army_margin_scale: float
    castle_margin_scale: float
    turns_to_termination_scale: float

    def __post_init__(self) -> None:
        for key in SCALAR_NORM_KEYS:
            _require_positive(key, getattr(self, key))

    def to_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in SCALAR_NORM_KEYS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ScalarNormalization:
        if data is None:
            raise ObjectiveConfigError(
                "scalar_normalization is required; omit nothing (fail closed)"
            )
        missing = [k for k in SCALAR_NORM_KEYS if k not in data]
        if missing:
            raise ObjectiveConfigError(
                f"scalar_normalization missing required keys: {missing}"
            )
        extra = [k for k in data if k not in SCALAR_NORM_KEYS]
        if extra:
            raise ObjectiveConfigError(
                f"scalar_normalization has unknown keys: {extra}"
            )
        return cls(
            **{
                k: _require_positive(k, data[k])
                for k in SCALAR_NORM_KEYS
            }
        )


@dataclass(frozen=True)
class ObjectiveConfig:
    """Complete fail-closed objective for one training or ablation candidate."""

    name: str
    loss_weights: LossWeights
    exploration: ExplorationConfig
    scalar_normalization: ScalarNormalization
    mode: str = "training"  # "training" | "rated"
    reward: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("training", "rated"):
            raise ObjectiveConfigError(f"mode must be training|rated, got {self.mode!r}")
        if self.mode == "rated" and self.exploration.exploration_enabled:
            raise ObjectiveConfigError(
                "rated mode forbids root noise and action temperature "
                f"(got {self.exploration.to_dict()})"
            )
        from training.morpheus.objective.reward import (
            assert_no_shaping,
            reward_contract,
        )

        spec = dict(self.reward) if self.reward is not None else reward_contract()
        assert_no_shaping(spec)
        object.__setattr__(self, "reward", spec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "loss_weights": self.loss_weights.to_dict(),
            "exploration": self.exploration.to_dict(),
            "scalar_normalization": self.scalar_normalization.to_dict(),
            "reward": dict(self.reward or {}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObjectiveConfig:
        if not isinstance(data, Mapping):
            raise ObjectiveConfigError("objective config must be an object")
        required = (
            "name",
            "loss_weights",
            "exploration",
            "scalar_normalization",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise ObjectiveConfigError(
                f"objective config missing required top-level keys: {missing}"
            )
        mode = str(data.get("mode", "training"))
        return cls(
            name=str(data["name"]),
            mode=mode,
            loss_weights=LossWeights.from_dict(data["loss_weights"]),
            exploration=ExplorationConfig.from_dict(data["exploration"]),
            scalar_normalization=ScalarNormalization.from_dict(
                data["scalar_normalization"]
            ),
            reward=data.get("reward"),
        )


def load_objective_config(path: Path | str) -> ObjectiveConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "objective" in data:
        data = data["objective"]
    return ObjectiveConfig.from_dict(data)


def load_pilot_objective_bundle(path: Path | str) -> dict[str, Any]:
    """Load provisional pilot freeze: training objective + rated twin + metadata."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "objective" not in data:
        raise ObjectiveConfigError(
            "pilot objective bundle requires an 'objective' object (fail closed)"
        )
    if "rated_objective" not in data:
        raise ObjectiveConfigError(
            "pilot objective bundle requires a 'rated_objective' object (fail closed)"
        )
    training = ObjectiveConfig.from_dict(data["objective"])
    rated = ObjectiveConfig.from_dict(data["rated_objective"])
    assert_rated_disables_exploration(rated)
    if bool(data.get("promotable_main_run")):
        raise ObjectiveConfigError(
            "pilot bundle must set promotable_main_run=false until Part 13 "
            "records the pending ablation measures"
        )
    return {
        "status": str(data.get("status") or "provisional_pilot"),
        "selected_candidate": str(data.get("selected_candidate") or training.name),
        "promotable_main_run": False,
        "selection_note": str(data.get("selection_note") or ""),
        "evidence": dict(data.get("evidence") or {}),
        "training": training,
        "rated": rated,
    }


def load_ablation_candidates(path: Path | str) -> list[ObjectiveConfig]:
    """Load an ablation file with an explicit ``candidates`` list."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "candidates" not in data:
        raise ObjectiveConfigError(
            "ablation config requires a candidates list (fail closed)"
        )
    candidates = data["candidates"]
    if not isinstance(candidates, Sequence) or not candidates:
        raise ObjectiveConfigError("candidates must be a non-empty list")
    out: list[ObjectiveConfig] = []
    for i, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise ObjectiveConfigError(f"candidates[{i}] must be an object")
        out.append(ObjectiveConfig.from_dict(row))
    return out


def assert_rated_disables_exploration(config: ObjectiveConfig) -> None:
    """Exit-criterion helper: rated mode must disable exploration."""
    if config.mode != "rated":
        raise ObjectiveConfigError("expected mode='rated'")
    rated = ExplorationConfig.rated()
    if config.exploration.root_noise_epsilon != 0.0:
        raise ObjectiveConfigError("rated root_noise_epsilon must be 0")
    if config.exploration.action_temperature != 1.0:
        raise ObjectiveConfigError("rated action_temperature must be 1")
    if config.exploration.deterministic_turn != 0:
        raise ObjectiveConfigError("rated deterministic_turn must be 0")
    _ = rated
