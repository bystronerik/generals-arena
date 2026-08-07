"""Named Morpheus deployment configuration (Part 09).

Play-time fields live in ``deployment.json`` inside the bot closure. The operator
copy under ``scripts/configs/morpheus/online-runtime.json`` must match after
qualification. Every estimator and safety field is explicit — no silent defaults
at acceptance time.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from runtime import (
    COST_COMPONENTS,
    DEFAULT_OFFLINE_P99_MS,
    RuntimeConfig,
)

BOT_DIR = Path(__file__).resolve().parent
DEFAULT_DEPLOYMENT_PATH = BOT_DIR / "deployment.json"

P99_ESTIMATOR_TYPE = "nearest_rank_empirical"
P99_WARMUP_RULE = "max_of_offline_p99_and_local_samples_until_window_full"


@dataclass
class DeploymentConfig:
    """Complete coupled online configuration for one checkpoint family."""

    # Inference / export identity
    inference_runtime: str = "torch.jit.script+float32"
    quantization_format: str = "float32"
    quantization_engine: str = "none"
    trunk_channels: int = 64
    network_width: int = 64

    # Coupled search / belief budget
    n_particles: int = 32
    target_simulations: int = 8
    min_simulations: int = 8
    pending_leaf_batch: int = 4
    max_proposal_batch: int = 16
    search_depth: int = 8
    max_forward_equivalents: int = 113
    max_tree_nodes: int = 4096
    widen_freeze_below: int = 16

    # Deadlines and reserve
    normal_deadline_ms: float = 125.0
    reserve_ms: float = 25.0
    first_move_limit_ms: float = 8500.0
    admission_guard_ms: float = 10.0
    resident_memory_target_mb: float = 256.0
    memory_cap_bytes: int = 2 * 1024**3

    # Part 07 estimator fields (Part 09 selects measured values)
    p99_estimator_type: str = P99_ESTIMATOR_TYPE
    p99_warmup_rule: str = P99_WARMUP_RULE
    p99_window: int = 64
    offline_p99_ms: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_OFFLINE_P99_MS)
    )

    # First-move warm-up batch shapes exercised at load
    warmup_batch_shapes: tuple[int, ...] = (1, 4, 16)

    # Belief limitation recorded at qualification (not a quality gate)
    belief_quality_threshold: Optional[float] = None
    belief_limitation_note: str = (
        "No belief-quality threshold is defined. Qualification rejects a config "
        "that hides belief collapse behind a fast reply; recovery and ESS stay "
        "visible in probe metrics."
    )

    # Machine note — local latency is conditional on this host
    qualification_host: str = ""
    qualification_cpu: str = ""

    def to_runtime_config(self) -> RuntimeConfig:
        offline = {name: float(self.offline_p99_ms.get(name, 1.0)) for name in COST_COMPONENTS}
        for name, value in self.offline_p99_ms.items():
            offline[str(name)] = float(value)
        return RuntimeConfig(
            normal_deadline_ms=float(self.normal_deadline_ms),
            reserve_ms=float(self.reserve_ms),
            first_move_limit_ms=float(self.first_move_limit_ms),
            target_simulations=int(self.target_simulations),
            min_simulations=int(self.min_simulations),
            pending_leaf_batch=int(self.pending_leaf_batch),
            max_forward_equivalents=int(self.max_forward_equivalents),
            admission_guard_ms=float(self.admission_guard_ms),
            max_tree_nodes=int(self.max_tree_nodes),
            resident_memory_target_mb=float(self.resident_memory_target_mb),
            p99_window=int(self.p99_window),
            offline_p99_ms=offline,
            n_particles=int(self.n_particles),
            max_proposal_batch=int(self.max_proposal_batch),
            search_depth=int(self.search_depth),
            widen_freeze_below=int(self.widen_freeze_below),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["warmup_batch_shapes"] = list(self.warmup_batch_shapes)
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DeploymentConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            if key == "warmup_batch_shapes":
                kwargs[key] = tuple(int(x) for x in value)
            elif key == "offline_p99_ms":
                kwargs[key] = {str(k): float(v) for k, v in dict(value).items()}
            elif key == "belief_quality_threshold":
                kwargs[key] = None if value is None else float(value)
            else:
                kwargs[key] = value
        cfg = cls(**kwargs)
        missing = [name for name in COST_COMPONENTS if name not in cfg.offline_p99_ms]
        if missing:
            raise ValueError(
                f"deployment offline_p99_ms missing components: {missing}"
            )
        if cfg.p99_estimator_type != P99_ESTIMATOR_TYPE:
            raise ValueError(
                f"unsupported p99_estimator_type {cfg.p99_estimator_type!r}"
            )
        if not cfg.p99_warmup_rule:
            raise ValueError("p99_warmup_rule must be explicit")
        if cfg.p99_window < 1:
            raise ValueError("p99_window must be >= 1")
        if cfg.admission_guard_ms < 0:
            raise ValueError("admission_guard_ms must be >= 0")
        return cfg


def load_deployment(path: Optional[Path] = None) -> DeploymentConfig:
    target = Path(path) if path is not None else DEFAULT_DEPLOYMENT_PATH
    if not target.is_file():
        raise FileNotFoundError(f"deployment config missing: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"deployment root must be an object: {target}")
    return DeploymentConfig.from_mapping(data)


def save_deployment(config: DeploymentConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()
    offline = data.get("offline_p99_ms", {})
    data["offline_p99_ms"] = {
        str(k): (1.0 if v != v else float(v)) for k, v in offline.items()
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def try_load_deployment(path: Optional[Path] = None) -> DeploymentConfig:
    """Load deployment.json when present; otherwise return Part 07 placeholders."""
    target = Path(path) if path is not None else DEFAULT_DEPLOYMENT_PATH
    if target.is_file():
        return load_deployment(target)
    return DeploymentConfig()
