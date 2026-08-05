"""Deployment-matched final self-play and calibration (Part 14 exit gate).

Before a checkpoint can reach Part 15, this phase must run with the Part 09
deployment settings. Training-time search budgets must not leak into the
calibration report.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from training.morpheus.self_play.driver import DriverConfig, run_batch
from training.morpheus.trainer.manifest import EVENT_DEPLOYMENT_CALIBRATION


@dataclass(frozen=True)
class CalibrationResult:
    ok: bool
    checkpoint_id: str
    games: int
    wall_s: float
    a100_hours: float
    deployment: dict[str, Any]
    self_play_report: dict[str, Any]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "event": EVENT_DEPLOYMENT_CALIBRATION,
            "checkpoint_id": self.checkpoint_id,
            "games": self.games,
            "wall_s": self.wall_s,
            "a100_hours": self.a100_hours,
            "deployment": dict(self.deployment),
            "self_play_report": dict(self.self_play_report),
            "notes": list(self.notes),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def _driver_from_deployment(
    deployment: Mapping[str, Any],
    self_play: Mapping[str, Any],
    *,
    output: Path,
) -> DriverConfig:
    """Build a DriverConfig that prefers deployment search fields over training ones."""
    merged = dict(self_play)
    for key in (
        "n_particles",
        "target_simulations",
        "min_simulations",
        "search_depth",
        "pending_leaf_batch",
    ):
        if key in deployment:
            merged[key] = deployment[key]
    if "normal_deadline_ms" in deployment:
        merged["deadline_ms"] = float(deployment["normal_deadline_ms"])
    merged["output"] = str(output)
    return DriverConfig.from_dict(merged)


def run_deployment_calibration(
    *,
    run_dir: Path,
    checkpoint_id: str,
    deployment: Mapping[str, Any],
    self_play: Mapping[str, Any],
    repo_root: Path | None = None,
) -> CalibrationResult:
    """Run a short self-play batch under deployment settings and record the report."""
    run_dir = Path(run_dir)
    out = run_dir / "calibration" / checkpoint_id
    out.mkdir(parents=True, exist_ok=True)
    shards_dir = out / "shards"
    driver = _driver_from_deployment(deployment, self_play, output=shards_dir)
    t0 = time.perf_counter()
    batch = run_batch(driver, output=shards_dir, repo_root=repo_root)
    wall_s = time.perf_counter() - t0
    report = {
        "games": int(driver.games),
        "completed": len(batch.shards),
        "output": str(shards_dir),
        "proportions": dict(batch.proportions),
        "driver": {
            "n_particles": driver.n_particles,
            "target_simulations": driver.target_simulations,
            "min_simulations": driver.min_simulations,
            "search_depth": driver.search_depth,
            "pending_leaf_batch": driver.pending_leaf_batch,
            "deadline_ms": driver.deadline_ms,
            "games": driver.games,
            "max_turns": driver.max_turns,
            "seat_search": driver.seat_search,
        },
    }
    ok = len(batch.shards) > 0
    result = CalibrationResult(
        ok=bool(ok),
        checkpoint_id=checkpoint_id,
        games=int(report["games"]),
        wall_s=wall_s,
        a100_hours=wall_s / 3600.0,
        deployment=dict(deployment),
        self_play_report=report,
        notes=(
            "Deployment-matched calibration uses Part 09 search fields.",
            "Training search budgets must not appear in the driver block above.",
            "Arena acceptance remains Part 16; this event is deployment_calibration only.",
        ),
    )
    path = out / "calibration.json"
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
