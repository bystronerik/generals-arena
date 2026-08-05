"""Steady-state throughput and A100 budget formulas for Part 13."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

A100_BUDGET_HOURS = 48.0

A100_LINE_ITEMS = (
    "jax_preflight",
    "throughput_qualification",
    "learning_curve_pilot",
    "objective_ablations",
    "gpu_self_play",
    "main_training",
    "deployment_calibration",
)


@dataclass(frozen=True)
class A100ChargeSheet:
    jax_preflight: float = 0.0
    throughput_qualification: float = 0.0
    learning_curve_pilot: float = 0.0
    objective_ablations: float = 0.0
    gpu_self_play: float = 0.0
    main_training: float = 0.0
    deployment_calibration: float = 0.0

    def total(self) -> float:
        return sum(getattr(self, name) for name in A100_LINE_ITEMS)

    def fits_budget(self, budget: float = A100_BUDGET_HOURS) -> bool:
        return self.total() <= float(budget) + 1e-12

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in A100_LINE_ITEMS}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> A100ChargeSheet:
        raw = dict(data or {})
        kwargs = {}
        for f in fields(cls):
            kwargs[f.name] = float(raw.get(f.name, 0.0) or 0.0)
        return cls(**kwargs)


@dataclass(frozen=True)
class SteadyStateRates:
    completed_games: int
    worker_hours: float
    workers: int
    positions_per_game: float
    self_play_wall_hours: float
    checkpoint_count: int
    games_per_worker_hour: float
    aggregate_games_per_hour: float
    positions_supply_per_hour: float
    total_games: float
    games_per_checkpoint: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def games_per_worker_hour(completed_games: int, worker_hours: float) -> float:
    if worker_hours <= 0:
        raise ValueError("worker_hours must be > 0")
    if completed_games < 0:
        raise ValueError("completed_games must be >= 0")
    return float(completed_games) / float(worker_hours)


def aggregate_games_per_hour(workers: int, games_per_worker_hour_value: float) -> float:
    if workers < 0:
        raise ValueError("workers must be >= 0")
    return float(workers) * float(games_per_worker_hour_value)


def positions_supply_per_hour(
    aggregate_games_per_hour_value: float,
    positions_per_game: float,
) -> float:
    return float(aggregate_games_per_hour_value) * float(positions_per_game)


def total_games_for_wall(
    aggregate_games_per_hour_value: float,
    self_play_wall_hours: float,
) -> float:
    return float(aggregate_games_per_hour_value) * float(self_play_wall_hours)


def games_per_checkpoint(total_games: float, checkpoint_count: int) -> int:
    if checkpoint_count <= 0:
        raise ValueError("checkpoint_count must be > 0")
    return int(math.floor(float(total_games) / float(checkpoint_count)))


def steady_state_rates(
    *,
    completed_games: int,
    worker_hours: float,
    workers: int,
    positions_per_game: float,
    self_play_wall_hours: float,
    checkpoint_count: int,
) -> SteadyStateRates:
    gph_worker = games_per_worker_hour(completed_games, worker_hours)
    agg = aggregate_games_per_hour(workers, gph_worker)
    pos = positions_supply_per_hour(agg, positions_per_game)
    total = total_games_for_wall(agg, self_play_wall_hours)
    gpc = games_per_checkpoint(total, checkpoint_count)
    return SteadyStateRates(
        completed_games=int(completed_games),
        worker_hours=float(worker_hours),
        workers=int(workers),
        positions_per_game=float(positions_per_game),
        self_play_wall_hours=float(self_play_wall_hours),
        checkpoint_count=int(checkpoint_count),
        games_per_worker_hour=gph_worker,
        aggregate_games_per_hour=agg,
        positions_supply_per_hour=pos,
        total_games=total,
        games_per_checkpoint=gpc,
    )


def aggregate_a100_hours(charges: A100ChargeSheet | Mapping[str, Any]) -> dict[str, Any]:
    sheet = (
        charges
        if isinstance(charges, A100ChargeSheet)
        else A100ChargeSheet.from_mapping(charges)
    )
    total = sheet.total()
    return {
        "lines": sheet.to_dict(),
        "total_a100_hours": total,
        "budget_hours": A100_BUDGET_HOURS,
        "fits_budget": sheet.fits_budget(),
        "remaining_hours": A100_BUDGET_HOURS - total,
    }


def learner_starved(
    *,
    positions_supply_per_hour: float,
    positions_consume_per_hour: float,
) -> bool:
    """True when shard production cannot meet measured training consumption."""
    if positions_consume_per_hour <= 0:
        return False
    return float(positions_supply_per_hour) + 1e-12 < float(positions_consume_per_hour)


def select_layout(layout_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Pick layout by completed games per CPU-hour, then lower end-to-end latency."""
    if not layout_rows:
        raise ValueError("layout_rows must be non-empty")
    scored: list[tuple[float, float, Mapping[str, Any]]] = []
    for row in layout_rows:
        cpu_hours = float(row.get("cpu_hours") or 0.0)
        games = int(row.get("completed_games") or 0)
        if cpu_hours <= 0:
            continue
        games_per_cpu_hour = games / cpu_hours
        latency_s = float(row.get("mean_game_latency_s") or row.get("wall_s") or 0.0)
        scored.append((games_per_cpu_hour, -latency_s, row))
    if not scored:
        raise ValueError("no layout with cpu_hours > 0")
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = scored[0][2]
    return {
        "selected_name": best.get("name"),
        "selected": dict(best),
        "ranking": [
            {
                "name": r.get("name"),
                "games_per_cpu_hour": gph,
                "mean_game_latency_s": float(
                    r.get("mean_game_latency_s") or r.get("wall_s") or 0.0
                ),
            }
            for gph, _, r in scored
        ],
    }
