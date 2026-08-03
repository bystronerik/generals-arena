"""Army-scale measurement for Morpheus Part 03.

Replays bootstrap trajectories and reports army quantiles by turn band and
outcome, plus quantization error under ``log1p(x) / log1p(scale)``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from arena.records.trajectories import read_trajectory, replay_states
from training.morpheus.corpus.coverage import TURN_BANDS, turn_band_for

DEFAULT_SCALE = 4096.0
QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)


def army_value(x: np.ndarray, scale: float) -> np.ndarray:
    denom = math.log1p(scale)
    return np.clip(np.log1p(np.maximum(x.astype(np.float64), 0.0)) / denom, 0.0, 1.0)


def quantization_error(armies: np.ndarray, scale: float, levels: int = 256) -> dict[str, float]:
    """Uniform quantization of the transformed value into ``levels`` bins."""
    t = army_value(armies, scale)
    q = np.round(t * (levels - 1)) / (levels - 1)
    # Reconstruct approx via inverse: expm1(q * log1p(scale))
    recon = np.expm1(q * math.log1p(scale))
    err = np.abs(recon - armies.astype(np.float64))
    rel = err / np.maximum(armies.astype(np.float64), 1.0)
    return {
        "mae": float(err.mean()) if len(err) else 0.0,
        "max_abs": float(err.max()) if len(err) else 0.0,
        "mean_rel": float(rel.mean()) if len(rel) else 0.0,
        "p99_rel": float(np.quantile(rel, 0.99)) if len(rel) else 0.0,
    }


def _collect_armies(trajectories_dir: Path, max_games: int) -> dict[str, Any]:
    paths = sorted(trajectories_dir.glob("*.traj.jsonl.gz"))
    if max_games > 0:
        paths = paths[:max_games]

    by_band: dict[str, list[int]] = defaultdict(list)
    by_outcome: dict[str, list[int]] = defaultdict(list)
    extremes: list[int] = []
    all_armies: list[int] = []
    game_count = 0

    for path in paths:
        traj = read_trajectory(path)
        end = traj.end
        winner = str(end.get("winner", "draw"))
        outcome = winner if winner in ("a", "b") else "draw"
        game_count += 1
        for turn, state, _info in replay_states(traj):
            armies = np.asarray(state.armies, dtype=np.int32)
            ownership = np.asarray(state.ownership)
            # Positive stacks on owned cells only.
            for p in (0, 1):
                vals = armies[ownership[p]]
                vals = vals[vals > 0]
                if vals.size == 0:
                    continue
                band = turn_band_for(int(turn)) or "other"
                as_list = vals.astype(int).tolist()
                by_band[band].extend(as_list)
                by_outcome[outcome].extend(as_list)
                all_armies.extend(as_list)
                extremes.extend(vals[vals >= 512].astype(int).tolist())

    return {
        "game_count": game_count,
        "by_band": {k: np.asarray(v, dtype=np.int32) for k, v in by_band.items()},
        "by_outcome": {
            k: np.asarray(v, dtype=np.int32) for k, v in by_outcome.items()
        },
        "all": np.asarray(all_armies, dtype=np.int32),
        "extremes": np.asarray(extremes, dtype=np.int32),
    }


def _quantile_table(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {f"p{int(q * 1000) / 10:.1f}".replace(".0", ""): 0.0 for q in QUANTILES}
    qs = np.quantile(arr.astype(np.float64), QUANTILES)
    out = {}
    labels = ("p50", "p90", "p99", "p99.9", "max")
    for label, value in zip(labels, qs):
        out[label] = float(value)
    return out


def measure_army_normalization(
    trajectories_dir: Path,
    *,
    army_scale: float = DEFAULT_SCALE,
    max_games: int = 0,
) -> dict[str, Any]:
    """Return a JSON-serializable army-normalization report."""
    trajectories_dir = Path(trajectories_dir)
    data = _collect_armies(trajectories_dir, max_games)
    all_arr = data["all"]
    sample_count = int(all_arr.size)

    band_stats = {
        name: _quantile_table(data["by_band"].get(name, np.zeros(0, dtype=np.int32)))
        for name, _, _ in TURN_BANDS
    }
    outcome_stats = {
        key: _quantile_table(data["by_outcome"].get(key, np.zeros(0, dtype=np.int32)))
        for key in ("a", "b", "draw")
    }

    ordinary = all_arr[all_arr < 512] if all_arr.size else all_arr
    extreme = data["extremes"]

    err_all = quantization_error(all_arr, army_scale) if sample_count else {}
    err_ord = quantization_error(ordinary, army_scale) if ordinary.size else {}
    err_ext = quantization_error(extreme, army_scale) if extreme.size else {}

    # Keep 4096 unless the max stack saturates hard and p99 sits well below mid-scale.
    saturating = bool(all_arr.size and all_arr.max() > army_scale)
    p99 = float(np.quantile(all_arr, 0.99)) if sample_count else 0.0
    recommendation = (
        "replace_scale"
        if saturating and p99 > army_scale * 0.5
        else "keep_default"
    )

    return {
        "trajectories": str(trajectories_dir),
        "game_count": data["game_count"],
        "sample_count": sample_count,
        "selected_scale": float(army_scale),
        "transform": f"clip(log1p(max(x,0)) / log1p({army_scale}), 0, 1)",
        "quantiles_by_turn_band": band_stats,
        "quantiles_by_outcome": outcome_stats,
        "overall_quantiles": _quantile_table(all_arr),
        "quantization_error": {
            "all": err_all,
            "ordinary_lt_512": err_ord,
            "extreme_ge_512": err_ext,
            "levels": 256,
        },
        "recommendation": recommendation,
        "notes": (
            "Part 03 default remains 4096 until a later part writes a new scale "
            "into the model manifest."
        ),
    }
