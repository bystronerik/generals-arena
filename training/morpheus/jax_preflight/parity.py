"""CPU/GPU state snapshot comparison for the competition transition."""
from __future__ import annotations

from typing import Any

import numpy as np

COMPARE_KEYS = (
    "armies",
    "ownership",
    "castles",
    "time",
    "winner",
    "army_totals",
    "land_totals",
)


def state_info_snapshot(state, info) -> dict[str, Any]:
    """Serialize the fields Part 00 requires for CPU/GPU parity."""
    return {
        "armies": np.asarray(state.armies),
        "ownership": np.asarray(state.ownership),
        "castles": np.asarray(state.castles),
        "time": int(np.asarray(state.time)),
        "winner": int(np.asarray(state.winner)),
        "army_totals": np.asarray(info.army),
        "land_totals": np.asarray(info.land),
    }


def compare_snapshots(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Return per-field equality for two snapshots."""
    fields: dict[str, bool] = {}
    for key in COMPARE_KEYS:
        av, bv = a[key], b[key]
        if isinstance(av, np.ndarray) or isinstance(bv, np.ndarray):
            fields[key] = bool(np.array_equal(np.asarray(av), np.asarray(bv)))
        else:
            fields[key] = av == bv
    return {
        "match": all(fields.values()),
        "fields": fields,
    }


def snapshots_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return bool(compare_snapshots(a, b)["match"])


def snapshot_to_jsonable(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Convert arrays to nested lists for JSON reports."""
    out: dict[str, Any] = {}
    for key, value in snapshot.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out
