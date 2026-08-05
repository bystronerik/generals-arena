"""Cadence pilot decision rule for Part 13."""

from __future__ import annotations

from typing import Any, Mapping

from training.morpheus.curriculum.confidence import may_advance_toward_earlier_class


def evaluate_cadence_candidate(
    *,
    name: str,
    games_per_checkpoint: int,
    checkpoint_count: int,
    class_wdl: Mapping[int, Mapping[str, int]] | None = None,
    active_classes: list[int] | None = None,
    belief_calibration_regressed: bool | None = None,
    pairwise_verdict: str | None = None,
) -> dict[str, Any]:
    """Apply the three Part 13 cadence conditions.

    A candidate is useful only when all three hold:

    1. curriculum confidence rule (Part 10) passes;
    2. held-out belief calibration does not regress;
    3. arena pairwise contrast returns ``improvement``.
    """
    classes = list(active_classes or [])
    wdl = {int(k): dict(v) for k, v in (class_wdl or {}).items()}
    if classes and wdl:
        curriculum = may_advance_toward_earlier_class(wdl, active_classes=classes)
    else:
        curriculum = {
            "ok": False,
            "reason": "missing_class_wdl_or_active_classes",
            "classes": {},
        }

    if belief_calibration_regressed is None:
        calibration = {
            "ok": False,
            "regressed": None,
            "reason": "belief_calibration_not_measured",
        }
    else:
        calibration = {
            "ok": not bool(belief_calibration_regressed),
            "regressed": bool(belief_calibration_regressed),
            "reason": None if not belief_calibration_regressed else "regressed",
        }

    verdict = (pairwise_verdict or "").strip().lower() or None
    pairwise = {
        "ok": verdict == "improvement",
        "verdict": verdict,
        "reason": None
        if verdict == "improvement"
        else "pairwise_not_improvement_or_missing",
    }

    useful = bool(curriculum.get("ok")) and bool(calibration["ok"]) and bool(pairwise["ok"])
    return {
        "name": name,
        "games_per_checkpoint": int(games_per_checkpoint),
        "checkpoint_count": int(checkpoint_count),
        "curriculum": curriculum,
        "belief_calibration": calibration,
        "pairwise": pairwise,
        "useful": useful,
    }


def select_smallest_useful_cadence(
    candidates: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """First useful cadence ordered by ascending games_per_checkpoint."""
    ordered = sorted(
        candidates,
        key=lambda c: (
            int(c.get("games_per_checkpoint") or 0),
            int(c.get("checkpoint_count") or 0),
            str(c.get("name") or ""),
        ),
    )
    for row in ordered:
        if bool(row.get("useful")):
            return {
                "selected": dict(row),
                "selected_name": row.get("name"),
                "useful_found": True,
            }
    return {
        "selected": None,
        "selected_name": None,
        "useful_found": False,
        "reason": "no_tested_cadence_met_curriculum_calibration_and_pairwise_rules",
    }
