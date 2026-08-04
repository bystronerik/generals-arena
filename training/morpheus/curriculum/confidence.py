"""Pilot non-degenerate WDL confidence rule for curriculum promotion."""

from __future__ import annotations

import math
from typing import Any


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (0.0, 1.0)
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1)")
    # z for common levels; fall back to normal approximation via erfinv.
    z_table = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = z_table.get(round(confidence_level, 2))
    if z is None:
        # P(|Z| < z) = confidence_level → erf(z/√2) = confidence_level
        z = math.sqrt(2.0) * _erfinv(confidence_level)
    n = float(trials)
    p = float(successes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _erfinv(y: float) -> float:
    """Inverse error function for confidence levels not in the table."""
    # Winitzki approximation.
    a = 0.147
    sgn = 1.0 if y >= 0 else -1.0
    ln = math.log(1.0 - y * y)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return sgn * math.sqrt(math.sqrt(first * first - ln / a) - first)


DEFAULT_CONFIDENCE_RULE: dict[str, Any] = {
    "name": "non_degenerate_wdl_by_class",
    "selected_before_main_run": True,
    "interval_method": "wilson",
    "confidence_level": 0.95,
    "min_samples_per_active_class": 32,
    "non_degenerate_wdl": {
        "min_lower_win_rate": 0.05,
        "max_upper_win_rate": 0.95,
        "require_both_wins_and_losses": True,
    },
    "full_start_decisive_rate": {
        "min_samples": 50,
        "min_lower_bound": 0.05,
    },
    "held_out_arena": {
        "require_pairwise_contrast": True,
        "decision_rule": "docs/arena/decision-rule.md",
    },
    "promotion_default": (
        "Do not move sampling weight toward an earlier class until every "
        "active class has a non-degenerate WDL target under this rule."
    ),
    "evidence_required": [
        "wdl_intervals_by_class",
        "full_start_decisive_rate",
        "held_out_arena_strength",
    ],
    "rejected_evidence": ["training_loss_alone"],
}


def class_wdl_is_non_degenerate(
    wins: int,
    losses: int,
    draws: int = 0,
    *,
    rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one class against the pilot confidence rule."""
    rule = rule or DEFAULT_CONFIDENCE_RULE
    trials = int(wins) + int(losses) + int(draws)
    min_n = int(rule["min_samples_per_active_class"])
    nd = rule["non_degenerate_wdl"]
    level = float(rule["confidence_level"])
    lo, hi = wilson_interval(int(wins), trials, confidence_level=level)
    both = (int(wins) > 0 and int(losses) > 0) if nd["require_both_wins_and_losses"] else True
    ok = (
        trials >= min_n
        and both
        and lo >= float(nd["min_lower_win_rate"])
        and hi <= float(nd["max_upper_win_rate"])
    )
    return {
        "ok": ok,
        "trials": trials,
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "wilson": {"low": lo, "high": hi, "confidence_level": level},
        "min_samples": min_n,
    }


def may_advance_toward_earlier_class(
    class_wdl: dict[int, dict[str, int]],
    *,
    active_classes: list[int],
    rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Promotion gate: every active class must be non-degenerate."""
    rule = rule or DEFAULT_CONFIDENCE_RULE
    per_class = {}
    for cid in active_classes:
        stats = class_wdl.get(int(cid), {"wins": 0, "losses": 0, "draws": 0})
        per_class[int(cid)] = class_wdl_is_non_degenerate(
            int(stats.get("wins", 0)),
            int(stats.get("losses", 0)),
            int(stats.get("draws", 0)),
            rule=rule,
        )
    ok = all(v["ok"] for v in per_class.values()) if per_class else False
    return {
        "ok": ok,
        "promotion_default": rule["promotion_default"],
        "classes": per_class,
    }
