"""Every yankee knob in one frozen dataclass, plus a research-only override.

Two reasons this exists rather than the module constants proteus has.

1. **One place to read the whole program's calibration.** yankee's behaviour is
   split across a classifier, a switcher and two vendored cores, each of which
   upstream configures from its own `frozen=True` dataclass. Collecting the
   knobs that this effort actually moves into one object is what makes an
   increment's diff legible: `blitz_config()` and `boom_params()` below
   translate back into the cores' own config types.

2. **A sweep needs to vary one knob without forking a hash per cell.**
   `YANKEE_TUNE` (JSON, environment) overrides fields at construction. It is
   research-only plumbing: unset — which is what the competition runner and
   every stored game does — the shipped defaults below are the whole program,
   and a malformed value falls back to them rather than crashing a match.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace

TUNE_ENV = "YANKEE_TUNE"


@dataclass(frozen=True)
class YankeeParams:
    """Shipped values. Anything not re-measured here carries proteus's."""

    # ------------------------------------------------------------- switcher
    min_confidence: float = 0.55
    leave_spine_streak: int = 25
    return_spine_streak: int = 6
    cooldown: int = 60

    # ------------------------------------------------------------ classifier
    duel_army: int = 15
    duel_deadline: int = 250
    evidence_turn: int = 60
    economy_base: float = 0.60
    economy_growth: float = 0.01

    # --------------------------------------------------------- home pressure
    pressure_radius: int = 8

    # ----------------------------------------------- blitz core (BlitzConfig)
    #: Only enemy cells this close to our general count as a home threat.
    blitz_defense_dist: int = 12
    #: Extra army wanted at home on top of the incoming threat.
    blitz_defense_margin: int = 2
    blitz_opening_end: int = 50
    blitz_rally_ticks: int = 14
    blitz_rebuild_ticks: int = 50
    blitz_min_strike_floor: int = 18
    blitz_strike_ratio: float = 0.7
    blitz_feed_budget: int = 20

    # ------------------------------------------------- boom core (BoomParams)
    boom_threat_dist: int = 12
    boom_intercept_dist: int = 8
    boom_guard_margin: int = 2
    boom_defenders_counted: int = 2
    boom_commit_turn: int = 200


def load_params(base: YankeeParams | None = None) -> YankeeParams:
    """Shipped params, with `YANKEE_TUNE` applied when a sweep set it."""
    params = base or YankeeParams()
    raw = os.environ.get(TUNE_ENV)
    if not raw:
        return params
    try:
        overrides = json.loads(raw)
    except (TypeError, ValueError):
        return params
    if not isinstance(overrides, dict):
        return params
    known = {f.name for f in fields(params)}
    clean = {k: v for k, v in overrides.items() if k in known}
    if not clean:
        return params
    try:
        return replace(params, **clean)
    except (TypeError, ValueError):
        return params


def blitz_config(params: YankeeParams):
    """A `BlitzConfig` for the vendored blitz core, carrying yankee's values."""
    from yankee.blitz_core import BlitzConfig

    return BlitzConfig(
        opening_end=params.blitz_opening_end,
        rally_ticks=params.blitz_rally_ticks,
        rebuild_ticks=params.blitz_rebuild_ticks,
        min_strike_floor=params.blitz_min_strike_floor,
        strike_ratio=params.blitz_strike_ratio,
        feed_budget=params.blitz_feed_budget,
        defense_dist=params.blitz_defense_dist,
        defense_margin=params.blitz_defense_margin,
    )


def boom_params(params: YankeeParams):
    """A `BoomParams` for the vendored boom core, carrying yankee's values."""
    from yankee.boom_core import BoomParams

    return BoomParams(
        threat_dist=params.boom_threat_dist,
        intercept_dist=params.boom_intercept_dist,
        guard_margin=params.boom_guard_margin,
        defenders_counted=params.boom_defenders_counted,
        commit_turn=params.boom_commit_turn,
    )
