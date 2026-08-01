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
    #: Standing home guard, as a fraction of the opponent's mobile army.
    #: **Measured and rejected.** Over 400 paired games against cm_hunter:
    #: 0.0 -> 0.922, 0.35 -> 0.850 (paired +5/-34), 0.60 -> 0.770 (+0/-61).
    #: Monotone in the dose, so this is the guard and not the noise: blitz
    #: wins by racing, and army parked at home is army that loses the race —
    #: it saves the general it was sized for and then loses the game it was
    #: taken from. Kept at 0 (upstream blitz) and left reachable, because a
    #: negative that specific is worth being able to re-run.
    #: See `blitz_core.BlitzConfig.guard_ratio`.
    blitz_guard_ratio: float = 0.0
    blitz_guard_from: int = 60
    blitz_guard_until: int = 320
    blitz_guard_cap: int = 40

    # ------------------------------------------------- boom core (BoomParams)
    boom_threat_dist: int = 12
    boom_intercept_dist: int = 8
    boom_guard_margin: int = 2
    boom_defenders_counted: int = 2
    boom_commit_turn: int = 200

    # ------------------------------------------------------------------ MCTS
    #: Master switch. `{"mcts_enabled": false}` is the control arm for the
    #: search increment (`search.py` explains why the search is scoped rather
    #: than global). On, measured: 0.922 -> 0.943 against cm_hunter over 400
    #: paired games, paired flips +22/-14.
    mcts_enabled: bool = True
    #: Self-enforced wall-clock cap on the search alone, milliseconds.
    mcts_budget_ms: float = 40.0
    #: Ceiling on the whole move — heuristic plus search. The search's real cap
    #: is `min(budget, this - already spent)`, so a slow core shrinks the
    #: search instead of pushing the turn past RULES.md §08's 150 ms.
    mcts_latency_cap_ms: float = 110.0
    #: Search only when an enemy stack is this close to our general, or one of
    #: ours this close to a *known* enemy general.
    mcts_window: int = 8
    #: Plies simulated per rollout; both seats move each ply. Long enough that
    #: a fight at the general usually *resolves* into a capture inside the
    #: rollout: at 12 plies almost every playout ended on `_evaluate`, and a
    #: heuristic value is exactly what the first measured version got wrong.
    mcts_rollout_depth: int = 24
    #: Iteration ceiling, so the search cannot spend a whole budget it did not
    #: need on a machine faster than the one it was calibrated on.
    mcts_max_iters: int = 400
    #: UCT exploration constant.
    mcts_c: float = 1.2
    #: Root shortlist size. Child 0 is always the core's own move. 8 and 12
    #: measured within noise of each other (0.943 / 0.945); 12 is kept because
    #: it costs nothing — the budget, not the shortlist, is what binds.
    mcts_root_moves: int = 12
    #: Playout randomisation. At 0 the greedy policy makes every rollout from
    #: a root identical and the search collapses to a single line.
    mcts_epsilon: float = 0.25

    # --------------------------------------------------------- deathtouch (§07)
    #: Turn the third core takes over. RULES.md §07's own threshold; the engine
    #: pins it too (`GeneralsEnv(mode="competition").deathtouch_turn`).
    deathtouch_from: int = 800
    #: `{"deathtouch_enabled": false}` is the control arm for the endgame
    #: increment. On, measured over 400 paired games against the four
    #: draw-heavy opponents: draws 37 -> 17, wins 318 -> 336, losses 45 -> 47.
    deathtouch_enabled: bool = True
    deathtouch_garrison_margin: int = 2
    deathtouch_all_in_from: int = 1100


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
        guard_ratio=params.blitz_guard_ratio,
        guard_from=params.blitz_guard_from,
        guard_until=params.blitz_guard_until,
        guard_cap=params.blitz_guard_cap,
    )


def deathtouch_config(params: YankeeParams):
    """A `DeathtouchConfig` carrying yankee's values."""
    from yankee.deathtouch import DeathtouchConfig

    return DeathtouchConfig(
        touch_turn=params.deathtouch_from,
        garrison_margin=params.deathtouch_garrison_margin,
        all_in_from=params.deathtouch_all_in_from,
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
