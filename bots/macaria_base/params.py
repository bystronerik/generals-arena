"""Every macaria knob in one frozen dataclass, plus a research-only override.

Two reasons this file exists rather than the module constants blitz keeps.

1. **One place to read the whole program's calibration.** macaria is a vendored
   heuristic core plus a search that sometimes overrides it. Those are two
   independently tunable programs sharing one 100 ms turn, and a knob that
   lives in whichever module happened to need it makes an increment's diff
   illegible. `blitz_config()` at the bottom translates back into the core's
   own `frozen=True` config type, so the core keeps its own vocabulary and
   this file stays the single index.

2. **A sweep must vary one knob without forking a content hash per cell.**
   `MACARIA_TUNE` (JSON, environment) overrides fields at construction. It is
   research-only plumbing: unset — which is what the competition runner and
   every stored game does — the shipped defaults below are the whole program.
   A malformed value falls back to the defaults rather than crashing a match,
   because RULES.md §08 forfeits a bot that crashes and a sweep typo must not
   be able to spend a game.

The rule this file enforces: **no behavioural constant lives outside it.** That
includes the ones upstream blitz kept as module constants (`DEATHTOUCH_TURN`,
`CHASE_DEFEND_FROM`, the never-reserve strategy context, the `or 50` wave
period, the `// 2` general regen) — see `blitz_core`'s provenance docstring.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields, replace

from blitz_core import BlitzConfig

TUNE_ENV = "MACARIA_TUNE"


@dataclass(frozen=True)
class MacariaParams:
    """Shipped values. Anything not re-measured here carries blitz's."""

    # ------------------------------------------------ blitz core: opening
    #: First turn of the assault phase — the first land-bonus turn.
    blitz_opening_end: int = 50
    #: How far a stalled chain head may walk over owned land to find fresh
    #: neutral cells before the chain is abandoned.
    blitz_chain_detour: int = 3

    # --------------------------------------------- blitz core: wave cycle
    #: How long a wave concentrates army before it launches regardless.
    blitz_rally_ticks: int = 14
    #: Cap on the expansion stretch between a dead wave and the next rally.
    blitz_rebuild_ticks: int = 50
    #: Rally by walking the stack forward through our own surplus instead of
    #: ferrying stacks back to it.
    blitz_collection_walk: bool = True
    #: A strike stack this small has done its damage; the wave is over.
    blitz_spent_army: int = 5

    # ------------------------------------------------ blitz core: assault
    #: Absolute floor on a worthwhile strike stack.
    blitz_min_strike_floor: int = 18
    #: Stack target as a fraction of the opponent's *mobile* army.
    blitz_strike_ratio: float = 0.7
    #: Army budgeted per hop of the approach.
    blitz_travel_margin: float = 1.0
    #: Max turns a blocked strike spends reinforcing before giving up.
    blitz_feed_budget: int = 20
    #: Switch the strike stack to another cell once that cell holds this
    #: multiple of the current stack.
    blitz_restack_ratio: float = 1.5

    # ---------------------------------------------- blitz core: targeting
    #: Re-estimate a fogged enemy general at most this often.
    blitz_retarget_interval: int = 20
    #: How far behind their visible front line to look for their general.
    blitz_contact_radius: int = 6

    # ------------------------------------------------ blitz core: defence
    #: Only enemy cells within this BFS distance of our general count as a
    #: home threat — blitz would rather race than turtle.
    blitz_defense_dist: int = 12
    #: Extra army wanted at home on top of the incoming threat.
    blitz_defense_margin: int = 2

    # ------------------------------------------------ blitz core: pathing
    #: Per-army discount for routing the strike through our own cells.
    blitz_own_tile_bonus: float = 0.05
    #: Army count past which extra army stops making a cell cheaper.
    blitz_own_tile_bonus_cap: int = 8
    #: Cost of crossing neutral/fog land (slightly worse than own land).
    blitz_neutral_cost: float = 1.15
    #: Extra path cost per defending army on an enemy cell.
    blitz_enemy_cost_per_army: float = 0.03

    # ---------------------------- blitz core: upstream module constants
    #: From this turn any adjacent stack of 2+ takes a general (RULES.md §07).
    #: The engine pins the same threshold
    #: (`GeneralsEnv(mode="competition").deathtouch_turn`).
    blitz_deathtouch_turn: int = 800
    #: Shortly before deathtouch, kill any enemy cell that reaches our general.
    blitz_chase_defend_from: int = 780
    #: Turn the general starts holding a reserve back. Upstream blitz passes a
    #: turn no game reaches — it empties its home by design. Kept reachable.
    blitz_reserve_opening_end: int = 1 << 30
    #: Map-generation floor on general separation (RULES.md §01), the belief
    #: prior over where the enemy general can be.
    blitz_min_general_distance: int = 17
    #: Turns between land bonuses (RULES.md §04).
    blitz_land_bonus_period: int = 50
    #: Turns per general/castle army tick (RULES.md §04).
    blitz_general_growth_period: int = 2

    # ================================================================ MCTS
    # Owned by search.py. Anything the search reads is declared here.

    #: Master switch. `MACARIA_TUNE='{"mcts_enabled": false}'` is the control
    #: arm that separates the search's contribution from the vendored core's.
    mcts_enabled: bool = True

    #: Ceiling on the **whole move** — heuristic core plus search —
    #: milliseconds. RULES.md §08 allows 150 ms and forfeits at 50 late
    #: replies; 100 keeps 50 ms of reserve for interpreter jitter and the
    #: stdio round trip. The search's real deadline is
    #: `min(mcts_budget_ms, mcts_latency_cap_ms - already spent this turn)`,
    #: so a slow core shrinks the search rather than pushing the turn over.
    mcts_latency_cap_ms: float = 100.0

    #: Self-enforced wall-clock cap on the search alone, milliseconds. Below
    #: the whole-move cap so that a fast core does not license a long search.
    mcts_budget_ms: float = 55.0

    # --- knobs below this line are the search's own; see search.py ---

    # ------------------------------------------------- search: scoping
    #: T2 home tactical: a visible enemy stack of `mcts_min_enemy_army`+
    #: within this Manhattan distance of our general fires the search.
    mcts_scope_radius: int = 8
    #: T4 contested contact: a qualifying stack within this Manhattan
    #: distance of the core move's source or destination (post-opening).
    mcts_scope_radius_move: int = 3
    #: T1 finish window: with the enemy general's location known, one of our
    #: 2+-army cells within this Manhattan distance of it fires the search.
    mcts_finish_radius: int = 12
    #: T3 endgame: from this turn the search fires on any visible contact
    #: (2+ army, the lethal §07 minimum). Matches `blitz_chase_defend_from`.
    mcts_endgame_from: int = 780
    #: Smallest enemy stack that counts as tactical contact outside the
    #: endgame — 1-army cells are pinned, and tiny stacks are noise.
    mcts_min_enemy_army: int = 4

    # --------------------------------------------- search: move matrix
    #: Root candidate moves (ours); the core's move and PASS always survive
    #: the cap. Rows beyond the cap are the least tactical ones.
    mcts_max_root_moves: int = 10
    #: Opponent replies per root move, PASS and the two row-specific
    #: dangers (chase our source, attack our destination) included.
    mcts_opp_replies: int = 6
    #: Visible enemy stacks (largest first) that generate attack rows,
    #: reply columns, and the paranoid marcher.
    mcts_enemy_stacks: int = 3
    #: Iterative-deepening ladder of rollout horizons, in plies. Each pass
    #: re-evaluates the whole matrix; the deepest COMPLETED pass decides,
    #: and a pass cut by the deadline is discarded whole.
    mcts_horizons: tuple = (4, 6, 8, 10)

    # ----------------------------------------------- search: evaluation
    #: Leaf eval: weight of the army differential. Combat is diff-neutral
    #: (equal armies die on both sides), so this prices lost stacks and won
    #: garrisons, not aggression itself.
    mcts_w_army: float = 1.0
    #: Leaf eval: weight of the land differential (a won fight is
    #: army-neutral and land-positive; this is what makes trades attractive).
    mcts_w_land: float = 0.25
    #: Leaf eval: penalty per army of the biggest enemy stack near our
    #: general in excess of the general's own garrison.
    mcts_w_threat: float = 1.5
    #: Manhattan radius of the general-threat scan.
    mcts_threat_radius: int = 3
    #: Leaf eval: value of one BFS hop of remaining distance between our
    #: walker and the strike target. Without it the paranoid min prefers
    #: any local trade over pushing past a defender and the assault loses
    #: its tempo one override at a time.
    mcts_w_progress: float = 0.5
    #: Terminal value of a simulated win (loss is its negation); plies-to-
    #: outcome shift it so faster wins and slower losses rank higher.
    mcts_win_score: float = 1000000.0
    #: Value of a simulated mutual capture/touch — a DRAW under the
    #: deathtouch modifier, never a win.
    mcts_draw_score: float = 0.0

    # ---------------------------------------------- search: time control
    #: The best row must beat the core's own row by MORE than this to
    #: override; ties and noise keep the core's move. Measured at 1.0 the
    #: search overrode quiet rebuild moves often enough to starve the wave
    #: economy; 3.0 (three army-units of simulated advantage) keeps the
    #: overrides that decide fights and drops the marginal ones.
    mcts_override_margin: float = 3.0
    #: Decline to search — or to start another deepening pass — with less
    #: than this many milliseconds left before the deadline.
    mcts_min_headroom_ms: float = 3.0
    #: Hard-draw turn (RULES.md §07); rollouts stop simulating past it.
    mcts_draw_turn: int = 1200


def load_params(base: MacariaParams | None = None) -> MacariaParams:
    """Shipped params, with `MACARIA_TUNE` applied when a sweep set it."""
    params = base or MacariaParams()
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


def blitz_config(params: MacariaParams):
    """A `BlitzConfig` for the vendored core, carrying macaria's values."""
    return BlitzConfig(
        opening_end=params.blitz_opening_end,
        chain_detour=params.blitz_chain_detour,
        rally_ticks=params.blitz_rally_ticks,
        rebuild_ticks=params.blitz_rebuild_ticks,
        collection_walk=params.blitz_collection_walk,
        spent_army=params.blitz_spent_army,
        min_strike_floor=params.blitz_min_strike_floor,
        strike_ratio=params.blitz_strike_ratio,
        travel_margin=params.blitz_travel_margin,
        feed_budget=params.blitz_feed_budget,
        restack_ratio=params.blitz_restack_ratio,
        retarget_interval=params.blitz_retarget_interval,
        contact_radius=params.blitz_contact_radius,
        defense_dist=params.blitz_defense_dist,
        defense_margin=params.blitz_defense_margin,
        own_tile_bonus=params.blitz_own_tile_bonus,
        own_tile_bonus_cap=params.blitz_own_tile_bonus_cap,
        neutral_cost=params.blitz_neutral_cost,
        enemy_cost_per_army=params.blitz_enemy_cost_per_army,
        deathtouch_turn=params.blitz_deathtouch_turn,
        chase_defend_from=params.blitz_chase_defend_from,
        reserve_opening_end=params.blitz_reserve_opening_end,
        min_general_distance=params.blitz_min_general_distance,
        land_bonus_period=params.blitz_land_bonus_period,
        general_growth_period=params.blitz_general_growth_period,
    )
