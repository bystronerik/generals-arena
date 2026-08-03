"""Every behavioural constant for sosipolis. Named thresholds only."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    LATENCY_CAP_MS: int = 100
    SEARCH_BUDGET_MS: int = 90
    CONTACT_BUDGET_MS: int = 90
    STRIKE_BUDGET_MS: int = 90
    GATHER_BUDGET_MS: int = 40
    WAVE_BUDGET_MS: int = 80
    STRIKE_MARCH_BUDGET_MS: int = 55
    FIRST_MOVE_GRACE_MS: int = 9000
    MIN_GENERAL_DISTANCE: int = 17
    SECTION_ROWS: int = 3
    SECTION_COLS: int = 3
    POCKET_MAX_CELLS: int = 12
    CONTACT_REWEIGHT: float = 0.80
    CONTACT_SECTOR_FOCUS: float = 1.0
    TARGET_CONTACT_TURN: int = 82
    TARGET_SIGHT_TURN: int = 182
    STRIKE_TOWARD_BIAS: float = 0.93
    GATHER_WAVE_HINT: int = 6
    DEATHTOUCH_TURN: int = 800
    FINISH_MARGIN: int = 1
    MCTS_C: float = 1.0
    MCTS_MAX_ROOT: int = 12
    MCTS_MAX_ROOT_GATHER: int = 8
    MCTS_MAX_ROOT_WAVE: int = 12
    MCTS_MAX_ROOT_STRIKE: int = 4
    MCTS_ROLLOUT_DEPTH: int = 8
    # Mod-50 gather/wave clock (Kubic MEASURED).
    GATHER_PHASE_LO: int = 10
    GATHER_PHASE_HI: int = 27
    # Opening tempo (Kubic MEASURED).
    OPEN_END: int = 50
    OPEN_FLOOD_START: int = 12  # was 24 — flood earlier for land tempo
    OPEN_PULSE_TICKS: tuple = (3, 6, 9)
    OPEN_TILES_T50_LO: int = 20
    OPEN_TILES_T50_HI: int = 25
    # Rare recall (RECALL_PROX_D UNKNOWN; candidate 3).
    RECALL_PROX_D: int = 3
    TIP_AT_SIGHT_FLOOR: int = 10
    DEFENSE_RADIUS: int = 3
    CORRIDOR_WIDTH_MAX: int = 1
    HOME_BANK_SEARCH: int = 1
    HOME_BANK_CONTACT: int = 1
    HOME_BANK_STRIKE: int = 1
    DEFENSE_WEIGHT_SEARCH: float = 4.0
    DEFENSE_WEIGHT_CONTACT: float = 6.0
    DEFENSE_WEIGHT_STRIKE: float = 8.0
    CONTACT_STAGE_STACK: int = 35
    CONTACT_ASSAULT_STACK: int = 15
    STRIKE_LAND_ROOT_SLOTS: int = 0
    STRIKE_GATHER_WAVES_HINT: int = 4
    STRIKE_TIP_HOLD: int = 8
    STRIKE_PATH_BUFFER: int = 2
    STRIKE_REGEN_SLACK: int = 1
    STRIKE_MIN_TIP: int = 23
    STRIKE_TIP_ARMY_FRAC: float = 0.40
    STRIKE_TIP_MAX_DIST: int = 14
    STRIKE_TIP_FEED_BONUS: float = 400.0
    SEARCH_LAND_BONUS: float = 160.0  # was 110 — push land before contact
    SEARCH_FOG_BONUS: float = 120.0  # was 85
    SEARCH_ENEMY_BONUS: float = 95.0  # was 70
    SEARCH_HUNT_BONUS: float = 160.0
    CONTACT_ENEMY_BONUS: float = 420.0  # was 45 — must beat HUNT_STEP fog march
    CONTACT_LAND_BONUS: float = 55.0
    CONTACT_CANDIDATE_BONUS: float = 100.0
    CONTACT_CHASE_STEP_BONUS: float = 750.0  # tip step toward live army
    # Belief hunt (Macaria-style, local to map_memory + contact_mcts).
    HUNT_INTERVAL: int = 4
    HUNT_REVEAL_RADIUS: int = 2
    HUNT_PRIOR_DECAY: float = 0.35
    HUNT_TRAVEL_DECAY: float = 0.04
    HUNT_CONTACT_RADIUS: int = 10
    HUNT_STEP_BONUS: float = 320.0
    # Contact belief: promote near enemy_seen (toward / through their army).
    # Replaces old boost of high d_foot. Form: 1 + NEAR / (1 + d_foot).
    CONTACT_BELIEF_NEAR_FOOT: float = 2.0
    # First-contact axis: late flanks must not yank the original hunt, but the
    # tip must still fight visible enemy army on the approach.
    CONTACT_PRIMARY_PATH_WINDOW: int = 40
    CONTACT_AXIS_BONUS: float = 1.5
    CONTACT_AXIS_MIN_T: float = 0.55
    CONTACT_AXIS_OFF_PENALTY: float = 0.8
    CONTACT_AXIS_LATERAL: float = 0.06
    CONTACT_AXIS_RECOVER: bool = True
    CONTACT_AXIS_RECOVER_LATERAL: float = 8.0  # only snap if this far off-axis
    CONTACT_FLANK_DELTA_SCALE: float = 0.25
    CONTACT_CHASE_VISIBLE: float = 2.0  # belief boost near live enemy tiles
    CONTACT_CHASE_VISIBLE_RADIUS: int = 3
    # Chase fog ranking: cells per step of drift off the home→source ray.
    CONTACT_CHASE_LATERAL: float = 1.0
    # Fused contact evidence: every contact point votes on the general's cell.
    CONTACT_EVIDENCE_RADIUS: int = 4
    CONTACT_DENSITY_WEIGHT: float = 3.0
    CONTACT_OPEN_WEIGHT: float = 1.5
    # Early contact → trust axis more; late contact → weaker axis lock.
    CONTACT_AXIS_EARLY_TURN: int = 55
    CONTACT_AXIS_LATE_TURN: int = 100
    # ContactMCTS owns post-contact probe target (Phase 1 / Phase 2).
    CONTACT_PATH_MODE: str = "shallow"  # "shallow" | "macro_mcts"
    # Belief refresh costs ~4.5 ms median / 14 ms worst on a 21x20 board. At 3
    # the deadline expired inside _refresh_cache on 67% of contact turns and
    # the commitment was never re-examined. The move cap is 100 ms.
    CONTACT_PREP_BUDGET_MS: int = 20
    CONTACT_PATH_SCORE_BUDGET_MS: int = 8
    CONTACT_COMMIT_MIN_TURNS: int = 12
    CONTACT_SWITCH_RATIO: float = 1.35
    CONTACT_SWITCH_MARGIN: float = 0.08
    CONTACT_INVALIDATE_DISTANCE: int = 2
    CONTACT_MAX_MACROS: int = 6
    CONTACT_CLUSTER_RADIUS: int = 6
    CONTACT_OPP_SIDE_BFS: int = 8
    CONTACT_FORCE_OPP_SWITCH: bool = False
    CONTACT_ENEMY_OBS_AGE: int = 40
    CONTACT_MACRO_DEPTH: int = 3
    CONTACT_MACRO_ROOTS: int = 9
    # After a spent wave: continue past tip; do not free-pick a corner macro.
    CONTACT_EXTEND_STEPS: int = 4
    CONTACT_HOLD_GATHER: bool = True
    CONTACT_REPLACE_JUMP_MAX: int = 8
    CASTLE_MAX: int = 4
    CASTLE_START_TURN: int = 116
    CASTLE_KEEP: int = 1
    CASTLE_MIN_LAND: int = 20
    CASTLE_ENEMY_CLEAR: int = 2
    CASTLE_ABORT_TURN: int = 900
    CASTLE_SPACING: int = 7
    CASTLE_FRONTIER_MAX: int = 2
    BUILD_BASE_COST: int = 35
    BUILD_SURCHARGE_CAP: int = 14
    BUILD_SURCHARGE_PER_STEP: int = 2
    CHAIN_CONTINUE_PRIOR: float = 3.5
    CHAIN_BREAK_PRIOR: float = 0.25
    GENERAL_PULL_ODD_PREF: bool = True


PARAMS = Params()

# Wire protocol type codes.
T_FOG = 0
T_PLAIN = 1
T_MOUNTAIN = 2
T_CASTLE = 3
T_GENERAL = 4
T_STRUCT_FOG = 5

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PASS = (1, 0, 0, 0, 0)
BUILD_PASS = 2  # action[0] == 2 means build
