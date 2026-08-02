"""Every behavioural constant for sosipolis. Named thresholds only."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    LATENCY_CAP_MS: int = 100
    SEARCH_BUDGET_MS: int = 90
    CONTACT_BUDGET_MS: int = 90
    STRIKE_BUDGET_MS: int = 90
    FIRST_MOVE_GRACE_MS: int = 9000
    MIN_GENERAL_DISTANCE: int = 17
    SECTION_ROWS: int = 3
    SECTION_COLS: int = 3
    POCKET_MAX_CELLS: int = 12
    CONTACT_REWEIGHT: float = 0.80
    CONTACT_SECTOR_FOCUS: float = 1.0
    TARGET_CONTACT_TURN: int = 82
    TARGET_SIGHT_TURN: int = 182
    STRIKE_TOWARD_BIAS: float = 0.92
    GATHER_WAVE_HINT: int = 6
    DEATHTOUCH_TURN: int = 800
    FINISH_MARGIN: int = 1
    MCTS_C: float = 1.2
    MCTS_MAX_ROOT: int = 16
    MCTS_ROLLOUT_DEPTH: int = 8
    DEFENSE_RADIUS: int = 3
    CORRIDOR_WIDTH_MAX: int = 1
    HOME_BANK_SEARCH: int = 4
    HOME_BANK_CONTACT: int = 5
    HOME_BANK_STRIKE: int = 8
    DEFENSE_WEIGHT_SEARCH: float = 8.0
    DEFENSE_WEIGHT_CONTACT: float = 10.0
    DEFENSE_WEIGHT_STRIKE: float = 16.0
    CONTACT_STAGE_STACK: int = 35
    CONTACT_ASSAULT_STACK: int = 55
    STRIKE_LAND_ROOT_SLOTS: int = 2
    STRIKE_GATHER_WAVES_HINT: int = 4
    STRIKE_TIP_HOLD: int = 8
    STRIKE_PATH_BUFFER: int = 1
    STRIKE_TIP_FEED_BONUS: float = 250.0
    SEARCH_LAND_BONUS: float = 80.0
    SEARCH_FOG_BONUS: float = 50.0
    SEARCH_ENEMY_BONUS: float = 70.0
    SEARCH_HUNT_BONUS: float = 90.0
    CONTACT_ENEMY_BONUS: float = 45.0
    CONTACT_LAND_BONUS: float = 40.0
    CONTACT_CANDIDATE_BONUS: float = 80.0
    # Belief hunt (Macaria-style, local to map_memory + contact_mcts).
    HUNT_INTERVAL: int = 8
    HUNT_REVEAL_RADIUS: int = 2
    HUNT_PRIOR_DECAY: float = 0.35
    HUNT_TRAVEL_DECAY: float = 0.05
    HUNT_CONTACT_RADIUS: int = 6
    HUNT_STEP_BONUS: float = 220.0
    CASTLE_MAX: int = 0
    CASTLE_START_TURN: int = 10
    CASTLE_KEEP: int = 3
    CASTLE_MIN_LAND: int = 5
    CASTLE_ENEMY_CLEAR: int = 5
    CASTLE_ABORT_TURN: int = 100
    BUILD_BASE_COST: int = 35
    BUILD_SURCHARGE_CAP: int = 14
    BUILD_SURCHARGE_PER_STEP: int = 2


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
