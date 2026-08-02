"""Every behavioural constant for sosipolis. Named thresholds only."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    LATENCY_CAP_MS: int = 100
    SEARCH_BUDGET_MS: int = 70
    CONTACT_BUDGET_MS: int = 75
    STRIKE_BUDGET_MS: int = 80
    FIRST_MOVE_GRACE_MS: int = 9000
    MIN_GENERAL_DISTANCE: int = 17
    SECTION_ROWS: int = 3
    SECTION_COLS: int = 3
    POCKET_MAX_CELLS: int = 12
    CONTACT_REWEIGHT: float = 0.80
    CONTACT_SECTOR_FOCUS: float = 0.95
    TARGET_CONTACT_TURN: int = 82
    TARGET_SIGHT_TURN: int = 182
    STRIKE_TOWARD_BIAS: float = 0.84
    GATHER_WAVE_HINT: int = 4
    DEATHTOUCH_TURN: int = 800
    FINISH_MARGIN: int = 2
    MCTS_C: float = 1.2
    MCTS_MAX_ROOT: int = 12
    MCTS_ROLLOUT_DEPTH: int = 8
    DEFENSE_RADIUS: int = 2
    CORRIDOR_WIDTH_MAX: int = 1
    CASTLE_MAX: int = 1
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
