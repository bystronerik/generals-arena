"""Opponent observation model for the migrated strategy bots.

Grid-native port of generals-bot ``bots/common/oppmodel.py``. Feed it the
observation every turn; it accumulates the observable signals bots use for
threat detection, counter windows, and (proteus) strategy classification.

``update`` is idempotent per turn, so a composite bot may share one model
across several strategy cores and warm them all safely.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from _common.strategy_common import locate_own_general
from _common.tactics import UNREACHABLE, multi_bfs, visible_enemy_tiles


@dataclass
class OpponentModel:
    """Rolling observation of the 1v1 opponent."""

    # Time series (per turn, appended on update()).
    turns: list[int] = field(default_factory=list)
    opp_land: list[int] = field(default_factory=list)
    opp_army: list[int] = field(default_factory=list)
    my_land: list[int] = field(default_factory=list)
    my_army: list[int] = field(default_factory=list)

    # Signals.
    closest_enemy_dist: int = UNREACHABLE  # min BFS dist enemy cell -> my general, ever
    closest_enemy_dist_now: int = UNREACHABLE
    biggest_enemy_stack: int = 0           # largest single visible enemy army ever
    biggest_enemy_stack_now: int = 0
    enemy_castles_seen: int = 0            # visible enemy-owned castles (max ever)
    first_contact_turn: int | None = None  # first turn an enemy cell was visible
    my_general: tuple[int, int] | None = None

    def update(self, obs) -> None:
        """Record this turn's observations (no-op if already recorded)."""
        if self.turns and self.turns[-1] == obs.turn:
            return
        self.turns.append(obs.turn)
        self.opp_land.append(obs.opp_land)
        self.opp_army.append(obs.opp_army)
        self.my_land.append(obs.my_land)
        self.my_army.append(obs.my_army)

        if self.my_general is None:
            self.my_general = locate_own_general(obs)

        enemy_cells = visible_enemy_tiles(obs)
        if enemy_cells:
            if self.first_contact_turn is None:
                self.first_contact_turn = obs.turn
            self.biggest_enemy_stack_now = max(
                obs.army_grid[r][c] for r, c in enemy_cells
            )
            self.biggest_enemy_stack = max(
                self.biggest_enemy_stack, self.biggest_enemy_stack_now
            )
            if self.my_general is not None:
                dist = multi_bfs(obs, [self.my_general])
                d = min(dist[r][c] for r, c in enemy_cells)
                self.closest_enemy_dist_now = d
                self.closest_enemy_dist = min(self.closest_enemy_dist, d)
        else:
            self.biggest_enemy_stack_now = 0
            self.closest_enemy_dist_now = UNREACHABLE

        castles_now = sum(
            1
            for r, c in enemy_cells
            if obs.type_grid[r][c] == 3
        )
        self.enemy_castles_seen = max(self.enemy_castles_seen, castles_now)

    # ------------------------------------------------------------- signals
    def opp_tile_rate(self, window: int = 50) -> float:
        """Opponent land gained per turn over the last ``window`` turns."""
        if len(self.turns) < 2:
            return 0.0
        n = min(window, len(self.opp_land) - 1)
        return (self.opp_land[-1] - self.opp_land[-1 - n]) / max(n, 1)

    def opp_total_drop(self, window: int = 20) -> int:
        """How much opponent total army fell within the last window (>=0)."""
        if len(self.opp_army) < 2:
            return 0
        n = min(window, len(self.opp_army) - 1)
        recent = self.opp_army[-1 - n:]
        return max(recent) - self.opp_army[-1]

    def under_attack(self, threshold_dist: int = 7) -> bool:
        """An enemy cell is currently near our general."""
        return self.closest_enemy_dist_now <= threshold_dist

    def army_ratio(self) -> float:
        """my_army / opp_army (inf-safe)."""
        if not self.opp_army or self.opp_army[-1] <= 0:
            return 1.0
        return self.my_army[-1] / self.opp_army[-1]

    def opponent_mobile(self) -> int:
        """Opponent army free to move (total minus one pinned per cell)."""
        if not self.opp_army:
            return 0
        return max(0, self.opp_army[-1] - self.opp_land[-1])
